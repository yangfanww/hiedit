from typing import Dict
from omegaconf import DictConfig
import os
import json
from transformers import AutoTokenizer

import math

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from nets import HiNet, RLEditNet

from editor.base import BaseEditor
from util import get_module, get_shape
from glue_eval.glue_eval import GLUEEval
import numpy as np
from nets import RLEditNet

from itertools import islice

from tqdm import tqdm
import swanlab

from util import (
    get_module,
    get_shape,
    empty_cache,
    cross_entropy,
    kl_div,
    succ_ratios
)

class HIEDIT(BaseEditor):

    def __init__(
        self,
        config: DictConfig,
        model: nn.Module
    ):
        super().__init__(
            config,
            model
        )

        self.net = nn.ModuleDict({
            str(k): RLEditNet(
                *k,
                config.editor.rank,
                config.editor.n_blocks,
                v,
                config.editor.lr
            )
            for k, v in self.shape_counter.items()
        }).to(config.editor_device)

        self.opt = torch.optim.Adam(
            self.net.parameters(),
            config.editor.meta_lr
        )

        self.hi_net = HiNet(
            shape_counter=self.shape_counter,
            hidden_size=config.editor.hidden_size,
            edit_modules=config.model.edit_modules,
            name2idx=self.name2idx,
            device=config.editor_device,
        ).to(config.editor_device)

        self.hi_opt = torch.optim.Adam(
            self.hi_net.parameters(),
            config.editor.meta_lr,
        )

    def train(self, loader: DataLoader, mode: str = None):
        """
        The training method for HiEdit.
        """
        for param in self.net.parameters():
            param.requires_grad = True
        for param in self.hi_net.parameters():
            param.requires_grad = True

        sequence_tuples = []
        max_steps = self.config.num_seq
        time_decay = self.config.editor.time_decay
        limited_loader = islice(loader, max_steps)
        for tuples in tqdm(limited_loader, desc=f"Train", ncols=100, total=max_steps):
            sequence_tuples.append(tuples)
            self.cache(tuples["edit_tuples"])
            param_shifts, mask = self.predict_param_shifts()
            selected_param_shifts = {}
            for module_idx, module_name in enumerate(self.config.model.edit_modules):
                selected_param_shifts[module_name] = param_shifts[module_name] * mask[module_idx]

            # random_mask = mask[torch.randperm(len(self.config.model.edit_modules))]
            # random_param_shifts = {}
            # for module_idx, module_name in enumerate(self.config.model.edit_modules):
            #     random_param_shifts[module_name] = param_shifts[module_name] * random_mask[module_idx]

            self.model.zero_grad()

            l2_reg_loss = 0
            for _, selected_param_shift in selected_param_shifts.items():
                l2_reg_loss += torch.sum(selected_param_shift ** 2)
            l2_reg_loss *= self.config.editor.reg_coef

            gen_losses_show = []
            self.edit_model(selected_param_shifts, False)
            tot_loss_e = 0
            for _, _tuples in enumerate(reversed(sequence_tuples)):
                loss_e = 0
                for t in _tuples["equiv_tuples"]:
                    if "old_labels" in t:
                        old_labels = t.pop("old_labels")
                    logits = self.model(**t)["logits"]
                    try:
                        t["old_labels"] = old_labels
                    except:
                        pass
                    loss = cross_entropy(logits, t["labels"])
                    loss_e += loss
                gen_losses_show.append(loss_e.item())
                tot_loss_e += (loss_e * pow(time_decay, _))

                if _ + 1 >= self.config.editor.back_depth:
                    break
            self.edit_model(selected_param_shifts, True)

            self.edit_model(param_shifts, False)
            # self.edit_model(random_param_shifts, False)
            full_tot_loss_e = 0
            for _, _tuples in enumerate(reversed(sequence_tuples)):
                loss_e = 0
                for t in _tuples["equiv_tuples"]:
                    if "old_labels" in t:
                        old_labels = t.pop("old_labels")
                    logits = self.model(**t)["logits"]
                    try:
                        t["old_labels"] = old_labels
                    except:
                        pass
                    loss = cross_entropy(logits, t["labels"])
                    loss_e += loss
                full_tot_loss_e += (loss_e * pow(time_decay, _))

                if _ + 1 >= self.config.editor.back_depth:
                    break
            self.edit_model(param_shifts, True)
            # self.edit_model(random_param_shifts, True)

            loc_losses_show = []
            tot_loss_loc = 0
            for _, _tuples in enumerate(reversed(sequence_tuples)):
                loss_loc = 0
                for t in _tuples["unrel_tuples"]:
                    if "old_labels" in t:
                        old_labels = t.pop("old_labels")
                    with torch.no_grad():
                        refer_logits = self.model(**t)["logits"]
                    self.edit_model(selected_param_shifts, False)
                    logits = self.model(**t)["logits"]
                    try:
                        t["old_labels"] = old_labels
                    except:
                        pass
                    loss = kl_div(
                        refer_logits,
                        logits,
                        t["labels"]
                    )
                    loss_loc += (self.config.editor.loc_coef * loss)
                    self.edit_model(selected_param_shifts, True)
                loc_losses_show.append(loss_loc.item())
                tot_loss_loc += (loss_loc * pow(time_decay, _))

                if _ + 1 >= self.config.editor.back_depth:
                    break

            full_tot_loss_loc = 0
            for _, _tuples in enumerate(reversed(sequence_tuples)):
                loss_loc = 0
                for t in _tuples["unrel_tuples"]:
                    if "old_labels" in t:
                        old_labels = t.pop("old_labels")
                    with torch.no_grad():
                        refer_logits = self.model(**t)["logits"]
                    self.edit_model(param_shifts, False)
                    # self.edit_model(random_param_shifts, False)
                    logits = self.model(**t)["logits"]
                    try:
                        t["old_labels"] = old_labels
                    except:
                        pass
                    loss = kl_div(
                        refer_logits,
                        logits,
                        t["labels"]
                    )
                    loss_loc += (self.config.editor.loc_coef * loss)
                    self.edit_model(param_shifts, True)
                    # self.edit_model(random_param_shifts, True)
                loc_losses_show.append(loss_loc.item())
                full_tot_loss_loc += (loss_loc * pow(time_decay, _))

                if _ + 1 >= self.config.editor.back_depth:
                    break

            loss_low = l2_reg_loss + tot_loss_e + tot_loss_loc
            loss_hi = loss_low - full_tot_loss_e.detach() - full_tot_loss_loc.detach()
            
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.net.parameters():
                param.requires_grad = False
            for param in self.hi_net.parameters():
                param.requires_grad = True
            loss_hi.backward(retain_graph=True)

            for param in self.hi_net.parameters():
                param.requires_grad = False
            for param in self.model.parameters():
                param.requires_grad = True
            loss_low.backward()

            for param in self.hi_net.parameters():
                param.requires_grad = False
            for param in self.net.parameters():
                param.requires_grad = True
            self.edit_model(selected_param_shifts, False)

            self.update_hinet(update=False)
            swanlab.log({
                f"gen_loss_hi": np.mean(gen_losses_show),
                f"loc_loss_hi": np.mean(loc_losses_show)
            })
            self.update_hypernet(mask, update=False)
            swanlab.log({
                "gen_loss_low": np.mean(gen_losses_show),
                "loc_loss_low": np.mean(loc_losses_show)
            })

        self.opt.step()
        self.opt.zero_grad()
        self.hi_opt.step()
        self.hi_opt.zero_grad()


    def predict_param_shifts(self) -> Dict[str, torch.FloatTensor]:
        shape_list, keys_list, values_grad_list = [], [], []
        for module_idx, module_name in enumerate(self.config.model.edit_modules):
            shape = get_shape(get_module(self.model, module_name))
            keys = torch.load(f"{self.config.editor.cache_dir}/{self.config.model.name}_{self.config.editor.name}_{self.config.dataset.name}_{self.config.num_seq}_k{self.config.editor.n_layers}/{module_idx}_0_keys.pth")
            values_grad = torch.load(f"{self.config.editor.cache_dir}/{self.config.model.name}_{self.config.editor.name}_{self.config.dataset.name}_{self.config.num_seq}_k{self.config.editor.n_layers}/{module_idx}_0_values_grad.pth")
            shape_list.append(shape)
            keys_list.append(keys)
            values_grad_list.append(values_grad)
        mask = self.hi_net(shape_list, keys_list, values_grad_list, self.config.editor.n_layers)
    
        param_shifts = {}
        for module_idx, module_name in enumerate(self.config.model.edit_modules):
            shape = get_shape(get_module(self.model, module_name))
            net = self.net[str(shape)]
            layer_idx = torch.LongTensor([self.name2idx[module_name]]).to(self.config.editor_device)
            keys = keys_list[module_idx]
            values_grad = values_grad_list[module_idx]
            pesudo_keys, pesudo_values_grad = net(keys, values_grad, layer_idx)
            param_shift = - net.lr(layer_idx) * pesudo_keys.T @ pesudo_values_grad
            param_shifts[module_name] = param_shift

        return param_shifts, mask
    

    def update_hypernet(self, mask: Dict[str, torch.FloatTensor], update: bool):        
        self.opt.zero_grad()
        for module_idx, module_name in enumerate(self.config.model.edit_modules,):
            if not mask[module_idx]:
                continue
            shape = get_shape(get_module(self.model, module_name))
            net = self.net[str(shape)]
            layer_idx = torch.LongTensor([self.name2idx[module_name]]).to(self.config.editor_device)
            module = get_module(self.model, module_name)
            module_grad = module.weight.grad.to(torch.float32)
            if isinstance(module, nn.Linear):
                module_grad = module_grad.T
            for idx in range(math.ceil(self.config.dataset.n_edits / self.config.dataset.batch_size)):
                keys = torch.load(f"{self.config.editor.cache_dir}/{self.config.model.name}_{self.config.editor.name}_{self.config.dataset.name}_{self.config.num_seq}_k{self.config.editor.n_layers}/{module_idx}_{idx}_keys.pth")
                values_grad = torch.load(f"{self.config.editor.cache_dir}/{self.config.model.name}_{self.config.editor.name}_{self.config.dataset.name}_{self.config.num_seq}_k{self.config.editor.n_layers}/{module_idx}_{idx}_values_grad.pth")
                pesudo_keys, pesudo_values_grad = net(keys, values_grad, layer_idx)
                param_shift = - net.lr(layer_idx) * pesudo_keys.T @ pesudo_values_grad
                (module_grad * param_shift).sum().backward()

        clip_grad_norm_(
            self.net.parameters(),
            self.config.editor.max_grad_norm
        )
        
        if update == True:
            self.opt.step()
            self.opt.zero_grad()


    def update_hinet(self, update: bool):
        clip_grad_norm_(
            self.hi_net.parameters(),
            self.config.editor.max_grad_norm
        )

        if update == True:
            self.hi_opt.step()
            self.hi_opt.zero_grad()


    def sequential_valid(self, loader: DataLoader):
        max_steps = self.config.num_seq
        limited_loader = islice(loader, max_steps)
        for _, tuples in enumerate(tqdm(limited_loader, desc="Valid", ncols=100, total=max_steps)):
            if self.config.glue_step > 0:
                if _ == 0 or (_+1) % self.config.glue_step == 0:
                    tokenizer = AutoTokenizer.from_pretrained(self.config.model.name_or_path)
                    glue_eval = GLUEEval(self.model, tokenizer, number_of_tests = 100)
                    out_file = f"glue_eval/results/"
                    if not os.path.exists(out_file):
                        os.makedirs(out_file, exist_ok=True)
                    out_file = f"glue_eval/results/{self.config.editor.cache_dir}_k{self.config.editor.n_layers}_{self.config.model.name}_{self.config.editor.name}_{self.config.dataset.name}_{self.config.num_seq}_{_}_glue.json"
                    glue_results = {'edit_num': -1}
                    glue_results = glue_eval.evaluate(glue_results, out_file, nli_flag = True, sst_flag = True, cola_flag=True, rte_flag=True, mmlu_flag = True, mrpc_flag = True)
                    with open(out_file, "w") as f:
                        json.dump(glue_results, f, indent=4)
                    print("GLEU: ", np.mean([v["f1"] for k, v in glue_results.items() if isinstance(v, dict)]))
            self.cache(tuples["edit_tuples"])
            param_shifts, mask = self.predict_param_shifts()
            selected_param_shifts = {}
            for module_idx, module_name in enumerate(self.config.model.edit_modules):
                selected_param_shifts[module_name] = param_shifts[module_name] * mask[module_idx]
            self.edit_model(selected_param_shifts, False)
            self.tuples_list.append(tuples)
            self.opt.zero_grad()
        edit_succs, gen_succs, loc_succs = [], [], []
        for k, s in zip(
            ["edit_tuples", "equiv_tuples", "unrel_tuples"],
            [edit_succs, gen_succs, loc_succs]
        ):
            for tuple in tqdm(self.tuples_list, desc=k, ncols=100):
                for t in tuple[k]:
                    if "old_labels" in t:
                        old_labels = t.pop("old_labels")
                    with torch.no_grad():
                        logits = self.model(**t)["logits"]
                    try:
                        t["old_labels"] = old_labels
                    except:
                        pass
                    if self.config.dataset.name == "counterfact":
                        t["old_labels"] = old_labels
                        s += succ_ratios(logits, t["labels"], t["old_labels"])
                    else:
                        s += succ_ratios(logits, t["labels"])
        print({
            f"ES": np.mean(edit_succs),
            f"GS": np.mean(gen_succs),
            f"LS": np.mean(loc_succs),
            f"Pre_ES": np.mean(edit_succs[:self.config.num_pre]),
            f"Pre_GS": np.mean(gen_succs[:self.config.num_pre]),
            f"Pre_LS": np.mean(loc_succs[:self.config.num_pre])
        })
        swanlab.log({
            f"ES": np.mean(edit_succs),
            f"GS": np.mean(gen_succs),
            f"LS": np.mean(loc_succs),
            f"Pre_ES": np.mean(edit_succs[:self.config.num_pre]),
            f"Pre_GS": np.mean(gen_succs[:self.config.num_pre]),
            f"Pre_LS": np.mean(loc_succs[:self.config.num_pre])
        })


    def run(self, train_loader: DataLoader, valid_loader: DataLoader):
        for epoch in tqdm(range(self.config.editor.n_epochs), desc="epoch"):
            self.train(train_loader)
            self.reset_model()
            if self.config.editor.save_checkpoint:
                save_dir = "checkpoints"
                torch.save(self.net.state_dict(), f"{save_dir}/{self.config.editor.cache_dir}_k{self.config.editor.n_layers}_{self.config.model.name}_{self.config.editor.name}_{self.config.dataset.name}_{self.config.num_seq}_{epoch}_low_net.pth")
                torch.save(self.opt.state_dict(), f"{save_dir}/{self.config.editor.cache_dir}_k{self.config.editor.n_layers}_{self.config.model.name}_{self.config.editor.name}_{self.config.dataset.name}_{self.config.num_seq}_{epoch}_low_opt.pth")
                torch.save(self.hi_net.state_dict(), f"{save_dir}/{self.config.editor.cache_dir}_k{self.config.editor.n_layers}_{self.config.model.name}_{self.config.editor.name}_{self.config.dataset.name}_{self.config.num_seq}_{epoch}_hi_net.pth")
                torch.save(self.hi_opt.state_dict(), f"{save_dir}/{self.config.editor.cache_dir}_k{self.config.editor.n_layers}_{self.config.model.name}_{self.config.editor.name}_{self.config.dataset.name}_{self.config.num_seq}_{epoch}_hi_opt.pth")
                print("-----Saved checkpoints-----")

            self.net.eval(), self.hi_net.eval()
            self.sequential_valid(valid_loader)
            self.reset_model()

            self.net.train(), self.hi_net.train()
            empty_cache(self.config.editor.cache_dir, self.config)
