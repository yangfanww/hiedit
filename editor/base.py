from typing import Dict, List
from omegaconf import DictConfig

from collections import Counter
import os

import torch
import torch.nn as nn


from model import make_model
from util import (
    get_module,
    get_shape,
    TracerDict,
    cross_entropy,
)


class BaseEditor:

    def __init__(
        self,
        config: DictConfig,
        model: nn.Module
    ):
        
        self.config = config
        self.model = model
        
        shape_counter = Counter()
        self.name2idx = {}
        for module_name in config.model.edit_modules:
            shape = get_shape(get_module(model, module_name))
            self.name2idx[module_name] = shape_counter[shape]
            shape_counter[shape] += 1
        
        self.shape_counter = shape_counter

        self.tuples_list = []


    def edit_model(
        self,
        param_shifts: Dict[str, torch.FloatTensor],
        is_reverse: bool
    ):
        
        for module_name, param_shift in param_shifts.items():
            module = get_module(self.model, module_name)
            if isinstance(module, nn.Linear):
                param_shift = param_shift.T
            if is_reverse:
                param_shift = - param_shift
            module.weight.data += param_shift.to(module.weight.data.dtype)


    def reset_model(self):
        del self.model
        torch.cuda.empty_cache()
        self.model = make_model(self.config.model).to(self.config.model_device)


    def cache(self, tuples: List[Dict[str, torch.LongTensor]]):

        for idx, t in enumerate(tuples):
            
            if "old_labels" in t:
                old_labels = t.pop("old_labels")

            with TracerDict(
                self.model,
                self.config,
                t
            ) as tr:
                logits = self.model(**t)["logits"]
                cross_entropy(logits, t["labels"]).backward()
        
            for module_idx, module_name in enumerate(self.config.model.edit_modules):
                shape = get_shape(get_module(self.model, module_name))
                keys = tr[module_name].keys.to(torch.float32).to(self.config.editor_device)
                values_grad = tr[module_name].values_grad.to(torch.float32).to(self.config.editor_device)
                self.net[str(shape)].normalizer.update(torch.cat((keys, values_grad), -1))
                dir_path = f"{self.config.editor.cache_dir}/{self.config.model.name}_{self.config.editor.name}_{self.config.dataset.name}_{self.config.num_seq}_k{self.config.editor.n_layers}"
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                torch.save(keys, f"{dir_path}/{module_idx}_{idx}_keys.pth")
                torch.save(values_grad, f"{dir_path}/{module_idx}_{idx}_values_grad.pth")

            try:
                t["old_labels"] = old_labels
            except:
                pass