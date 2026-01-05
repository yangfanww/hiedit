from typing import Tuple, List, Dict
import torch
import torch.nn as nn
from collections import Counter

class RunningMeanStd(nn.Module):

    def __init__(self, size: int):
        super().__init__()

        self.register_buffer("n", torch.zeros(1))
        self.register_buffer("mean", torch.zeros((size)))
        self.register_buffer("var", torch.zeros((size)))
        self.register_buffer("std", torch.zeros((size)))


    def update(self, x: torch.FloatTensor):
        n_new = x.shape[0]
        n_total = self.n + n_new
        if n_new == 0:
            return
        delta = x.mean(0) - self.mean
        self.mean += n_new * delta / n_total
        var_internal_contrib = torch.zeros_like(self.var)
        if n_new > 1:
            var_internal_contrib = n_new * x.var(0)
        var_mean_distance_contrib = self.n * n_new * delta.pow(2) / n_total
        self.var += var_internal_contrib + var_mean_distance_contrib
        self.std = (self.var / (n_total - 1 + torch.finfo(x.dtype).eps)).sqrt()
        self.n = n_total


    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:

        return (x - self.mean) / (self.std + torch.finfo(x.dtype).eps)


class RLEditBlock(nn.Module):

    def __init__(self, size: int, rank: int, n_modules: int):
        super().__init__()

        self.A = nn.Parameter(torch.randn(size, rank))
        self.B = nn.Parameter(torch.zeros(rank, size))
        self.bias = nn.Parameter(torch.zeros(size))
        
        self.scale = nn.Embedding(n_modules, size)
        self.shift = nn.Embedding(n_modules, size)
        
        self.scale.weight.data.fill_(1)
        self.shift.weight.data.fill_(0)


    def forward(
        self,
        y: torch.FloatTensor,
        module_idx: torch.LongTensor
    ) -> torch.FloatTensor:

        x = y @ self.A @ self.B + self.bias
        x = x.clamp(0)

        x = self.scale(module_idx) * x + self.shift(module_idx)
        x = x + y

        return x


class RLEditNet(nn.Module):

    def __init__(
        self,
        key_size: int,
        value_size: int,
        rank: int,
        n_blocks: int,
        n_modules: int,
        lr: float
    ):
        super().__init__()
        self.key_size = key_size
        self.value_size = value_size

        self.normalizer = RunningMeanStd(key_size + value_size)
        self.blocks = nn.ModuleList([
            RLEditBlock(key_size + value_size, rank, n_modules)
            for _ in range(n_blocks)
        ])

        self.lr = nn.Embedding(n_modules, 1)
        self.lamda = nn.Embedding(n_modules, 1)
        
        self.lr.weight.data.fill_(lr)
        self.lamda.weight.data.fill_(0)


    def forward(
        self,
        keys: torch.FloatTensor,
        values_grad: torch.FloatTensor,
        module_idx: torch.LongTensor
    ) -> Tuple[torch.FloatTensor]:

        hidden_states = torch.cat((keys, values_grad), -1)
        hidden_states = self.normalizer(hidden_states)
        for block in self.blocks:
            hidden_states = block(hidden_states, module_idx)
        return hidden_states.split([self.key_size, self.value_size], -1)




class EncoderBlock(nn.Module):

    def __init__(
            self, 
            size: int, 
            hidden_size: int, 
            n_modules: int
        ):
        super().__init__()

        self.linear = nn.Linear(size, hidden_size)
        self.scale = nn.Embedding(n_modules, hidden_size)
        self.shift = nn.Embedding(n_modules, hidden_size)
        self.scale.weight.data.fill_(1)
        self.shift.weight.data.fill_(0)


    def forward(
        self,
        y: torch.FloatTensor,
        module_idx: torch.LongTensor
    ) -> torch.FloatTensor:

        x = self.linear(y)
        x = x.clamp(0)
        x = self.scale(module_idx) * x + self.shift(module_idx)
        return x


class Encoder(nn.Module):

    def __init__(
        self,
        key_size: int,
        value_size: int,
        hidden_size: int,
        n_modules: int,
    ):
        super().__init__()

        self.key_size = key_size
        self.value_size = value_size
        self.normalizer = RunningMeanStd(key_size + value_size)
        self.block = EncoderBlock(key_size + value_size, hidden_size, n_modules)

    def forward(
        self,
        keys: torch.FloatTensor,
        values_grad: torch.FloatTensor,
        module_idx: torch.LongTensor
    ) -> Tuple[torch.FloatTensor]:
        
        hidden_states = torch.cat((keys, values_grad), -1)
        hidden_states = self.normalizer(hidden_states)
        hidden_states = self.block(hidden_states, module_idx)
        return hidden_states


class HiNet(nn.Module):
    def __init__(
            self,
            shape_counter: Counter,
            hidden_size: int,
            edit_modules: List[str],
            name2idx: Dict[str, int],
            device: str,
        ):
        super().__init__()

        self.edit_modules = edit_modules
        self.device = device
        self.name2idx = name2idx
        self.encoder = nn.ModuleDict({
            str(k): Encoder(
                *k,
                hidden_size,
                v,
            )
            for k, v in shape_counter.items()
        })
        self.gate = nn.Linear(hidden_size * len(edit_modules), len(edit_modules))

    def forward(
            self,
            shape_list: List[str],
            keys_list: List[torch.FloatTensor],
            values_grad_list: List[torch.FloatTensor],
            k: int
        ):

        hidden_state = torch.cat([
            self.encoder[str(shape)](
                keys, 
                values_grad, 
                torch.LongTensor([self.name2idx[module_name]]).to(self.device),
            ) 
            for module_name, shape, keys, values_grad in zip(self.edit_modules, shape_list, keys_list, values_grad_list)
        ], dim=-1)
        gate_out = self.gate(hidden_state.mean(0))
        _, topk_indices = torch.topk(gate_out, k=k, dim=-1)

        # STE
        mask = torch.zeros_like(gate_out)
        mask.scatter_(-1, topk_indices, 1)
        mask = (mask - gate_out).detach() + gate_out
        return mask