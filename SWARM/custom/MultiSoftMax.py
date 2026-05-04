
from typing import Tuple

import torch
import torch.nn as nn
class MultiSoftMax(nn.Module):
    def __init__(self, dim_begin: int, dim_end: int, sections: Tuple = None):
        super().__init__()
        self.dim_begin = dim_begin
        self.dim_end = dim_end
        self.sections = sections

        if sections:
            assert dim_end - dim_begin == sum(sections), "expected same length of sections and customized" \
                                                         "dims"
    def forward(self, input_tensor: torch.Tensor):
        x = input_tensor[..., self.dim_begin:self.dim_end]
        res = input_tensor.clone()
        res[..., self.dim_begin:self.dim_end] = torch.cat([
            xx.softmax(dim=-1) for xx in torch.split(x, self.sections, dim=-1)], dim=-1)
        return res
