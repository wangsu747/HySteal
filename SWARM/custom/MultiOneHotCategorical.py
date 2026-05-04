

from functools import reduce
from typing import Tuple

import torch
from torch.distributions import OneHotCategorical


class MultiOneHotCategorical(OneHotCategorical):

    def __init__(self, probs: torch.Tensor, sections: Tuple):
        self._sections = sections
        self._dists = [OneHotCategorical(x) for x in torch.split(probs, sections, dim=-1)]

    def sample(self, sample_shape=torch.Size()):
        res = torch.cat([dist.sample() for dist in self._dists], dim=-1)
        return res

    def log_prob(self, value):
        values = torch.split(value, self._sections, dim=-1)
        log_probs = [dist.log_prob(v) for dist, v in zip(self._dists, values)]
        return reduce(torch.add, log_probs)

    def entropy(self):
        entropy_list = [dist.entropy() for dist in self._dists]
        return reduce(torch.add, entropy_list)
