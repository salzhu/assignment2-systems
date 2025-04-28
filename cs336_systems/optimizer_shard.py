import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import timeit 
import pandas as pd 
import numpy as np 
import argparse

from typing import Any
from torch.optim import Optimizer as Optimizer

class OptimizerSharded(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls, **kwargs: Any):
        super().__init__(params)

        world_size = 2 

        raise NotImplementedError
    
    def step(self, closure, **kwargs):
        raise NotImplementedError
    
    def add_param_group(self, param_group: dict[str, Any]):
        raise NotImplementedError