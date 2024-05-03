import torch
from torch import nn
import numpy as np
import math
from torch.nn import LayerNorm
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')