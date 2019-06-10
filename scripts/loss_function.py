import os
from collections import OrderedDict
from functools import partial

import warnings
import os, re, sys, datetime, time, string
import numpy as np, scipy, random, pandas as pd
import pickle, json, collections
from pprint import pprint
import ruamel_yaml as yaml
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import sensus.common.global_variables as gv

import torch, torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from sensus.common.constants import *

#%matplotlib inline

# common imports and global settings
from IPython.core.interactiveshell import InteractiveShell
from IPython.display import display, HTML
InteractiveShell.ast_node_interactivity = "all"
display(HTML("<style>.container { width:90% !important; }</style>"))


warnings.filterwarnings('ignore')
from notebooks.chetan.fastaiConsole.fastai.fastai.text import *
from notebooks.chetan.fastai.fastai import *


import spacy



def loss_func1(pred_x, y, weights):
    pred_x = pred_x.double().cuda()
    y = y.double().cuda()

    sigm_pred = torch.sigmoid(pred_x)
    try:
        loss_ = 10 * weights.double() * (torch.exp(torch.abs(y - sigm_pred)).double() - torch.ones(1).double())
    except:
        loss_ = 10 * weights.double().cuda() * (
                    torch.exp(torch.abs(y - sigm_pred)).double() - torch.ones(1).double().cuda())
    # print(torch.mean(loss_))
    loss_val = torch.mean(loss_)
    return loss_val.float()