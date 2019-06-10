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
from configuration import *
from helpers import *

#%matplotlib inline

# common imports and global settings
from IPython.core.interactiveshell import InteractiveShell
from IPython.display import display, HTML
InteractiveShell.ast_node_interactivity = "all"
display(HTML("<style>.container { width:90% !important; }</style>"))


warnings.filterwarnings('ignore')

from notebooks.chetan.fastaiConsole.fastai.fastai.text import *

import spacy


def pre_train(work_dir, data_loader_folder='DataLoader', dataLoader_name='data_lm', arch=AWD_LSTM, custom_config=False,
              pretrained=False, pretrained_fnames=None, drop_mult=0.5, opt_func='Adam', loss_func=None,
              metrics=[accuracy], true_wd=True, bn_wd=True, wd=0.01, train_bn=True, lr=3e-4,
              model_save_path='DataLoader/models/model.pth', vocab_save_path='vocab/itos.pkl',
              encoding_save_path='encoding/encoder.pth', DataLoaderLM_save_path='DataLoader/data_lm.pkl',
              DataLoaderCl_save_path='DataLoader/data_cl.pkl', cuda_id=-1, bs = 64, discriminative = False):
    if cuda_id != -1 and torch.cuda.is_available():
        torch.cuda.set_device(cuda_id)
        gv.reset(device="gpu")
        print(f"Current cuda device: {torch.cuda.current_device()} " f"with device capability {torch.cuda.get_device_capability(cuda_id)}"
        )
    else:
        print("Cuda device unavailable or not selected.")
        gv.reset(device="cpu")
        print("Training on CPU... will be very slow")
    data_path = work_dir / data_loader_folder

    assert os.path.isdir(work_dir / data_loader_folder) is True
    data_path = work_dir / data_loader_folder
    data_lm = load_data(data_path, dataLoader_name)
    ## import config file in header
    if custom_config is not False:
        if arch is AWD_LSTM:
            config = awd_lstm_lm_config_custom
        if arch is Transformer:
            config = tfmer_lm_config_custom
        if arch is TransformerXL:
            config = tfmerXL_lm_config_custom

    else:
        config = None

    learn = language_model_learner(data_lm, arch=arch, drop_mult=drop_mult, pretrained_fnames=pretrained_fnames,
                                   config=config, pretrained=pretrained, loss_func=loss_func, metrics=metrics,
                                   true_wd=true_wd, wd=wd, train_bn=train_bn)
    # lrs = np.array([lr / 6, lr / 3, lr, lr])
    if discriminative:
        lrs = slice(lr / 6, lr)
    else:
        lrs = lr
    learn.fit_one_cycle(1, max_lr=lrs, moms=(0.8, 0.7))
    learn.unfreeze()
    learn.fit_one_cycle(10, max_lr=lrs, moms=(0.8, 0.7))
    save_workpsace(work_dir, save_type=['model', 'encoding'], data_lm=data_lm, learner_model=learn,
                   model_path=model_save_path, encoding_path=encoding_save_path, vocab_path=vocab_save_path)
