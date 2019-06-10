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
from configuration import *
from helpers import *

import spacy




def fine_tune(work_dir, data_loader_folder, dataLoader_name='data_lm', arch=AWD_LSTM, custom_config=False,
              pretrained=False, pretrained_fnames=None, drop_mult=0.5, opt_func='Adam', loss_func=None,
              metrics=[accuracy], true_wd=True, bn_wd=True, wd=0.01, train_bn=True, lr=3e-4,
              model_save_path='DataLoader/models/modelFine.pth', vocab_save_path='vocab/itosFine.pkl',
              encoding_save_path='encoding/encoderFine.pth', DataLoaderLM_save_path='DataLoader/data_lmFine.pkl',
              DataLoaderCl_save_path='DataLoader/data_clFine.pkl', cuda_id=-1, vocab_weight_balance=False,
              truncate_val=None):
    assert os.path.isdir(work_dir / data_loader_folder) is True

    data_lm = load_data(work_dir / data_loader_folder, dataLoader_name)

    lrs = 2.6
    vocab_length = len(data_lm.vocab.itos)
    print('vocab length', vocab_length)
    pre_model_state = torch.load(str(pretrained_fnames[0]) + '.pth')
    if vocab_weight_balance is True and list(pre_model_state.items())[0][1].shape[0] != vocab_length:
        pre_model_state = torch.load(str(pretrained_fnames[0]) + '.pth')

        new_state_dict = augument_state_dict(pre_model_state, vocab_length, trunc=truncate_val)
        print('shape of new encodings is', list(new_state_dict.items())[0][1].shape)
        torch.save(new_state_dict, str(pretrained_fnames[0]) + '_new.pth')
        # pretrained_fnames[0] = str(pretrained_fnames[0]) + 'new'
    # else:
    #     new_state_dict = pretrained_fnames[0]
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

    print(custom_config)

    learn = language_model_learner(data_lm, arch=arch, drop_mult=drop_mult,
                                   config=config, pretrained=False, loss_func=loss_func, metrics=metrics,
                                   true_wd=true_wd, wd=wd, train_bn=train_bn)
    if pretrained is True and pretrained_fnames is not None:
        if vocab_weight_balance is False:
            learn.load_pretrained(wgts_fname=str(pretrained_fnames[0]) + '.pth', itos_fname=str(pretrained_fnames[1]) + '.pkl')
            print('loading pretrained model')
        else:
            learn.load_pretrained(wgts_fname=str(pretrained_fnames[0]) + '_new' + '.pth',
                                  itos_fname=str(pretrained_fnames[1]))

        # learn.load_encoder(pretrained_fnames[0] + 'new')
        learn.freeze()
    learn.fit_one_cycle(1, max_lr=lr, moms=(0.8, 0.7))
    learn.unfreeze()
    learn.fit_one_cycle(10, max_lr=lr, moms=(0.8, 0.7))

    save_workpsace(work_dir, save_type=['model', 'encoding'], data_lm=data_lm, learner_model=learn,
                   model_path=model_save_path, encoding_path=encoding_save_path, vocab_path=vocab_save_path)

