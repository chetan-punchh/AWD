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



def train_clas(work_dir, data_loader_folder, n_labels, dataLoader_name='data_cl', arch=AWD_LSTM, custom_config=False,
               pretrained=False, pretrained_fnames=None, drop_mult=0.5, opt_func='Adam', loss_func=None,
               metrics=[accuracy], true_wd=True, bn_wd=True, wd=0.01, train_bn=True,
               DataLoaderLM_save_path='DataLoader/data_lmFine.pkl',
               DataLoaderCl_save_path='DataLoader/data_clFine.pkl', cuda_id=-1, use_discriminative=False,
               chain_thaw=False, lr = 3e-04, model_save_path = 'DataLoader/models/model', encoding_save_path = 'encoding/encoder'):
    assert Path(work_dir / data_loader_folder).exists()
    lrm = 2.6

    if pretrained is True and pretrained_fnames is not None:
        assert Path(str(work_dir/pretrained_fnames[0]) + '.pth').exists()
        assert Path(str(work_dir/pretrained_fnames[1]) + '.pkl').exists()
    ## import config file in header
    if custom_config is not False:
        if arch is AWD_LSTM:
            config = awd_lstm_clas_config_custom
        if arch is Transformer:
            config = tfmer_clas_config_custom
        if arch is TransformerXL:
            config = tfmerXL_clas_config_custom

    else:
        config = None

    data_cl = load_data(work_dir / data_loader_folder, dataLoader_name)
    labels = data_cl.train_ds.y.items
    class_InvWeights = get_InverseClassWeights(labels, n_labels=n_labels)

    loss_func = partial(loss_func1, weights=class_InvWeights)

    learn = text_classifier_learner(data_cl, arch=arch, drop_mult=drop_mult, config=config, pretrained=False,
                                    true_wd=true_wd, wd=wd, train_bn=train_bn)
    learn.loss_func = loss_func
    if pretrained is True and pretrained_fnames is not None:
        # learn.load_pretrained(wgts_fname=pretrained_fnames[0] + '.pth', itos_fname=pretrained_fnames[1] + '.pkl')
        learn.load_encoder(str(work_dir/pretrained_fnames[0]))
    learn.freeze()

    if use_discriminative:
        lrs = np.array([lr / (lrm ** 4), lr / (lrm ** 3), lr / (lrm ** 2), lr / lrm, lr])
    else:
        lrs = lr

    if chain_thaw:
        lrs = np.array([0.0001, 0.0001, 0.0001, 0.0001, 0.001])
        n_layers = config['n_layers']
        learn.fit_one_cycle(1, max_lr=lrs, moms=(0.8, 0.7))
        for i in range(n_layers - 1):
            learn.freeze_to(-(i + 1))
            learn.fit_one_cycle(1, max_lr=lrs, moms=(0.8, 0.7))

    learn.unfreeze()
    learn.fit_one_cycle(2, max_lr=lrs, moms=(0.8, 0.7))

    save_workpsace(work_dir, save_type=['model', 'encoding'], learner_model=learn,
                   model_path=model_save_path, encoding_path=encoding_save_path)