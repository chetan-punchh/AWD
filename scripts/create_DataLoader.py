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

import spacy

from helpers import *
#print('imported the files')









def get_DataLoader(path, data_trn, data_val, n_labels, n_textfields=1, input_sub="tmp",
                   max_vocab=60000,
                   min_freq=1,
                   use_pretrain_vocab=False,
                   pretrain_path=None,
                   pretrain_input_sub="tmp",
                   no_update_pretrain_vocab=True,
                   pretrain_vocab_name=None,
                   pretrain_vocab_truncate=None, Save_Path = None, DataLoader_save_path = 'DataLoader/data_lm.pkl', vocab_save_path = 'vocab/itos.pkl', bs = 64):
    vocab_updated = False

    # df_tr = data_trn.dropna(axis=0, subset=['text'])
    #     # df_v = data_val.dropna(axis=0, subset=['text'])
    if Save_Path is None:
        Save_Path = path

    if n_labels is None:

        data_lm_preTrain = TextLMDataBunch.from_df(path, train_df=data_trn, valid_df=data_val, max_vocab=max_vocab,
                                          min_freq=min_freq, bs = bs)
    # data_lm_PreTrain = load_data(Data_path/'DataLoader', 'data_ClYelp.pkl')
    else:
        data_lm_preTrain = TextLMDataBunch.from_df(path, train_df=data_trn, valid_df=data_val, max_vocab=max_vocab,
                                          min_freq=min_freq, text_cols=n_labels, bs = bs)
    # data_lm_preTrain = load_data(Save_Path/'DataLoader', 'data_lm_preTrain_wiki.pkl')

    #     p = Path(dir_path)
    #     assert p.exists(), f"Error: {p} does not exist."
    #     vocab_path = p / input_sub
    #     assert vocab_path.exists(), f"Error: {vocab_path} does not exist."
    if use_pretrain_vocab:
        if pretrain_path is not None:
            pretrain_path = Path(pretrain_path)
            assert pretrain_path.exists(), f"Error: {pretrain_path} does not exist."
        else:
            raise ValueError("'use_pretrain_vocab' is set True, yet 'pretrain_path' is not provided.")

    current_vocab = data_lm_preTrain.vocab.itos
    # current_vocab.append('asdfghj')
    if use_pretrain_vocab:
        if pretrain_vocab_name is None:
            pretrain_vocab_name = "itos.pkl"
        try:
            pre_itos = pickle.load(open(pretrain_path / pretrain_vocab_name, "rb"))
        except FileNotFoundError:
            raise FileNotFoundError("'itos.pkl' for pretrained dataset not found.")
        if not no_update_pretrain_vocab:
            if pretrain_vocab_truncate:
                assert isinstance(pretrain_vocab_truncate, int)
                pre_itos = pre_itos[0:pretrain_vocab_truncate]
            extra_space = max(0, max_vocab - len(current_vocab))
            print('adding vocab')
            for o in current_vocab:
                if extra_space == 0:
                    print('no more extra space')
                    break
                if o not in pre_itos:
                    print(f'adding word {o} to the vocab')
                    pre_itos.append(o)
                    extra_space -= 1
        data_lm_preTrain.vocab.itos = pre_itos
        vocab_updated = True

    save_workpsace(work_dir=Save_Path, data_lm=data_lm_preTrain, save_type=['data_lm', 'vocab'], DataLoaderLM_path=DataLoader_save_path, vocab_path = vocab_save_path)
    return data_lm_preTrain, vocab_updated, pretrain_vocab_truncate


# def loss_func1(pred_x, y, weights):
#     pred_x = pred_x.double().cuda()
#     y = y.double().cuda()
#
#     sigm_pred = torch.sigmoid(pred_x)
#     try:
#         loss_ = 10 * weights.double() * (torch.exp(torch.abs(y - sigm_pred)).double() - torch.ones(1).double())
#     except:
#         loss_ = 10 * weights.double().cuda() * (
#                     torch.exp(torch.abs(y - sigm_pred)).double() - torch.ones(1).double().cuda())
#     # print(torch.mean(loss_))
#     loss_val = torch.mean(loss_)
#     return loss_val.float()


def get_Clas_DataLoader(path, data_trn, data_val, n_labels,
                        max_vocab=60000,
                        min_freq=1,
                        use_pretrain_vocab=False,
                        pretrain_path=None,
                        pretrain_input_sub="tmp",
                        no_update_pretrain_vocab=True,
                        pretrain_vocab_name=None,
                        pretrain_vocab_truncate=None, DataLoader_name=None, bs=32, Save_Path = None, DataLoader_save_path = 'DataLoader/data_cl_fineTune_Rev.pkl'):
    # df_tr = data_trn.dropna(axis=0, subset=['text'])
    # df_v = data_val.dropna(axis=0, subset=['text'])
    if Save_Path is None:
        Save_Path = path

    if use_pretrain_vocab:
        if pretrain_path is not None:
            pretrain_path = Path(pretrain_path)
            assert pretrain_path.exists(), f"Error: {pretrain_path} does not exist."
            data_lm = load_data(Path(pretrain_path), DataLoader_name)
            vocab_ = data_lm.vocab
        else:
            raise ValueError("'use_pretrain_vocab' is set True, yet 'pretrain_path' is not provided.")
            vocab_ = None

    data_cl = TextClasDataBunch.from_df(path, train_df=data_trn, valid_df=data_val, max_vocab=max_vocab,
                                        min_freq=min_freq, vocab=vocab_, bs=bs, text_cols='text',
                                        label_cols=np.arange(n_labels).tolist())

    save_workpsace(work_dir=Save_Path, data_cl=data_cl, save_type=['data_cl'],
                   DataLoaderCl_path=DataLoader_save_path)

    return data_cl
