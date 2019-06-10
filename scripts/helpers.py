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





def load_data_training(data_path, input_sub='tmp', train_fname='train.csv', val_fname='val.csv', n_labels=None,
                       n_textfield=1):
    #     df_trnYelp = pd.read_csv(data_path / train_fname, header = None)
    #     df_valYelp = pd.read_csv(data_path / val_fname, header = None)
    if n_textfield == 1 and n_labels != None:
        df_trn = pd.read_csv(data_path / train_fname, header=None)
        df_val = pd.read_csv(data_path / val_fname, header=None)
        df_trn.rename(columns=lambda x: f'label_{x}' if x < n_labels else 'text', inplace=True)
        df_val.rename(columns=lambda x: f'label_{x}' if x < n_labels else 'text', inplace=True)
    elif n_textfield == 1 and n_labels == None:
        df_trn = pd.read_csv(data_path / train_fname, header=None, names=['text'])
        df_val = pd.read_csv(data_path / val_fname, header=None, names=['text'])
        ## Create Dummy Variables for Labels if needed
        if not 'labels' in df_trn.columns.tolist():
            df_trn.insert(0, 'labels', value=0)
        if not 'labels' in df_val.columns.tolist():
            df_val.insert(0, 'labels', value=0)

    df_tr = df_trn.dropna(axis=0, subset=['text'])
    df_v = df_val.dropna(axis=0, subset=['text'])

    return df_tr.copy(), df_v.copy()

def save_workpsace(work_dir, save_type = None,learner_model = None, data_lm = None, data_cl = None, model_path = 'models/model.pth', vocab_path = 'vocab/itos.pkl', encoding_path = 'encoding/encoder.pth', DataLoaderLM_path = 'DataLoader/data_lm.pkl', DataLoaderCl_path = 'DataLoader/data_cl.pkl'):
    ## Can save DataLoader, Embedding, Vocab, Model
    work_dir = Path(work_dir)
    assert os.path.isdir(work_dir) is True
    if 'model' in save_type and learner_model is not None:
        check_create_dir(work_dir / model_path)
        learner_model.save(work_dir / model_path)
    if 'data_lm' in save_type and data_lm is not None:
        check_create_dir(work_dir / DataLoaderLM_path)
        data_lm.save(work_dir / DataLoaderLM_path)
    if 'data_cl' in save_type and data_cl is not None:
        check_create_dir(work_dir / DataLoaderCl_path)
        data_cl.save(work_dir / DataLoaderCl_path)
    if 'vocab' in save_type and data_lm is not None:
        check_create_dir(work_dir / vocab_path)
        data_lm.vocab.save(work_dir / vocab_path)
    if 'encoding' in save_type and learner_model is not None:
        check_create_dir(work_dir / encoding_path)
#         torch.save(learner_model.model.state_dict(), work_dir / encoding_path)
#         split_path = encoding_path.split('.')
#         fastaiEnc_path = split_path[0] + '_fastai_save'
# #         assert os.pathisdir(work_dir/ )
#         print(work_dir / fastaiEnc_path)
        learner_model.save_encoder(work_dir / encoding_path)


def check_create_dir(path):
    dir_str = str(path)
    path_list = dir_str.split('/')
    path_list = path_list[:-1]
    path_way = '/'.join(path_list)
    if os.path.isdir(path_way) is False:
        os.mkdir(path_way)


def augument_tensor(old_tensor, size):
    assert size >= 1
    if len(old_tensor.shape) > 1:
        old_tensorAvg = old_tensor.mean(0)
        old_tensor_rep = old_tensorAvg.repeat(size, 1)
        return torch.cat([old_tensor, old_tensor_rep], 0)
    else:
        old_tensorAvg = old_tensor.mean()
        old_tensor_rep = old_tensorAvg.repeat(size)
        return torch.cat([old_tensor, old_tensor_rep])


def merge_state(cur_listState, listState, listPos):
    for states, pos in zip(listState, listPos):
        list_W = list(cur_listState[pos])
        list_W[1] = states
        tuple_W = tuple(list_W)
        cur_listState[pos] = tuple_W
    new_s = OrderedDict(cur_listState)
    return new_s


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


def augument_state_dict(model_state, vocab_size, trunc=None):
    listStates = list(model_state.items())
    encodings = listStates[0][1]
    encoding_weights = listStates[1][1]
    decoder_emb = listStates[-2][1]
    decoder_bias = listStates[-1][1]
    if trunc is not None:
        encodings = encodings[:trunc]
        encoding_weights = encoding_weights[:trunc]
        decoder_emb = decoder_emb[:trunc]
        decoder_bias = decoder_bias[:trunc]
    new_encodings = augument_tensor(encodings, vocab_size - encodings.shape[0])
    new_encoding_weights = augument_tensor(encodings, vocab_size - encodings.shape[0])
    new_decoder_emb = augument_tensor(encodings, vocab_size - encodings.shape[0])
    new_decoder_bias = augument_tensor(encodings, vocab_size - encodings.shape[0])
    new_state = merge_state(listStates, [new_encodings, new_encoding_weights, new_decoder_emb, new_decoder_bias],
                            [0, 1, -2, -1])
    return new_state


def get_InverseClassWeights(lab, n_labels):
    if n_labels == 1:
        weights = np.bincount(lab)
    else:
        weights = np.count_nonzero(lab, axis = 0)
    weights = 1.0/weights
    weights /= weights.sum()
    return torch.from_numpy(weights)