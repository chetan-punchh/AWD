import os
from collections import OrderedDict
from functools import partial

import argparse
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
from create_DataLoader import *
from configuration import *
from pre_train import pre_train
from fine_tune import fine_tune
from train_clas import train_clas

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store_true")
parser.add_argument("-pt", "--pre_train", help="pre train on wiki103",
                    action="store_true")
parser.add_argument("-dt", "--domain_specific_train", help="fine tune on yelp dataset",
                    action="store_true")
parser.add_argument("-ft", "--fine_tune", help="fine tune on the reviews",
                    action="store_true")
parser.add_argument("-cl", "--classifier", help="train classifier",
                    action="store_true")
args = parser.parse_args()

#print('verbosity', args.verbose)

if args.verbose:
    print('packages imported')


## Check GPU Availability
if args.verbose:
    try:
        print("CUDA device available: \t\t", torch.cuda.is_available())
        print("CUDA device count: \t\t", torch.cuda.device_count())
        print("CUDA device name: \t\t", torch.cuda.get_device_name(0))
        print("CUDA device capability: \t", torch.cuda.get_device_capability(0))
        print("CUDA device properties: \n", torch.cuda.get_device_properties(0))
    except:
        print('gpu not found \nInstall Dependencies')


WORK_DIR = Path(os.path.dirname(SENSUS_DIR))
os.chdir(WORK_DIR)
DATA_DIR = Path(DATA_DIR)
CUR_DIR = Data_path = WORK_DIR /'notebooks/chetan'


if args.verbose:
    print("Set working directory: ", WORK_DIR)

    print("Data directory: ", DATA_DIR)

## Load spacy model
lang = 'en'
try:
    spacy.load(lang)
except:
    # TODO handle tokenization of Chinese, Japanese, Korean
    print(f"spacy tokenization model is not installed for {lang}.")
    lang = lang if lang in ["en", "de", "es", "pt", "fr", "it", "nl"] else "xx"
    os.system('python -m spacy download {}'.format(lang))

## Training With Wikipedia


if args.pre_train:
    data_path = DATA_DIR / 'wikipedia/wiki/en'
    input_sub = "tmp"
    train_fname = "train.csv"
    val_fname = "val.csv"

    df_trnC, df_valC = load_data_training(data_path=data_path, n_labels=None, n_textfield=1)
    #print(df_trnC.head())
    assert df_trnC.columns.tolist() == ['labels', 'text']
    assert df_valC.columns.tolist() == ['labels', 'text']

    ## Create Dataloader for wiki103

    #data_lm_preTrain, vocab_update_wiki, truncate_val = get_DataLoader(path=data_path, data_trn=df_trnC,
     #                                                                 data_val=df_valC, n_labels=None, Save_Path = Data_path, DataLoader_save_path = 'DataLoader/data_lm_preTrain_wiki.pkl', vocab_save_path = 'vocab/preTrain_wiki_itos.pkl', bs = 32)
    #
    # ## Save the Dataloader
    #
    # save_workpsace(work_dir=Data_path, data_lm=data_lm_preTrain, save_type=['data_lm'],
    #                DataLoaderLM_path='DataLoader/data_lm_preTrain_wiki.pkl')

    # Load the dataloader
    data_lm = load_data(Data_path / 'DataLoader', 'data_lm_preTrain_wiki.pkl')
    # print(awd_lstm_lm_config_custom)
    pre_train(Data_path, data_loader_folder='DataLoader', dataLoader_name='data_lm_preTrain_wiki.pkl',
              model_save_path='DataLoader/models/preTrain_wiki_model',
              encoding_save_path='encoding/preTrain_wiki_encoding',
              cuda_id=0, pretrained=False, custom_config = True, lr = 1e-01, arch = AWD_LSTM)



if args.domain_specific_train:
    # tokenization parameters
    data_path = DATA_DIR / 'yelp'
    input_sub = "tmp"
    train_fname = "train.csv"
    val_fname = "val.csv"
    n_labels = 1
    n_textfield = 1
    chunksize = 100000
    lang = 'en'

    df_trnYelp, df_valYelp = load_data_training(data_path=data_path, n_labels=1, n_textfield=1)
    vocab_update = False

    # data_lmYelp, vocab_update, truncate_val = get_DataLoader(path=data_path, data_trn=df_trnYelp, data_val=df_valYelp, n_labels = n_labels,
    #                                                          use_pretrain_vocab=False, pretrain_path=Data_path / 'vocab',
    #                                                          pretrain_vocab_name='pretrain_wiki_itos.pkl',
    #                                                          no_update_pretrain_vocab=True, max_vocab=70000, Save_Path = Data_path, DataLoader_save_path = 'DataLoader/data_lm_domainSpecific_Yelp.pkl', vocab_save_path = 'vocab/domainSpecific_Yelp_itos.pkl', bs = 32)

    fine_tune(work_dir=Data_path, data_loader_folder='DataLoader', dataLoader_name='data_lm_domainSpecific_Yelp.pkl', arch=AWD_LSTM,
              custom_config=True, pretrained=True,
              pretrained_fnames=[Data_path / 'DataLoader/models/preTrain_wiki_model', Data_path / 'vocab/preTrain_wiki_itos'],
              drop_mult=0.5, opt_func='Adam', loss_func=None, metrics=[accuracy], true_wd=True, bn_wd=True, wd=0.01,
              train_bn=True, lr=1e-01, model_save_path='DataLoader/models/domainSpecific_Yelp_model', encoding_save_path='encoding/domainSpecific_Yelp_encoding',
              cuda_id=0, vocab_weight_balance=vocab_update)


if args.fine_tune:
    data_path = DATA_DIR / 'phase1/bml'

    df_trnRev, df_valRev = load_data_training(data_path=data_path, n_labels=10, n_textfield=1)

    #print(df_trnRev.head())

    data_lmRev, vocab_update, truncate_val = get_DataLoader(path=data_path, data_trn=df_trnRev, data_val=df_valRev,
                                                            use_pretrain_vocab=False, pretrain_path=Data_path / 'vocab',
                                                            pretrain_vocab_name='pretrain_wiki_itos.pkl',
                                                            no_update_pretrain_vocab=True, max_vocab=70000, n_labels=10, Save_Path = Data_path, DataLoader_save_path = 'DataLoader/data_lm_fineTune_Rev.pkl', vocab_save_path = 'vocab/fineTune_Rev_itos.pkl', bs = 8)

    #print(data_lmRev.show_batch())

    fine_tune(work_dir=Data_path, data_loader_folder='DataLoader', dataLoader_name='data_lm_fineTune_Rev.pkl', arch=AWD_LSTM,
              custom_config=True, pretrained=True,
              pretrained_fnames=[Data_path / 'DataLoader/models/domainSpecific_Yelp_model', Data_path / 'vocab/domainSpecific_Yelp_itos'],
              drop_mult=0.5, opt_func='Adam', loss_func=None, metrics=[accuracy], true_wd=True, bn_wd=True, wd=0.01,
              train_bn=True, lr=3e-4, model_save_path='DataLoader/models/modelRevFine',
              vocab_save_path='vocab/itosRevFine.pkl', encoding_save_path='encoding/encoderRevFine',
              DataLoaderLM_save_path='DataLoader/data_lmYelpFine.pkl',
              DataLoaderCl_save_path='DataLoader/data_clYelpFine.pkl', cuda_id=0, vocab_weight_balance=vocab_update)


if args.classifier:
    data_path = DATA_DIR / 'phase1/bml'

    df_trnRev, df_valRev = load_data_training(data_path=data_path, n_labels=10, n_textfield=1)

    dataRev_clas = get_Clas_DataLoader(path=data_path, data_trn=df_trnRev, data_val=df_valRev, n_labels=10,
                                       max_vocab=70000,
                                       min_freq=1,
                                       use_pretrain_vocab=True,
                                       pretrain_path=Data_path / 'DataLoader',
                                       DataLoader_name='data_lm_fineTune_Rev.pkl', Save_Path = Data_path, DataLoader_save_path = 'DataLoader/data_cl_fineTune_Rev.pkl', bs=8)

    train_clas(work_dir=Data_path, data_loader_folder='DataLoader', dataLoader_name='data_cl_fineTune_Rev.pkl', n_labels=10,
               arch=AWD_LSTM, custom_config=True, pretrained=True,
               pretrained_fnames=[Data_path / 'encoding/encoderRevFine', Data_path / 'vocab/fineTune_Rev_itos'],
               drop_mult=0.5, opt_func='Adam', loss_func=None, metrics=[accuracy], true_wd=True, bn_wd=True, wd=0.01,
               train_bn=True, lr=1e-2, model_save_path='DataLoader/models/modelClassRev', encoding_save_path='encoding/encoderClasRev',
               cuda_id=0, use_discriminative=True, chain_thaw=True)







