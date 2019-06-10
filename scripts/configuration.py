from notebooks.chetan.fastaiConsole.fastai.fastai.text import *


# awd_lstm_lm_config = dict(emb_sz=400, n_hid=1150, n_layers=3, pad_token=1, qrnn=False, bidir=False, output_p=0.1,
#                           hidden_p=0.15, input_p=0.25, embed_p=0.02, weight_p=0.2, tie_weights=True, out_bias=True)
#
# awd_lstm_clas_config = dict(emb_sz=400, n_hid=1150, n_layers=3, pad_token=1, qrnn=False, bidir=False, output_p=0.4,
#                        hidden_p=0.3, input_p=0.4, embed_p=0.05, weight_p=0.5)
#
#
# tfmer_lm_config = dict(ctx_len=512, n_layers=12, n_heads=12, d_model=768, d_head=64, d_inner=3072, resid_p=0.1, attn_p=0.1,
#                          ff_p=0.1, embed_p=0.1, output_p=0., bias=True, scale=True, act=Activation.GeLU, double_drop=False,
#                          tie_weights=True, out_bias=False, init=init_transformer, mask=True)
#
# tfmer_clas_config = dict(ctx_len=512, n_layers=12, n_heads=12, d_model=768, d_head=64, d_inner=3072, resid_p=0.1, attn_p=0.1,
#                          ff_p=0.1, embed_p=0.1, output_p=0., bias=True, scale=True, act=Activation.GeLU, double_drop=False,
#                          init=init_transformer, mask=False)
#
#
# tfmerXL_lm_config = dict(ctx_len=150, n_layers=12, n_heads=10, d_model=410, d_head=41, d_inner=2100, resid_p=0.1, attn_p=0.1,
#                          ff_p=0.1, embed_p=0.1, output_p=0.1, bias=False, scale=True, act=Activation.ReLU, double_drop=True,
#                          tie_weights=True, out_bias=True, init=init_transformer, mem_len=150, mask=True)
#
# tfmerXL_clas_config = dict(ctx_len=150, n_layers=12, n_heads=10, d_model=410, d_head=41, d_inner=2100, resid_p=0.1, attn_p=0.1,
#                          ff_p=0.1, embed_p=0.1, output_p=0.1, bias=False, scale=True, act=Activation.ReLU, double_drop=True,
#                          init=init_transformer, mem_len=150, mask=False)

##
awd_lstm_lm_config_custom = awd_lstm_lm_config.copy()
awd_lstm_lm_config_custom['emb_sz'] = 400
awd_lstm_lm_config_custom['n_hid'] = 1150
awd_lstm_lm_config_custom['n_layers'] = 3
awd_lstm_lm_config_custom['pad_token'] = 1
awd_lstm_lm_config_custom['qrnn'] = True
awd_lstm_lm_config_custom['bidir'] = False
awd_lstm_lm_config_custom['output_p'] = 0.1
awd_lstm_lm_config_custom['hidden_p'] = 0.15
awd_lstm_lm_config_custom['input_p'] = 0.25
awd_lstm_lm_config_custom['embed_p'] = 0.02
awd_lstm_lm_config_custom['weight_p'] = 0.2
awd_lstm_lm_config_custom['tie_weights'] = True
awd_lstm_lm_config_custom['out_bias'] = True
##


awd_lstm_clas_config_custom = awd_lstm_clas_config.copy()
awd_lstm_clas_config_custom['emb_sz'] = 400
awd_lstm_clas_config_custom['n_hid'] = 1150
awd_lstm_clas_config_custom['n_layers'] = 3
awd_lstm_clas_config_custom['pad_token'] = 1
awd_lstm_clas_config_custom['qrnn'] = True
awd_lstm_clas_config_custom['bidir'] = False
awd_lstm_clas_config_custom['output_p'] = 0.1
awd_lstm_clas_config_custom['hidden_p'] = 0.15
awd_lstm_clas_config_custom['input_p'] = 0.25
awd_lstm_clas_config_custom['embed_p'] = 0.02
awd_lstm_clas_config_custom['weight_p'] = 0.2


tfmer_lm_config_custom = tfmer_lm_config.copy()
tfmer_lm_config_custom['ctx_len'] = 512
tfmer_lm_config_custom['n_layers'] = 12
tfmer_lm_config_custom['n_heads'] = 12
tfmer_lm_config_custom['d_model'] = 768
tfmer_lm_config_custom['d_head'] = 64
tfmer_lm_config_custom['d_inner'] = 3072
tfmer_lm_config_custom['resid_p'] = 0.1
tfmer_lm_config_custom['attn_p'] = 0.1
tfmer_lm_config_custom['ff_p'] = 0.1
tfmer_lm_config_custom['embed_p'] = 0.1
tfmer_lm_config_custom['output_p'] = 0.0
tfmer_lm_config_custom['bias'] = True
tfmer_lm_config_custom['scale'] = True
tfmer_lm_config_custom['double_drop'] = False
tfmer_lm_config_custom['tie_weights'] = True
tfmer_lm_config_custom['out_bias'] = False
tfmer_lm_config_custom['mask'] = True


tfmer_clas_config_custom = tfmer_clas_config.copy()
tfmer_clas_config_custom['ctx_len'] = 512
tfmer_clas_config_custom['n_layers'] = 12
tfmer_clas_config_custom['n_heads'] = 12
tfmer_clas_config_custom['d_model'] = 768
tfmer_clas_config_custom['d_head'] = 64
tfmer_clas_config_custom['d_inner'] = 3072
tfmer_clas_config_custom['resid_p'] = 0.1
tfmer_clas_config_custom['attn_p'] = 0.1
tfmer_clas_config_custom['ff_p'] = 0.1
tfmer_clas_config_custom['embed_p'] = 0.1
tfmer_clas_config_custom['output_p'] = 0.0
tfmer_clas_config_custom['bias'] = True
tfmer_clas_config_custom['scale'] = True
tfmer_clas_config_custom['double_drop'] = False
tfmer_clas_config_custom['mask'] = True

tfmerXL_lm_config_custom = tfmerXL_lm_config.copy()
tfmerXL_lm_config_custom['ctx_len'] = 150
tfmerXL_lm_config_custom['n_layers'] = 12
tfmerXL_lm_config_custom['n_heads'] = 10
# tfmerXl_lm_config_custom['d_model'] = 410
tfmerXL_lm_config_custom['d_head'] = 41
tfmerXL_lm_config_custom['d_inner'] = 2100
tfmerXL_lm_config_custom['resid_p'] = 0.1
tfmerXL_lm_config_custom['attn_p'] = 0.1
tfmerXL_lm_config_custom['ff_p'] = 0.1
tfmerXL_lm_config_custom['embed_p'] = 0.1
tfmerXL_lm_config_custom['output_p'] = 0.0
tfmerXL_lm_config_custom['bias'] = False
tfmerXL_lm_config_custom['scale'] = True
tfmerXL_lm_config_custom['double_drop'] = True
tfmerXL_lm_config_custom['tie_weights'] = True
tfmerXL_lm_config_custom['out_bias'] = True
tfmerXL_lm_config_custom['mem_len'] = 150
tfmerXL_lm_config_custom['mask'] = True

tfmerXL_clas_config_custom = tfmerXL_clas_config.copy()
tfmerXL_clas_config_custom['ctx_len'] = 150
tfmerXL_clas_config_custom['n_layers'] = 12
tfmerXL_clas_config_custom['n_heads'] = 10
# tfmerXl_clas_config_custom['d_model'] = 410
tfmerXL_clas_config_custom['d_head'] = 41
tfmerXL_clas_config_custom['d_inner'] = 2100
tfmerXL_clas_config_custom['resid_p'] = 0.1
tfmerXL_clas_config_custom['attn_p'] = 0.1
tfmerXL_clas_config_custom['ff_p'] = 0.1
tfmerXL_clas_config_custom['embed_p'] = 0.1
tfmerXL_clas_config_custom['output_p'] = 0.0
tfmerXL_clas_config_custom['bias'] = False
tfmerXL_clas_config_custom['scale'] = True
tfmerXL_clas_config_custom['double_drop'] = True
tfmerXL_clas_config_custom['tie_weights'] = True
tfmerXL_clas_config_custom['out_bias'] = True
tfmerXL_clas_config_custom['mem_len'] = 150
tfmerXL_clas_config_custom['mask'] = True











