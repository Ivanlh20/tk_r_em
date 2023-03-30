#-*- coding: utf-8 -*-
"""
Created on Sun Feb 17 22:30:18 2019

__author__ = "Ivan Lobato"
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.append('E:/Neural_network/nt_tf_lib')
sys.path.append('/media/hdd_1/nt_tf_lib')
sys.path.append('/ceph/users/gbz64553/data/Neural_network/nt_tf_lib')

#########################################################################################
import tensorflow as tf

#########################################################################################
import nn_var_glob as nvgb
import nn_fcns_gen as nfge
import nn_fcns_lays as nfly
import nn_fcns_nets as nnet
import nn_fcns_losses as nfls
import nn_fcns_callbacks as nfcb
import nn_fcns_optimizers as nfop

#########################################################################################
x_typ_mat = [np.uint16]
y_typ_mat = [np.uint16]

x_typ_rtf = [tf.uint16]
y_typ_rtf = [tf.uint16]

x_typ_tf = [tf.float32]
y_typ_tf = [tf.float32]

x_ndim = [4]
y_ndim = [4]

#########################################################################################
EE_STD_X = 0.1
EE_W_Y = 0.1
EE_STD_LCN = 0.1
GRAD_MAX = 100.00

#########################################################################################
# multi-local constrast normalization loss: I found out a proportion between mean(MLN(2+4+8+16))/L1 = 2.62 for EE_STD = 0.1
MLWT_KSZ = [2, 4, 8, 16]
MLWT_WGTL = np.array([1.0, 1.33, 1.66, 2.0], np.float32)
MLWT_WGTL /= MLWT_WGTL.sum()

#########################################################################################
FFT_N = 0.1250
FFT_FR = 2.0*np.power(256.0, -FFT_N)

#########################################################################################
# DOWNSCALING factor
DS_FTR = 1
RS_0 = 64
RS_E = 192
V_MIN = 1.0e-9

R_FTR_OPT = 0 # 0: Deactivated, 1:Fix value, 2:Increasing, 3:Random
R_FTR_VAL = 1.0

#########################################################################################
# Loss normalization
LOSS_NORM = True
MATLAB_EXPORT = False

#########################################################################################
GAN_TYP = 0 # 0:deactivated, 1: Active
# 0: gan - gen
# 1: gan - disc
# 2: L1
# 3: L2
# 4: L1 - MLWT
# 5: L1 - FFT
# 6: mean
# 7: std

#########################################################################################
####################################################	0,    1,      2,     3 ,       4,      5,     6,      7
LOSS_TYP, LOSS_WGT_TYP = nfge.fcn_loss_type_weight([5.0e-4, 1.0000, 10.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000], 1.0)

#########################################################################################
###################################### database #########################################
#########################################################################################
def fcn_g_mean_std(y, axis, std_min):
	y_mean, y_var = tf.nn.moments(y, axes=axis, keepdims=True)
	y_std = tf.math.maximum(tf.sqrt(y_var), std_min)

	return y_mean, y_std

def fcn_wt_x(x, axis, std_min):
	x_sft, x_sc = fcn_g_mean_std(x, axis, std_min)
	x_t = (x - x_sft)/x_sc

	return x_t

def fcn_wt_xy(x, y, axis, std_min):
	x_sft, x_sc = fcn_g_mean_std(x, axis, std_min)
	x_t = (x - x_sft)/x_sc
	y_t = (y - x_sft)/x_sc

	return x_t, y_t

#########################################################################################
def fcn_scale_inputs_x_y(x, y, axis):
	sft, sc = fcn_g_mean_std(x, axis, EE_STD_X)
	x = (x - sft)/sc
	y = (y - sft)/sc

	return x, y

#########################################################################################
@tf.function
def fcn_ds_map(record_string, bb_norm, bb_aug):
	data = nfge.fcn_ds_parse_sgl_ex_string_string(record_string)

	x = nfge.fcn_string_dc(data['x'], x_typ_rtf[0], x_typ_tf[0])
	x = nfge.fcn_x_reshape(x)

	y = nfge.fcn_string_dc(data['y'], y_typ_rtf[0], y_typ_tf[0])
	y = nfge.fcn_y_reshape(y)

	if bb_norm:
		x, y = fcn_scale_inputs_x_y(x, y, [0, 1])

	return x, y

#########################################################################################
if R_FTR_OPT==0:
	def fcn_x(x, y, ibg, ibg_m):
		return x

elif R_FTR_OPT==1:
	def fcn_x(x, y, ibg, ibg_m):
		r = R_FTR_VAL
		x_op = tf.expand_dims(x, axis=-1)
		x_op = y + r*(x_op - y)

		return x_op

elif R_FTR_OPT==2:
	def fcn_x(x, y, ibg, ibg_m):
		r = tf.minimum(1.0, tf.maximum(0.0, ibg/ibg_m))
		x_op = tf.expand_dims(x, axis=-1)
		x_op = y + r*(x_op - y)

		return x_op

else:
	def fcn_x(x, y, ibg, ibg_m):
		r = tf.random.uniform((), dtype=tf.float32)
		x_op = tf.expand_dims(x[..., 0], axis=-1)
		x_op = y + r*(x_op - y)

		return x_op

def fcn_gn(x, sigma, ftr_min, ftr_max):
	r = tf.random.uniform(tf.shape(sigma), minval=ftr_min, maxval=ftr_max, dtype=x_typ_tf[0], seed=2000)
	return x + tf.random.normal(tf.shape(x), 0.0, r*sigma, dtype=x_typ_tf[0])

def fcn_input_disc(x, y, y_p):
	bb_noise = tf.less(tf.random.uniform((), dtype=tf.float32, seed=2001), 0.50)
	if bb_noise:
		sigma = tf.maximum(0.01, tf.math.reduce_std(y, axis=[1, 2], keepdims=True))	
		y = fcn_gn(y, sigma, 0.001, 0.10)
		y_p = fcn_gn(y_p, sigma, 0.001, 0.10)

	return tf.concat([x, y], axis=-1), tf.concat([x, y_p], axis=-1)

if nvgb.isone(FFT_N):
	def fcn_pow_fft_n(y):
		return y
elif nvgb.iszero(FFT_N-0.5):
	def fcn_pow_fft_n(y):
		return tf.sqrt(y)

else:
	def fcn_pow_fft_n(y):
		return tf.pow(tf.math.maximum(V_MIN, y), FFT_N)

def fcn_sft_std(x, kn_sz):
	# local constrast normalization
	x_sft = tf.nn.avg_pool2d(x, kn_sz, strides=(1, 1), padding='SAME')
	x_std = tf.nn.avg_pool2d(tf.math.squared_difference(x, x_sft), kn_sz, strides=(1, 1), padding='SAME')
	x_std = tf.math.maximum(tf.sqrt(x_std), EE_STD_LCN)
	# # Gaussian smoothing
	# x_sft = tfa.image.gaussian_filter2d(x_sft, filter_shape=2*kn_sz[0], sigma = 0.5*kn_sz[0], padding='REFLECT')
	# x_std = tfa.image.gaussian_filter2d(x_std, filter_shape=2*kn_sz[0], sigma = 0.5*kn_sz[0], padding='REFLECT')

	return x_sft, x_std

def fcn_mlwt(y_t, y_p, kn_sz):
	# get smooth shift and scaling
	x_sft, x_sc = fcn_sft_std(y_t, kn_sz)
	# normalization
	y_t = (y_t - x_sft)/x_sc
	y_p = (y_p - x_sft)/x_sc
	# whitening transform
	y_t, y_p = fcn_wt_xy(y_t, y_p, [1, 2], EE_W_Y)
	return y_t, y_p

if LOSS_NORM:
	def fcn_g_weight(y):
		y_std = tf.math.reduce_std(y, axis=[1, 2], keepdims=True)
		y_std = tf.math.maximum(y_std, EE_W_Y)
		y_w = 1.0/y_std
		return y_w

	def fcn_g_apply_weight(y_t, y_p):
		w = fcn_g_weight(y_t)

		# apply weights
		y_t = w*y_t
		y_p = w*y_p

		return y_t, y_p
else:
	def fcn_g_apply_weight(y_t, y_p):
		return y_t, y_p

#########################################################################################
def fmet_l1(y_true, y_pred):
	return nfls.fls_mae(y_true, y_pred)

#########################################################################################
def fls_l1(y_true, y_pred):
	return nfls.fls_mae(y_true, y_pred)

#########################################################################################
def fls_l2(y_true, y_pred):
	loss = tf.math.squared_difference(y_true, y_pred)
	return tf.reduce_mean(tf.math.real(loss))

#########################################################################################
def fls_l1_mlwt(y_true, y_pred):
	# whitening transform
	y_t, y_p = fcn_wt_xy(y_true, y_pred, [1, 2], EE_W_Y)

	# multilocal whitening transform
	y_t_t, y_p_t = fcn_mlwt(y_t, y_p, (MLWT_KSZ[0], MLWT_KSZ[0]))
	loss = MLWT_WGTL[0]*nfls.fls_mae(y_t_t, y_p_t)

	y_t_t, y_p_t = fcn_mlwt(y_t, y_p,(MLWT_KSZ[1], MLWT_KSZ[1]))
	loss += MLWT_WGTL[1]*nfls.fls_mae(y_t_t, y_p_t)

	y_t_t, y_p_t = fcn_mlwt(y_t, y_p, (MLWT_KSZ[2], MLWT_KSZ[2]))
	loss += MLWT_WGTL[2]*nfls.fls_mae(y_t_t, y_p_t)

	y_t_t, y_p_t = fcn_mlwt(y_t, y_p, (MLWT_KSZ[3], MLWT_KSZ[3]))
	loss += MLWT_WGTL[3]*nfls.fls_mae(y_t_t, y_p_t)

	return loss

#########################################################################################
def fls_l1_fft(y_true, y_pred):
	loss = FFT_FR*(y_true[..., -1]-y_pred[..., -1]) # in is place in this position in order to avoid numerical overflow
	loss = tf.abs(tf.signal.rfft2d(loss))
	loss = fcn_pow_fft_n(loss)
	return tf.reduce_mean(loss)

#########################################################################################
def fls_l1_mean(y_true, y_pred):
	loss = tf.abs(tf.reduce_mean(y_true, axis=[1, 2]) - tf.reduce_mean(y_pred, axis=[1, 2]))
	return tf.reduce_mean(loss)

#########################################################################################
def fls_l1_std(y_true, y_pred):
	loss = tf.abs(tf.math.reduce_std(y_true, axis=[1, 2]) - tf.math.reduce_std(y_pred, axis=[1, 2]))
	return tf.reduce_mean(loss)

#########################################################################################
if LOSS_TYP[2]:
	def fls_l1_w(y_true, y_pred):
		loss = fls_l1(y_true, y_pred)
		loss_w = LOSS_WGT_TYP[2]*loss

		return loss_w, loss
else:
	def fls_l1_w(y_true, y_pred):
		return tf.constant(0, tf.float32), tf.constant(0, tf.float32)

if LOSS_TYP[3]:
	def fls_l2_w(y_true, y_pred):
		loss = fls_l2(y_true, y_pred)
		loss_w = LOSS_WGT_TYP[3]*loss

		return loss_w, loss
else:
	def fls_l2_w(y_true, y_pred):
		return tf.constant(0, tf.float32), tf.constant(0, tf.float32)

if LOSS_TYP[4]:
	def fls_l1_mlwt_w(y_true, y_pred):
		loss = fls_l1_mlwt(y_true, y_pred)
		loss_w = LOSS_WGT_TYP[4]*loss

		return loss_w, loss
else:
	def fls_l1_mlwt_w(y_true, y_pred):
		return tf.constant(0, tf.float32), tf.constant(0, tf.float32)

if LOSS_TYP[5]:
	def fls_l1_fft_w(y_true, y_pred):
		loss = fls_l1_fft(y_true, y_pred)
		loss_w = LOSS_WGT_TYP[5]*loss

		return loss_w, loss
else:
	def fls_l1_fft_w(y_true, y_pred):
		return tf.constant(0, tf.float32), tf.constant(0, tf.float32)

#########################################################################################
if LOSS_TYP[6]:
	def fls_l1_mean_w(y_true, y_pred):
		loss = fls_l1_mean(y_true, y_pred)
		loss_w = LOSS_WGT_TYP[6]*loss

		return loss_w, loss
else:
	def fls_l1_mean_w(y_true, y_pred):
		return tf.constant(0, tf.float32), tf.constant(0, tf.float32)

if LOSS_TYP[7]:
	def fls_l1_std_w(y_true, y_pred):
		loss = fls_l1_std(y_true, y_pred)
		loss_w = LOSS_WGT_TYP[7]*loss

		return loss_w, loss	
else:
	def fls_l1_std_w(y_true, y_pred):
		return tf.constant(0, tf.float32), tf.constant(0, tf.float32)		

#########################################################################################
def fls_pw_w(y_t_i, y_p_i):
	y_t, y_p = fcn_g_apply_weight(y_t_i, y_p_i)

	loss_l1_w, loss_l1 = fls_l1_w(y_t, y_p)

	loss_l2_w, loss_l2 = fls_l2_w(y_t, y_p)

	loss_l1_mlwt_w, loss_l1_mlwt = fls_l1_mlwt_w(y_t, y_p)

	loss_l1_fft_w, loss_l1_fft = fls_l1_fft_w(y_t, y_p)

	loss_l1_mean_w, loss_l1_mean = fls_l1_mean_w(y_t, y_p)

	loss_l1_std_w, loss_l1_std = fls_l1_std_w(y_t, y_p)

	loss_pw_w = loss_l1_w + loss_l2_w + loss_l1_mlwt_w  + loss_l1_fft_w + loss_l1_mean_w + loss_l1_std_w

	met_l1 = fmet_l1(y_t_i, y_p_i)

	return {'loss_pw_w': loss_pw_w, \
			'loss_l1_w': loss_l1_w, 'loss_l1': loss_l1, \
			'loss_l2_w': loss_l2_w, 'loss_l2': loss_l2, \
			'loss_l1_mlwt_w': loss_l1_mlwt_w, 'loss_l1_mlwt': loss_l1_mlwt, \
			'loss_l1_fft_w': loss_l1_fft_w, 'loss_l1_fft': loss_l1_fft, \
			'loss_l1_mean_w': loss_l1_mean_w, 'loss_l1_mean': loss_l1_mean,\
			'loss_l1_std_w': loss_l1_std_w, 'loss_l1_std': loss_l1_std,\
			'met_l1': met_l1}

#########################################################################################
def fls_adv(y_d_real, y_d_gen):
	y_real = y_d_real - tf.reduce_mean(y_d_gen)
	y_gen = y_d_gen - tf.reduce_mean(y_d_real)

	loss_real = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(y_real), y_real)
	loss_gen = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(y_gen), y_gen)

	loss = loss_real + loss_gen
	return loss

def fls_adv_w(y_d_real, y_d_gen):
	loss = fls_adv(y_d_real, y_d_gen)
	loss_w = LOSS_WGT_TYP[0]*loss

	return loss_w, loss

#########################################################################################
def fls_disc(y_d_real, y_d_gen):
	y_real = y_d_real - tf.reduce_mean(y_d_gen)
	y_gen = y_d_gen - tf.reduce_mean(y_d_real)

	d = 0.05
	y_ones = tf.ones_like(y_real)-d
	loss_real = tf.keras.losses.BinaryCrossentropy(from_logits=True)(y_ones, y_real)
	loss_gen = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(y_gen), y_gen)

	loss = loss_real + loss_gen
	return loss

def fls_disc_w(y_d_real, y_d_gen):
	loss = fls_disc(y_d_real, y_d_gen)
	loss_w = LOSS_WGT_TYP[1]*loss

	return loss_w, loss

#########################################################################################
##################################### generator #########################################
#########################################################################################
# CGRDN global residual connection between input and output
def fcn_net_gen_v1(x, ftrs_i, ftrs_o, ftrs_g, ftrs_btn, kn_sz_dus, stri_dus, kn_sz, act_str, n_lays, n_rdb, n_grdb, name, parm_init=None, parm_reg=None, parm_cstr=None):
	dilt_rt = 1
	parm_dp = nvgb.fcn_read_dropout_parm_dict()
	parm_norm = nvgb.fcn_read_normalization_parm_dict()
	dsn_typ = nvgb.DENSENET_TYPE
	use_bi = nfly.fcn_use_bias_norm_parm(parm_norm)

	act_str_fst = None
	act_str_down = None
	act_str_up = None
	act_str_last = None

	dpr_tra_0, dpr_tra_dn, dpr_tra_e, dpr_dn_1, dpr_dn_2 = nfly.fcn_set_dn_dropout(parm_dp['dp_opt'], parm_dp['dp_rt'])[0:5]

	# global residual
	x_i = x

	# first
	x = nfly.fcn_conv_2d(x, ftrs_i, kn_sz, (1, 1), act_str_fst, dilt_rt, True, 'same', dpr_tra_0, name + 'fst_0', parm_norm, parm_init, parm_reg, parm_cstr)
	
	# downsampling
	x = nfly.fcn_conv_2d_gen(x, ftrs_i, kn_sz_dus, stri_dus, act_str_down, dilt_rt, use_bi, 'same', dpr_tra_0, name + 'down_spl_0', parm_norm, parm_init, parm_reg, parm_cstr)
	
	# middle	
	x_skc = x

	x_cc = []
	for ik in range(n_grdb):
		name_grdb = name + 'g_' + str(ik+1)
		x = nnet.fcn_grdn(x, ftrs_g, ftrs_btn, kn_sz, act_str, n_lays, n_rdb, dilt_rt, (dpr_dn_1, dpr_dn_2), dsn_typ, name_grdb, parm_norm, parm_init, parm_reg, parm_cstr)
		x_cc.append(x)

	x_cc = tf.keras.layers.Concatenate(axis=3, name=name + 'g_concat')(x_cc)
	x = nfly.fcn_conv_2d(x_cc, ftrs_i, (1, 1), (1, 1), nvgb.DENSENET_FSL_ACT, 1, True, 'same', 0.0, name + 'g_fsl', parm_norm, parm_init, parm_reg, parm_cstr)
	x = tf.keras.layers.Add(name=name + 'g_add')([x, x_skc])

	# upsampling
	x = nfly.fcn_dconv_2d_gen(x, ftrs_i, kn_sz_dus, stri_dus, act_str_up, dilt_rt, use_bi, 'same', dpr_tra_e, name + 'up_spl_0', parm_norm, parm_init, parm_reg, parm_cstr)

	# last
	x = nfly.fcn_conv_2d(x, ftrs_o, kn_sz, (1, 1), act_str_last, dilt_rt, True, 'same', 0.0, name + 'last_0', parm_norm, parm_init, parm_reg={'reg_wtyp': 0, 'reg_atyp': 0}, parm_cstr={'cstr_typ': 0})

	# global residual
	x = tf.keras.layers.Add(name=name + 'add_0')([x, x_i])

	return x

def nn_model_gen(input_shape, input_name, prefix_layer, model_name):
	modv = nvgb.MODV % 10
 
	if modv==1:
		n_lays = 4
		n_rdb = 4
		n_grdb = 4
		ftrs_g = 16
	elif modv==2:
		n_lays = 4
		n_rdb = 4
		n_grdb = 4
		ftrs_g = 24
	elif modv==3:
		n_lays = 4
		n_rdb = 4
		n_grdb = 4
		ftrs_g = 32
	elif modv==4:
		n_lays = 9
		n_rdb = 4
		n_grdb = 4
		ftrs_g = 32
	elif modv==5:
		n_lays = 4
		n_rdb = 4
		n_grdb = 4
		ftrs_g = 32
	elif modv==6:
		n_lays = 5
		n_rdb = 4
		n_grdb = 4
		ftrs_g = 32
	elif modv==7:
		n_lays = 6
		n_rdb = 4
		n_grdb = 4
		ftrs_g = 32
	elif modv==8:
		n_lays = 7
		n_rdb = 4
		n_grdb = 4
		ftrs_g = 32
	elif modv==9:
		n_lays = 8
		n_rdb = 4
		n_grdb = 4
		ftrs_g = 32

	ftrs_i = 64
	ftrs_btn = nvgb.DENSENET_BOTTLENECK_FR*ftrs_g
	ftrs_o = nvgb.Y_SHAPE[2]
	kn_sz_dus = (4, 4)
	stri_dus = (2, 2)
	kn_sz = (3, 3)
	act_str = nvgb.ACTIVATION_STR

	x_i = tf.keras.layers.Input(shape=input_shape, name=input_name, dtype='float32')

	x = fcn_net_gen_v1(x_i, ftrs_i, ftrs_o, ftrs_g, ftrs_btn, kn_sz_dus, stri_dus, kn_sz, act_str, n_lays, n_rdb, n_grdb, prefix_layer)
 
	return tf.keras.models.Model(inputs=x_i, outputs=x, name=model_name)

################################### discriminator #######################################
def fcn_net_disc_v1(x, ftrs_o, act_str, name, parm_norm=None, parm_init=None, parm_reg=None, parm_cstr=None):
	parm_norm = {'norm_typ': 1, 'norm_pos': 2, 'norm_m': 0.95, 'norm_eps': 1e-3}
	parm_init = {'init_typ': 7, 'init_sfr': 0.02}
	parm_reg = {'reg_wtyp': 0, 'reg_wkn': 2e-6, 'reg_wkn': 2e-5}
	parm_cstr = None

	ftrs_i = 64
	kn_sz = (4, 4)
	dp_rt = 0.0

	dilt_rt = 1
	x = nfly.fcn_conv_2d(x, ftrs_i, kn_sz, (2, 2), act_str, dilt_rt, False, 'same', 0.0, name + 'downspl_1', parm_norm, parm_init, parm_reg, parm_cstr)							# (bs, 128, 128, 1*ftrs_i)
	x = nfly.fcn_conv_2d_bna(x, 2*ftrs_i, kn_sz, (2, 2), act_str, dilt_rt, False, 'same', dp_rt, name + 'downspl_2', parm_norm, parm_init, parm_reg, parm_cstr)					# (bs, 64, 64, 2*ftrs_i)
	x = nfly.fcn_conv_2d_bna(x, 4*ftrs_i, kn_sz, (2, 2), act_str, dilt_rt, False, 'same', dp_rt, name + 'downspl_3', parm_norm, parm_init, parm_reg, parm_cstr)					# (bs, 32, 32, 4*ftrs_i)
	x = nfly.fcn_conv_2d_bna(x, 8*ftrs_i, kn_sz, (1, 1), act_str, dilt_rt, False, 'same', dp_rt, name + 'middle', parm_norm, parm_init, parm_reg, parm_cstr)						# (bs, 32, 32, 8*ftrs_i)
	x = nfly.fcn_conv_2d(x, ftrs_o, kn_sz, (1, 1), None, dilt_rt, True, 'same', 0.0, name + 'last', parm_norm, parm_init, parm_reg={'reg_wtyp': 0, 'reg_atyp': 0}, parm_cstr={'cstr_typ': 0})	# (bs, 32, 32, 1)

	return x

def nn_model_disc(input_shape, input_name, prefix_layer, model_name):
	ftrs_o = 1
	act_str = 'leaky_relu'

	x_i = tf.keras.layers.Input(shape=input_shape, name=input_name, dtype='float32')
	x = fcn_net_disc_v1(x_i, ftrs_o, act_str, prefix_layer)

	return tf.keras.models.Model(inputs=x_i, outputs=x, name=model_name)

#########################################################################################
######################################## model ##########################################
#########################################################################################
if GAN_TYP == 1:
	class My_model(tf.keras.Model):
		def __init__(self, input_shape, *args, **kwargs):
			super(My_model, self).__init__(*args, **kwargs)
	
			self.gen = nn_model_gen(input_shape, 'input_gen', 'gen_', 'nEM_model_rest')
			self.disc = nn_model_disc((input_shape[0], input_shape[1], 2), 'input_disc', 'disc_', 'nEM_model_disc')


			self.met_l1 = tf.keras.metrics.Mean(name="met_l1")
			self.val_met_l1= tf.keras.metrics.Mean(name="met_l1")

			self.loss_pw_w = tf.keras.metrics.Mean(name="loss_pw_w")
			self.val_loss_pw_w = tf.keras.metrics.Mean(name="loss_pw_w")

			self.loss_gen_w = tf.keras.metrics.Mean(name="loss_gen_w")
			self.val_loss_gen_w = tf.keras.metrics.Mean(name="loss_gen_w")

			self.loss_gen_reg = tf.keras.metrics.Mean(name="loss_gen_reg")
			self.val_loss_gen_reg = tf.keras.metrics.Mean(name="loss_gen_reg")

			self.loss_disc_adv_reg = tf.keras.metrics.Mean(name="loss_disc_adv_reg")
			self.val_loss_disc_adv_reg = tf.keras.metrics.Mean(name="loss_disc_adv_reg")

			if LOSS_TYP[0]:
				self.loss_gen_adv_w = tf.keras.metrics.Mean(name="loss_gen_adv_w")
				self.loss_gen_adv = tf.keras.metrics.Mean(name="loss_gen_adv")
				self.val_loss_gen_adv_w = tf.keras.metrics.Mean(name="loss_gen_adv_w")
				self.val_loss_gen_adv = tf.keras.metrics.Mean(name="val_loss_gen_adv")

			if LOSS_TYP[1]:
				self.loss_disc_adv_w = tf.keras.metrics.Mean(name="loss_disc_adv_w")
				self.loss_disc_adv = tf.keras.metrics.Mean(name="loss_disc_adv")
				self.val_loss_disc_adv_w = tf.keras.metrics.Mean(name="loss_disc_adv_w")
				self.val_loss_disc_adv = tf.keras.metrics.Mean(name="loss_disc_adv")

			if LOSS_TYP[2]:
				self.loss_l1_w = tf.keras.metrics.Mean(name="loss_l1_w")
				self.loss_l1 = tf.keras.metrics.Mean(name="loss_l1")
				self.val_loss_l1_w = tf.keras.metrics.Mean(name="loss_l1_w")
				self.val_loss_l1 = tf.keras.metrics.Mean(name="loss_l1")

			if LOSS_TYP[3]:
				self.loss_l2_w = tf.keras.metrics.Mean(name="loss_l2_w")
				self.loss_l2 = tf.keras.metrics.Mean(name="loss_l2")
				self.val_loss_l2_w = tf.keras.metrics.Mean(name="loss_l2_w")
				self.val_loss_l2 = tf.keras.metrics.Mean(name="loss_l2")

			if LOSS_TYP[4]:
				self.loss_l1_mlwt_w = tf.keras.metrics.Mean(name="loss_l1_mlwt_w")
				self.loss_l1_mlwt = tf.keras.metrics.Mean(name="loss_l1_mlwt")
				self.val_loss_l1_mlwt_w = tf.keras.metrics.Mean(name="loss_l1_mlwt_w")
				self.val_loss_l1_mlwt = tf.keras.metrics.Mean(name="loss_l1_mlwt")

			if LOSS_TYP[5]:
				self.loss_l1_fft_w = tf.keras.metrics.Mean(name="loss_l1_fft_w")
				self.loss_l1_fft = tf.keras.metrics.Mean(name="loss_l1_fft")
				self.val_loss_l1_fft_w = tf.keras.metrics.Mean(name="loss_l1_fft_w")
				self.val_loss_l1_fft = tf.keras.metrics.Mean(name="loss_l1_fft")

			if LOSS_TYP[6]:
				self.loss_l1_mean_w = tf.keras.metrics.Mean(name="loss_l1_mean_w")
				self.loss_l1_mean = tf.keras.metrics.Mean(name="loss_l1_mean")
				self.val_loss_l1_mean_w = tf.keras.metrics.Mean(name="loss_l1_mean_w")
				self.val_loss_l1_mean = tf.keras.metrics.Mean(name="loss_l1_mean")

			if LOSS_TYP[7]:
				self.loss_l1_std_w = tf.keras.metrics.Mean(name="loss_l1_std_w")
				self.loss_l1_std = tf.keras.metrics.Mean(name="loss_l1_std")
				self.val_loss_l1_std_w = tf.keras.metrics.Mean(name="loss_l1_std_w")
				self.val_loss_l1_std = tf.keras.metrics.Mean(name="loss_l1_std")


			self.ibg = tf.Variable(0.0, dtype=tf.float32, trainable=False)
			self.ibg_m = 2*nvgb.TRAIN_N_DATA//nvgb.BATCH_SIZE

		def compile(self, parm):
			super(My_model, self).compile()

			self.gen_opt = nfop.fcn_get_optimizer_from_vgb()

			self.lr_schedule_gen = nfcb.Cb_Lr_schedule_base(
					lr_min=parm['lr_min'],
					lr_max=parm['opt_lr'],
					m_min=parm['m_0'],
					m_max=parm['opt_m'],
					decay_steps=parm['decay_steps'],
					decay_rate=parm['decay_rate'],
					steps_per_cycle=parm['decay_steps'],
					warmup_steps=parm['warmup_steps'],
					cooldown_steps=parm['cooldown_steps'],
					lr_0=parm['lr_0'], 
					warmup_steps_0=parm['warmup_steps_0'], 
					cooldown_steps_0=parm['cooldown_steps_0'],
					decay_steps_0=parm['decay_steps_0'],
					lrs_m_pow=parm['lrs_m_pow'],
					lrs_lr_pow=parm['lrs_lr_pow'])

			#########################################################################################
			opt_typ, opt_lr, opt_m = 3, nvgb.OPTIMIZER_LR, 0.5
			
			self.disc_opt = nfop.fcn_get_optimizer(opt_typ=opt_typ, opt_lr=opt_lr,
						opt_m=opt_m, opt_nesterov=nvgb.OPTIMIZER_NESTEROV, 
						opt_beta_2=nvgb.OPTIMIZER_BETA_2, opt_eps=nvgb.OPTIMIZER_EPSILON,
						opt_ctyp=0, opt_cval=nvgb.OPTIMIZER_CLIP_VALUE)

			self.lr_schedule_disc = nfcb.Cb_Lr_schedule_base(
					lr_min=1e-4*opt_lr,
					lr_max=opt_lr,
					m_min=0.0,
					m_max=opt_m,
					decay_steps=parm['decay_steps'],
					decay_rate=parm['decay_rate'],
					steps_per_cycle=parm['decay_steps'],
					warmup_steps=parm['warmup_steps'],
					cooldown_steps=parm['cooldown_steps'],
					lr_0=1e-4*opt_lr, 
					warmup_steps_0=parm['warmup_steps_0'], 
					cooldown_steps_0=parm['cooldown_steps_0'],
					decay_steps_0=parm['decay_steps_0'],
					lrs_m_pow=parm['lrs_m_pow'],
					lrs_lr_pow=parm['lrs_lr_pow'])
		
		def reset_opt_iter(self):
			tf.keras.backend.set_value(self.gen_opt.iterations, 0)
			tf.keras.backend.set_value(self.disc_opt.iterations, 0)

		def set_opt_lr_m(self):
			self.lr_schedule_gen.get_set_opt_lr_m(self.gen_opt)
			self.lr_schedule_disc.get_set_opt_lr_m(self.disc_opt)

		def inc_opt_counter(self):
			self.lr_schedule_gen.inc_counter()
			self.lr_schedule_disc.inc_counter()

		def call(self, inputs, training=None, mask=None):
			return self.gen(inputs, training)

		@tf.function
		def train_step(self, data):
			x, y = data

			x = fcn_x(x, y, self.ibg, self.ibg_m)

			with tf.GradientTape() as tape_gen, tf.GradientTape() as tape_disc:
				# Forward pass
				y_p = self.gen(x, training=True)

				# pixelwise loss
				loss_gen_pw_dict = fls_pw_w(y, y_p)

				x_d, x_d_p = fcn_input_disc(x, y, y_p)
				y_d = self.disc(x_d, training=True)
				y_d_p = self.disc(x_d_p, training=True)

				# gan-gen loss
				loss_gen_adv_w, loss_gen_adv = fls_adv_w(y_d, y_d_p)

				# gen loss
				loss_gen_w = loss_gen_adv_w + loss_gen_pw_dict['loss_pw_w']

				# gan-disc loss
				loss_disc_adv_w, loss_disc_adv = fls_disc_w(y_d, y_d_p)

				# regularization loss
				loss_gen_reg = tf.reduce_sum(self.gen.losses)
				loss_disc_adv_reg = tf.reduce_sum(self.disc.losses)

				loss_gen_t = loss_gen_w + loss_gen_reg
				loss_disc_adv_t = loss_disc_adv_w + loss_disc_adv_reg

			# Compute gradient
			grad_gen_t = tape_gen.gradient(loss_gen_t, self.gen.trainable_variables)
			grad_disc_t = tape_disc.gradient(loss_disc_adv_t, self.disc.trainable_variables)

			#clip gradients
			grad_gen_t = nfop.fcn_optimizer_clip_gradients(self.gen_opt, grad_gen_t)
			grad_disc_t = nfop.fcn_optimizer_clip_gradients(self.disc_opt, grad_disc_t)

			# Update gradient
			self.gen_opt.apply_gradients(zip(grad_gen_t, self.gen.trainable_variables))
			self.disc_opt.apply_gradients(zip(grad_disc_t, self.disc.trainable_variables))

			# save metrics
			self.met_l1.update_state(loss_gen_pw_dict['met_l1'])
			metrics_out = {'met_l1': self.met_l1.result()}

			# save losses
			self.loss_pw_w.update_state(loss_gen_pw_dict['loss_pw_w'])
			metrics_out.update({'loss_pw_w': self.loss_pw_w.result()})
	
			self.loss_gen_w.update_state(loss_gen_w)
			metrics_out.update({'loss_gen_w': self.loss_gen_w.result()})

			self.loss_gen_reg.update_state(loss_gen_reg)
			metrics_out.update({'loss_gen_reg': self.loss_gen_reg.result()})

			self.loss_disc_adv_reg.update_state(loss_disc_adv_reg)
			metrics_out.update({'loss_disc_adv_reg': self.loss_disc_adv_reg.result()})
	

			if LOSS_TYP[0]:
				self.loss_gen_adv_w.update_state(loss_gen_adv_w)
				self.loss_gen_adv.update_state(loss_gen_adv)
				metrics_out.update({'loss_gen_adv_w': self.loss_gen_adv_w.result(), 'loss_gen_adv': self.loss_gen_adv.result()})
	
			if LOSS_TYP[1]:
				self.loss_disc_adv_w.update_state(loss_disc_adv_w)
				self.loss_disc_adv.update_state(loss_disc_adv)
				metrics_out.update({'loss_disc_adv_w': self.loss_disc_adv_w.result(), 'loss_disc_adv': self.loss_disc_adv.result()})

			if LOSS_TYP[2]:
				self.loss_l1_w.update_state(loss_gen_pw_dict['loss_l1_w'])
				self.loss_l1.update_state(loss_gen_pw_dict['loss_l1'])
				metrics_out.update({'loss_l1_w': self.loss_l1_w.result(), 'loss_l1': self.loss_l1.result()})

			if LOSS_TYP[3]:
				self.loss_l2_w.update_state(loss_gen_pw_dict['loss_l2_w'])
				self.loss_l2.update_state(loss_gen_pw_dict['loss_l2'])
				metrics_out.update({'loss_l2_w': self.loss_l2_w.result(), 'loss_l2': self.loss_l2.result()})

			if LOSS_TYP[4]:
				self.loss_l1_mlwt_w.update_state(loss_gen_pw_dict['loss_l1_mlwt_w'])
				self.loss_l1_mlwt.update_state(loss_gen_pw_dict['loss_l1_mlwt'])
				metrics_out.update({'loss_l1_mlwt_w': self.loss_l1_mlwt_w.result(), 'loss_l1_mlwt': self.loss_l1_mlwt.result()})

			if LOSS_TYP[5]:
				self.loss_l1_fft_w.update_state(loss_gen_pw_dict['loss_l1_fft_w'])
				self.loss_l1_fft.update_state(loss_gen_pw_dict['loss_l1_fft'])
				metrics_out.update({'loss_l1_fft_w': self.loss_l1_fft_w.result(), 'loss_l1_fft': self.loss_l1_fft.result()})

			if LOSS_TYP[6]:
				self.loss_l1_mean_w.update_state(loss_gen_pw_dict['loss_l1_mean_w'])
				self.loss_l1_mean.update_state(loss_gen_pw_dict['loss_l1_mean'])
				metrics_out.update({'loss_l1_mean_w': self.loss_l1_mean_w.result(), 'loss_l1_mean': self.loss_l1_mean.result()})

			if LOSS_TYP[7]:
				self.loss_l1_std_w.update_state(loss_gen_pw_dict['loss_l1_std_w'])
				self.loss_l1_std.update_state(loss_gen_pw_dict['loss_l1_std'])
				metrics_out.update({'loss_l1_std_w': self.loss_l1_std_w.result(), 'loss_l1_std': self.loss_l1_std.result()})

			self.ibg.assign_add(1.0)

			return metrics_out

		@tf.function
		def test_step(self, data):
			x, y = data

			# Forward pass
			y_p = self.gen(x, training=False)

			# pixelwise loss
			loss_gen_pw_dict = fls_pw_w(y, y_p)

			x_d, x_d_p = fcn_input_disc(x, y, y_p)
			y_d = self.disc(x_d, training=False)
			y_d_p = self.disc(x_d_p, training=False)

			# gan-gen loss
			loss_gen_adv_w, loss_gen_adv = fls_adv_w(y_d, y_d_p)

			# gen loss
			loss_gen_w = loss_gen_adv_w + loss_gen_pw_dict['loss_pw_w']

			# gan-disc loss
			loss_disc_adv_w, loss_disc_adv = fls_disc_w(y_d, y_d_p)

			# regularization loss
			loss_gen_reg = tf.reduce_sum(self.gen.losses)
			loss_disc_adv_reg = tf.reduce_sum(self.disc.losses)

			# save metrics
			self.val_met_l1.update_state(loss_gen_pw_dict['met_l1'])
			metrics_out = {'met_l1': self.val_met_l1.result()}

			# save losses
			self.val_loss_pw_w.update_state(loss_gen_pw_dict['loss_pw_w'])
			metrics_out.update({'loss_pw_w': self.val_loss_pw_w.result()})
	
			self.val_loss_gen_w.update_state(loss_gen_w)
			metrics_out.update({'loss_gen_w': self.val_loss_gen_w.result()})
	
			self.val_loss_gen_reg.update_state(loss_gen_reg)
			metrics_out.update({'loss_gen_reg': self.val_loss_gen_reg.result()})

			self.val_loss_disc_adv_reg.update_state(loss_disc_adv_reg)
			metrics_out.update({'loss_disc_adv_reg': self.val_loss_disc_adv_reg.result()})

			if LOSS_TYP[0]:
				self.val_loss_gen_adv_w.update_state(loss_gen_adv_w)
				self.val_loss_gen_adv.update_state(loss_gen_adv)
				metrics_out.update({'loss_gen_adv_w': self.val_loss_gen_adv_w.result(), 'loss_gen_adv': self.val_loss_gen_adv.result()})
	
			if LOSS_TYP[1]:
				self.val_loss_disc_adv_w.update_state(loss_disc_adv_w)
				self.val_loss_disc_adv.update_state(loss_disc_adv)
				metrics_out.update({'loss_disc_adv_w': self.val_loss_disc_adv_w.result(), 'loss_disc_adv': self.val_loss_disc_adv.result()})

			if LOSS_TYP[2]:
				self.val_loss_l1_w.update_state(loss_gen_pw_dict['loss_l1_w'])
				self.val_loss_l1.update_state(loss_gen_pw_dict['loss_l1'])
				metrics_out.update({'loss_l1_w': self.val_loss_l1_w.result(), 'loss_l1': self.val_loss_l1.result()})

			if LOSS_TYP[3]:
				self.val_loss_l2_w.update_state(loss_gen_pw_dict['loss_l2_w'])
				self.val_loss_l2.update_state(loss_gen_pw_dict['loss_l2'])
				metrics_out.update({'loss_l2_w': self.val_loss_l2_w.result(), 'loss_l2': self.val_loss_l2.result()})
	
			if LOSS_TYP[4]:
				self.val_loss_l1_mlwt_w.update_state(loss_gen_pw_dict['loss_l1_mlwt_w'])
				self.val_loss_l1_mlwt.update_state(loss_gen_pw_dict['loss_l1_mlwt'])
				metrics_out.update({'loss_l1_mlwt_w': self.val_loss_l1_mlwt_w.result(), 'loss_l1_mlwt': self.val_loss_l1_mlwt.result()})

			if LOSS_TYP[5]:
				self.val_loss_l1_fft_w.update_state(loss_gen_pw_dict['loss_l1_fft_w'])
				self.val_loss_l1_fft.update_state(loss_gen_pw_dict['loss_l1_fft'])
				metrics_out.update({'loss_l1_fft_w': self.val_loss_l1_fft_w.result(), 'loss_l1_fft': self.val_loss_l1_fft.result()})

			if LOSS_TYP[6]:
				self.val_loss_l1_mean_w.update_state(loss_gen_pw_dict['loss_l1_mean_w'])
				self.val_loss_l1_mean.update_state(loss_gen_pw_dict['loss_l1_mean'])
				metrics_out.update({'loss_l1_mean_w': self.val_loss_l1_mean_w.result(), 'loss_l1_mean': self.val_loss_l1_mean.result()})

			if LOSS_TYP[7]:
				self.val_loss_l1_std_w.update_state(loss_gen_pw_dict['loss_l1_std_w'])
				self.val_loss_l1_std.update_state(loss_gen_pw_dict['loss_l1_std'])
				metrics_out.update({'loss_l1_std_w': self.val_loss_l1_std_w.result(), 'loss_l1_std': self.val_loss_l1_std.result()})

			return metrics_out

		@property
		def metrics(self):
			metrics_out = [self.met_l1, self.val_met_l1,
							self.loss_pw_w, self.val_loss_pw_w, 
							self.loss_gen_w, self.val_loss_gen_w,
							self.loss_gen_reg, self.val_loss_gen_reg,
							self.loss_disc_adv_reg, self.val_loss_disc_adv_reg]

			if LOSS_TYP[0]:
				metrics_out.extend([self.loss_gen_adv_w, self.loss_gen_adv, self.val_loss_gen_adv_w, self.val_loss_gen_adv])

			if LOSS_TYP[1]:
				metrics_out.extend([self.loss_disc_adv_w, self.loss_disc_adv, self.val_loss_disc_adv_w, self.val_loss_disc_adv])

			if LOSS_TYP[2]:
				metrics_out.extend([self.loss_l1_w, self.loss_l1, self.val_loss_l1_w, self.val_loss_l1])

			if LOSS_TYP[3]:
				metrics_out.extend([self.loss_l2_w, self.loss_l2, self.val_loss_l2_w, self.val_loss_l2])

			if LOSS_TYP[4]:
				metrics_out.extend([self.loss_l1_mlwt_w, self.loss_l1_mlwt, self.val_loss_l1_mlwt_w, self.val_loss_l1_mlwt])

			if LOSS_TYP[5]:
				metrics_out.extend([self.loss_l1_fft_w, self.loss_l1_fft, self.val_loss_l1_fft_w, self.val_loss_l1_fft])

			if LOSS_TYP[6]:
				metrics_out.extend([self.loss_l1_mean_w, self.loss_l1_mean, self.val_loss_l1_mean_w, self.val_loss_l1_mean])

			if LOSS_TYP[7]:
				metrics_out.extend([self.loss_l1_std_w, self.loss_l1_std, self.val_loss_l1_std_w, self.val_loss_l1_std])

			return metrics_out
else:
	class My_model(tf.keras.Model):
		def __init__(self, input_shape, *args, **kwargs):
			super(My_model, self).__init__(*args, **kwargs)
	
			self.gen = nn_model_gen(input_shape, 'input_gen', 'gen_', 'nEM_model_rest')


			self.met_l1 = tf.keras.metrics.Mean(name="met_l1")
			self.val_met_l1= tf.keras.metrics.Mean(name="met_l1")

			self.loss_pw_w = tf.keras.metrics.Mean(name="loss_pw_w")
			self.val_loss_pw_w = tf.keras.metrics.Mean(name="loss_pw_w")

			self.loss_gen_reg = tf.keras.metrics.Mean(name="loss_gen_reg")
			self.val_loss_gen_reg = tf.keras.metrics.Mean(name="loss_gen_reg")

			if LOSS_TYP[2]:
				self.loss_l1_w = tf.keras.metrics.Mean(name="loss_l1_w")
				self.loss_l1 = tf.keras.metrics.Mean(name="loss_l1")
				self.val_loss_l1_w = tf.keras.metrics.Mean(name="loss_l1_w")
				self.val_loss_l1 = tf.keras.metrics.Mean(name="loss_l1")

			if LOSS_TYP[3]:
				self.loss_l2_w = tf.keras.metrics.Mean(name="loss_l2_w")
				self.loss_l2 = tf.keras.metrics.Mean(name="loss_l2")
				self.val_loss_l2_w = tf.keras.metrics.Mean(name="loss_l2_w")
				self.val_loss_l2 = tf.keras.metrics.Mean(name="loss_l2")

			if LOSS_TYP[4]:
				self.loss_l1_mlwt_w = tf.keras.metrics.Mean(name="loss_l1_mlwt_w")
				self.loss_l1_mlwt = tf.keras.metrics.Mean(name="loss_l1_mlwt")
				self.val_loss_l1_mlwt_w = tf.keras.metrics.Mean(name="loss_l1_mlwt_w")
				self.val_loss_l1_mlwt = tf.keras.metrics.Mean(name="loss_l1_mlwt")

			if LOSS_TYP[5]:
				self.loss_l1_fft_w = tf.keras.metrics.Mean(name="loss_l1_fft_w")
				self.loss_l1_fft = tf.keras.metrics.Mean(name="loss_l1_fft")
				self.val_loss_l1_fft_w = tf.keras.metrics.Mean(name="loss_l1_fft_w")
				self.val_loss_l1_fft = tf.keras.metrics.Mean(name="loss_l1_fft")

			if LOSS_TYP[6]:
				self.loss_l1_mean_w = tf.keras.metrics.Mean(name="loss_l1_mean_w")
				self.loss_l1_mean = tf.keras.metrics.Mean(name="loss_l1_mean")
				self.val_loss_l1_mean_w = tf.keras.metrics.Mean(name="loss_l1_mean_w")
				self.val_loss_l1_mean = tf.keras.metrics.Mean(name="loss_l1_mean")

			if LOSS_TYP[7]:
				self.loss_l1_std_w = tf.keras.metrics.Mean(name="loss_l1_std_w")
				self.loss_l1_std = tf.keras.metrics.Mean(name="loss_l1_std")
				self.val_loss_l1_std_w = tf.keras.metrics.Mean(name="loss_l1_std_w")
				self.val_loss_l1_std = tf.keras.metrics.Mean(name="loss_l1_std")

			self.ibg = tf.Variable(0.0, dtype=tf.float32, trainable=False)
			self.ibg_m = 2*nvgb.TRAIN_N_DATA//nvgb.BATCH_SIZE

		def compile(self, parm):
			super(My_model, self).compile()

			self.gen_opt = nfop.fcn_get_optimizer_from_vgb()

			self.lr_schedule_gen = nfcb.Cb_Lr_schedule_base(
					lr_min=parm['lr_min'],
					lr_max=parm['opt_lr'],
					m_min=parm['m_0'],
					m_max=parm['opt_m'],
					decay_steps=parm['decay_steps'],
					decay_rate=parm['decay_rate'],
					steps_per_cycle=parm['decay_steps'],
					warmup_steps=parm['warmup_steps'],
					cooldown_steps=parm['cooldown_steps'],
					lr_0=parm['lr_0'], 
					warmup_steps_0=parm['warmup_steps_0'], 
					cooldown_steps_0=parm['cooldown_steps_0'],
					decay_steps_0=parm['decay_steps_0'],
					lrs_m_pow=parm['lrs_m_pow'],
					lrs_lr_pow=parm['lrs_lr_pow'])

		def reset_opt_iter(self):
			tf.keras.backend.set_value(self.gen_opt.iterations, 0)

		def set_opt_lr_m(self):
			self.lr_schedule_gen.get_set_opt_lr_m(self.gen_opt)

		def inc_opt_counter(self):
			self.lr_schedule_gen.inc_counter()

		def call(self, inputs, training=None, mask=None):
			return self.gen(inputs, training)

		@tf.function
		def train_step(self, data):
			x, y = data

			x = fcn_x(x, y, self.ibg, self.ibg_m)

			with tf.GradientTape() as tape_gen:
				# Forward pass
				y_p = self.gen(x, training=True)

				# pixelwise loss
				loss_gen_pw_dict = fls_pw_w(y, y_p)

				# regularization loss
				loss_gen_reg = tf.reduce_sum(self.gen.losses)

				loss_gen_t = loss_gen_reg + loss_gen_pw_dict['loss_pw_w']

			# Compute gradient
			grad_gen_t = tape_gen.gradient(loss_gen_t, self.gen.trainable_variables)

			#clip gradients
			grad_gen_t = nfop.fcn_optimizer_clip_gradients(self.gen_opt, grad_gen_t)

			# Update gradient
			self.gen_opt.apply_gradients(zip(grad_gen_t, self.gen.trainable_variables))

			# save metrics
			self.met_l1.update_state(loss_gen_pw_dict['met_l1'])
			metrics_out = {'met_l1': self.met_l1.result()}

			# save losses
			self.loss_pw_w.update_state(loss_gen_pw_dict['loss_pw_w'])
			metrics_out.update({'loss_pw_w': self.loss_pw_w.result()})

			self.loss_gen_reg.update_state(loss_gen_reg)
			metrics_out.update({'loss_gen_reg': self.loss_gen_reg.result()})

			if LOSS_TYP[2]:
				self.loss_l1_w.update_state(loss_gen_pw_dict['loss_l1_w'])
				self.loss_l1.update_state(loss_gen_pw_dict['loss_l1'])
				metrics_out.update({'loss_l1_w': self.loss_l1_w.result(), 'loss_l1': self.loss_l1.result()})

			if LOSS_TYP[3]:
				self.loss_l2_w.update_state(loss_gen_pw_dict['loss_l2_w'])
				self.loss_l2.update_state(loss_gen_pw_dict['loss_l2'])
				metrics_out.update({'loss_l2_w': self.loss_l2_w.result(), 'loss_l2': self.loss_l2.result()})

			if LOSS_TYP[4]:
				self.loss_l1_mlwt_w.update_state(loss_gen_pw_dict['loss_l1_mlwt_w'])
				self.loss_l1_mlwt.update_state(loss_gen_pw_dict['loss_l1_mlwt'])
				metrics_out.update({'loss_l1_mlwt_w': self.loss_l1_mlwt_w.result(), 'loss_l1_mlwt': self.loss_l1_mlwt.result()})

			if LOSS_TYP[5]:
				self.loss_l1_fft_w.update_state(loss_gen_pw_dict['loss_l1_fft_w'])
				self.loss_l1_fft.update_state(loss_gen_pw_dict['loss_l1_fft'])
				metrics_out.update({'loss_l1_fft_w': self.loss_l1_fft_w.result(), 'loss_l1_fft': self.loss_l1_fft.result()})

			if LOSS_TYP[6]:
				self.loss_l1_mean_w.update_state(loss_gen_pw_dict['loss_l1_mean_w'])
				self.loss_l1_mean.update_state(loss_gen_pw_dict['loss_l1_mean'])
				metrics_out.update({'loss_l1_mean_w': self.loss_l1_mean_w.result(), 'loss_l1_mean': self.loss_l1_mean.result()})

			if LOSS_TYP[7]:
				self.loss_l1_std_w.update_state(loss_gen_pw_dict['loss_l1_std_w'])
				self.loss_l1_std.update_state(loss_gen_pw_dict['loss_l1_std'])
				metrics_out.update({'loss_l1_std_w': self.loss_l1_std_w.result(), 'loss_l1_std': self.loss_l1_std.result()})

			self.ibg.assign_add(1.0)

			return metrics_out

		@tf.function
		def test_step(self, data):
			x, y = data

			# Forward pass
			y_p = self.gen(x, training=False)

			# pixelwise loss
			loss_gen_pw_dict = fls_pw_w(y, y_p)

			# regularization loss
			loss_gen_reg = tf.reduce_sum(self.gen.losses)

			# save metrics
			self.val_met_l1.update_state(loss_gen_pw_dict['met_l1'])
			metrics_out = {'met_l1': self.val_met_l1.result()}

			# save losses
			self.val_loss_pw_w.update_state(loss_gen_pw_dict['loss_pw_w'])
			metrics_out.update({'loss_pw_w': self.val_loss_pw_w.result()})

			self.val_loss_gen_reg.update_state(loss_gen_reg)
			metrics_out.update({'loss_gen_reg': self.val_loss_gen_reg.result()})

			if LOSS_TYP[2]:
				self.val_loss_l1_w.update_state(loss_gen_pw_dict['loss_l1_w'])
				self.val_loss_l1.update_state(loss_gen_pw_dict['loss_l1'])
				metrics_out.update({'loss_l1_w': self.val_loss_l1_w.result(), 'loss_l1': self.val_loss_l1.result()})

			if LOSS_TYP[3]:
				self.val_loss_l2_w.update_state(loss_gen_pw_dict['loss_l2_w'])
				self.val_loss_l2.update_state(loss_gen_pw_dict['loss_l2'])
				metrics_out.update({'loss_l2_w': self.val_loss_l2_w.result(), 'loss_l2': self.val_loss_l2.result()})
	
			if LOSS_TYP[4]:
				self.val_loss_l1_mlwt_w.update_state(loss_gen_pw_dict['loss_l1_mlwt_w'])
				self.val_loss_l1_mlwt.update_state(loss_gen_pw_dict['loss_l1_mlwt'])
				metrics_out.update({'loss_l1_mlwt_w': self.val_loss_l1_mlwt_w.result(), 'loss_l1_mlwt': self.val_loss_l1_mlwt.result()})

			if LOSS_TYP[5]:
				self.val_loss_l1_fft_w.update_state(loss_gen_pw_dict['loss_l1_fft_w'])
				self.val_loss_l1_fft.update_state(loss_gen_pw_dict['loss_l1_fft'])
				metrics_out.update({'loss_l1_fft_w': self.val_loss_l1_fft_w.result(), 'loss_l1_fft': self.val_loss_l1_fft.result()})

			if LOSS_TYP[6]:
				self.val_loss_l1_mean_w.update_state(loss_gen_pw_dict['loss_l1_mean_w'])
				self.val_loss_l1_mean.update_state(loss_gen_pw_dict['loss_l1_mean'])
				metrics_out.update({'loss_l1_mean_w': self.val_loss_l1_mean_w.result(), 'loss_l1_mean': self.val_loss_l1_mean.result()})

			if LOSS_TYP[7]:
				self.val_loss_l1_std_w.update_state(loss_gen_pw_dict['loss_l1_std_w'])
				self.val_loss_l1_std.update_state(loss_gen_pw_dict['loss_l1_std'])
				metrics_out.update({'loss_l1_std_w': self.val_loss_l1_std_w.result(), 'loss_l1_std': self.val_loss_l1_std.result()})

			return metrics_out

		@property
		def metrics(self):
			metrics_out = [self.met_l1, self.val_met_l1,
							self.loss_pw_w, self.val_loss_pw_w, 
							self.loss_gen_reg, self.val_loss_gen_reg]

			if LOSS_TYP[2]:
				metrics_out.extend([self.loss_l1_w, self.loss_l1, self.val_loss_l1_w, self.val_loss_l1])

			if LOSS_TYP[3]:
				metrics_out.extend([self.loss_l2_w, self.loss_l2, self.val_loss_l2_w, self.val_loss_l2])

			if LOSS_TYP[4]:
				metrics_out.extend([self.loss_l1_mlwt_w, self.loss_l1_mlwt, self.val_loss_l1_mlwt_w, self.val_loss_l1_mlwt])

			if LOSS_TYP[5]:
				metrics_out.extend([self.loss_l1_fft_w, self.loss_l1_fft, self.val_loss_l1_fft_w, self.val_loss_l1_fft])

			if LOSS_TYP[6]:
				metrics_out.extend([self.loss_l1_mean_w, self.loss_l1_mean, self.val_loss_l1_mean_w, self.val_loss_l1_mean])

			if LOSS_TYP[7]:
				metrics_out.extend([self.loss_l1_std_w, self.loss_l1_std, self.val_loss_l1_std_w, self.val_loss_l1_std])

			return metrics_out
#########################################################################################
####################################### write image #####################################
#########################################################################################
class Cb_Test_Plot(tf.keras.callbacks.Callback):
	def __init__(self, parm): # add other arguments to __init__ if you need
		super(Cb_Test_Plot, self).__init__()

		self.wi_log_dir = os.path.join(parm['log_dir'], 'test')
		self.n_spl = parm['test_n_samples']
		self.parm = parm

		# spacing for plotting
		self.wspace = 0.025
		self.vspace = 0.010
		self.wsize_fig = 25.0

		self.file_test_writer = tf.summary.create_file_writer(logdir=self.wi_log_dir, max_queue=1)

		x, y = self.fcn_load_test_mat(self.parm['test_mat_path'])

		self.wi_ds_x = tf.data.Dataset.from_tensor_slices(x).batch(self.n_spl)

		self.wi_x = self.fcn_norm_x_image(x)

		self.wi_y_sft, self.wi_y_sc = nfge.fcn_get_norm_parm_image(y, [1, 2])
		self.wi_y = y

		self.wi_ibg = np.array(0).astype(np.int64)
		self.wi_ickp = np.array(0).astype(np.int64)

	def fcn_load_test_mat(self, path_mat):
		x_mat, y_mat = nfge.fcn_ds_read_x_y_mat(path_mat, 'x', 'y', x_typ_mat, y_typ_mat, x_ndim, y_ndim)

		x_mat = x_mat[..., 0:self.n_spl].copy()
		x_mat = np.moveaxis(x_mat, 3, 0)
		x = tf.convert_to_tensor(x_mat, x_typ_tf[0])

		y_mat = y_mat[..., 0:self.n_spl].copy()
		y_mat = np.moveaxis(y_mat, 3, 0)
		y = tf.convert_to_tensor(y_mat, y_typ_tf[0])

		x, y = fcn_resize_xy(x, y, False)

		x, y = fcn_scale_inputs_x_y(x, y, [1, 2])

		return x, y

	def fcn_norm_x_image(self, x, axis=[1, 2]):
		xn = nfge.fcn_norm_image(x, axis)
		return xn.numpy()

	def fcn_norm_y_image(self, y, axis=[1, 2]):
		yn = nfge.fcn_norm_image(y, axis, self.wi_y_sft, self.wi_y_sc)

		return yn.numpy()

	def fcn_xy_image_gen(self, y_t, y_p):
		x = self.wi_x
		
		if type(x) is not np.ndarray:
			x = x.numpy()

		if type(y_t) is not np.ndarray:
			y_t = y_t.numpy()

		if type(y_p) is not np.ndarray:
			y_p = y_p.numpy()

		yn = self.fcn_norm_y_image(y_t)
		yn_p = self.fcn_norm_y_image(y_p)

		rows = 4

		hsize_fig = (self.wsize_fig-(self.n_spl-1)*self.wspace)*rows/self.n_spl

		figure = plt.figure(figsize=(self.wsize_fig, hsize_fig))
		for ik in range(self.n_spl):
			x_ik = x[ik, ...].squeeze()

			dy = y_t[ik, ...].squeeze() - y_p[ik, ...].squeeze()
			ee = np.mean(np.fabs(dy))
			dyn = dy - np.min(dy)
			dyn = dyn/np.max(dyn)

			y_ik = yn[ik, ...].squeeze()
			y_p_ik = yn_p[ik, ...].squeeze()

			for iy in range(rows):
				im_x_ik = x_ik if iy==0 else y_p_ik if iy==1 else y_ik if iy==2 else dyn

				ax = plt.subplot(4, self.n_spl, iy*self.n_spl + ik + 1)
				ax.imshow(im_x_ik)
				ax.set_xticks([])
				ax.set_yticks([])
				ax.grid(False)

				if iy==0:
					title = 'e = {:4.3f}'.format(ee)
					ax.set_title(title, fontsize=14)

		figure.subplots_adjust(hspace=self.vspace, wspace=self.wspace)
		figure.tight_layout()

		return nfge.fcn_plot_to_image(figure)

	def on_train_begin(self, logs=None):
		self.model.reset_opt_iter()

	def on_train_batch_begin(self, batch, logs=None):
		self.model.set_opt_lr_m()

		# saving learning rate and momentum
		if self.wi_ibg % self.parm['log_update_freq'] == 0:	
			with self.file_test_writer.as_default():
				lr, m = nfop.fcn_read_opt_lr(self.model.gen_opt), nfop.fcn_read_opt_m(self.model.gen_opt)				
				tf.summary.scalar(name='batch_gen_lr', data=lr, step=self.wi_ibg)
				tf.summary.scalar(name='batch_gen_m', data=m, step=self.wi_ibg)

				if GAN_TYP == 1:
					lr, m = nfop.fcn_read_opt_lr(self.model.disc_opt), nfop.fcn_read_opt_m(self.model.disc_opt)				
					tf.summary.scalar(name='batch_disc_lr', data=lr, step=self.wi_ibg)
					tf.summary.scalar(name='batch_disc_m', data=m, step=self.wi_ibg)

		# saving test data
		if self.wi_ibg % self.parm['test_update_freq'] == 0:
			y_p = self.model.predict(self.wi_ds_x)
			with self.file_test_writer.as_default():
				tf.summary.image(name='data', data=self.fcn_xy_image_gen(self.wi_y, y_p), step=self.wi_ibg)
		
		# saving weights
		if self.wi_ibg % self.parm['log_checkpoint_freq'] == 0:
			self.model.gen.save_weights(self.parm['ckp_gen_path'].format(self.wi_ickp))
			if GAN_TYP == 1:
				self.model.disc.save_weights(self.parm['ckp_disc_path'].format(self.wi_ickp))

			self.wi_ickp += 1

		# reset metrics
		if (self.wi_ibg > 0) and (self.wi_ibg % self.parm['reset_metric_freq'] == 0):
			self.model.reset_metrics()

		self.wi_ibg += 1
		self.model.inc_opt_counter()

#########################################################################################
#########################################################################################
#########################################################################################
def fcn_load_weights(model, path_dir, bb_load_disc=True, by_name=True):
	if os.path.exists(path_dir):
		dir_weights = nfge.fcn_read_files(path_dir, ".h5")
		path_gen = [s for s in dir_weights if "gen" in s]
		path_gen.sort()
		if len(path_gen):
			model.gen = nfge.fcn_load_weights(model.gen, path_gen[-1], by_name=by_name, skip_mismatch=False)

		if bb_load_disc:
			path_disc = [s for s in dir_weights if "disc" in s]
			path_disc.sort()
			if len(path_disc) and (GAN_TYP == 1):
				model.disc = nfge.fcn_load_weights(model.disc, path_disc[-1], by_name=by_name, skip_mismatch=False)

	return model

def fcn_compile_and_fit(parm, bb_print_model_summary=False):
 
	def fcn_compile_strategy(parm, bb_print_model_summary):
		# set optimizer

		model = My_model(parm['input_shape'])

		if parm['bb_parm_search']:
			# save weights
			if not os.path.exists(parm['grid_search_dir']):
				os.makedirs(parm['grid_search_dir'])

			# load initial weights
			model = nfge.fcn_load_weights(model, parm['weigths_load_path'], bb_load_disc=True, by_name=True)

			# save weights
			model.save_weights(parm['weigths_search_path'])
		else:
			model = fcn_load_weights(model, parm['weigths_load_path'], bb_load_disc=True, by_name=True)

		# reset ibg value
		model.ibg.assign(0.0)

		# print summary
		if bb_print_model_summary:
			print(model.gen.summary())
			if GAN_TYP == 1:
				print(model.disc.summary())
			
		# print training information
		nvgb.fcn_print_training_parm(parm)

		# compile
		model.compile(parm)
  
		return model

	# clear session
	tf.keras.backend.clear_session()

	# generate model
	if nvgb.N_GPUS == 1:
		model = fcn_compile_strategy(parm, bb_print_model_summary)
	else:
		mirrored_strategy = tf.distribute.MirroredStrategy(cross_device_ops = tf.distribute.ReductionToOneDevice())
		with mirrored_strategy.scope():
			model = fcn_compile_strategy(parm, bb_print_model_summary)

	# set datasets
	train_dataset, val_dataset = nfge.fcn_load_datasets(fcn_ds_map, parm)

	if parm['bb_parm_search']:
		val_dataset = None
		parm['validation_steps'] = None
		parm['validation_freq'] = 1

	#fit
	model.fit(train_dataset,
		epochs=parm['epochs'],
		steps_per_epoch=parm['train_steps'],
		callbacks=parm['callbacks'],
		validation_data=val_dataset,
		validation_steps=parm['validation_steps'],
		validation_freq=parm['validation_freq'],
		initial_epoch=parm['initial_epoch'],
		verbose=parm['verbose_fit'])

#########################################################################################
#################################### initialization #####################################
#########################################################################################
def fcn_init(bb_parm_search=False):
	# parse arguments
	parser = nfge.fcn_parse_gen_input_parm()

	root = '/media/ssd_1'

	dat_id = vars(parser.parse_args())['modv'] // 100

	# 0: gan - gen
	# 1: gan - disc
	# 2: L1
	# 3: L2
	# 4: L1 - MLWT
	# 5: L1 - FFT
	# 6: mean
	# 7: std

	if dat_id == 1:
		dir_db = 'dataset_hrsem'
	elif dat_id == 2:
		dir_db = 'dataset_lrsem'
	elif dat_id == 3:
		dir_db = 'dataset_hrstem'	
	elif dat_id == 4:
		dir_db = 'dataset_lrstem'	
	elif dat_id == 5:
		dir_db = 'dataset_hrtem'
	elif dat_id == 6:
		dir_db = 'dataset_lrtem'

	parm = nfge.fcn_init(parser,
						db_dir=dir_db,
						opt_nesterov=False,
						opt_beta_2=0.999,
						opt_eps=1.0e-5,
						opt_ctyp=0, # 0: None, 1: clip by value, 2: clip by norm
						opt_cval=2.0, # 0: 4.0, 1: 8.0
						norm_trainable=True,
						norm_eps=1e-3,
						norm_pos=1, # 1: before layer, 2: after layer
						norm_reg_typ=0,
						norm_reg_gamma=0,
						norm_reg_beta=1e-8,
						norm_cstr_typ=0,
						norm_cstr_v0=0.01,
						norm_cstr_ve=8.0,
						norm_cstr_rt=0.01,
						norm_cstr_ax=[0],
						norm_renorm=False,
						norm_renorm_m=0.99,
						dp_spt=False, # use spatial dropout
						n_classes=1,
						res_sfr=1.0,
						dsn_typ=3, # 1: dense, 2: dense bottleneck, 3: residual dense, 4: residual dense bottleneck, 5: down/up residual dense, 6: down/up residual dense bottleneck
						dsn_compr=0.5,
						dsn_in_fr=2,
						dsn_btn_fr=4,
						dsn_fsl_act=None,
						bb_parm_search=bb_parm_search,
						root=root,
	  					fn_ext=None,
						gpu_memory_limit=None)

	parm['input_shape'] = (nvgb.X_SHAPE[0]//DS_FTR, nvgb.X_SHAPE[1]//DS_FTR, nvgb.X_SHAPE[2])
	parm['output_shape'] = (nvgb.Y_SHAPE[0]//DS_FTR, nvgb.Y_SHAPE[1]//DS_FTR, nvgb.Y_SHAPE[2])

	parm['x_shape'] = (nvgb.X_SHAPE[0], nvgb.X_SHAPE[1], nvgb.Y_SHAPE[2])
	parm['y_shape'] = (nvgb.Y_SHAPE[0], nvgb.Y_SHAPE[1], nvgb.Y_SHAPE[2])

	parm['x_typ_mat'] = x_typ_mat
	parm['y_typ_mat'] = y_typ_mat

	parm['x_typ_rtf'] = x_typ_rtf
	parm['y_typ_rtf'] = y_typ_rtf

	parm['x_typ_tf'] = x_typ_tf
	parm['y_typ_tf'] = y_typ_tf

	parm['ckp_default'] = False
	log_dir_root = os.path.split(parm['checkpoint_path'])[0]
	parm['ckp_gen_path'] = os.path.join(log_dir_root, 'gen-{:04d}.h5')
	parm['ckp_disc_path'] = os.path.join(log_dir_root, 'disc-{:04d}.h5')

	return	parm