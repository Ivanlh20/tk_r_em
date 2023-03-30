#-*- coding: utf-8 -*-
"""
Created on Thu Sep 26 14:31:55 2019

@author: Ivan
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import numpy as np
import sys

#########################################################################################
import tensorflow as tf

#########################################################################################
import nn_var_glob as nvgb
import nn_fcns_gen as nfge
import nn_fcns_callbacks as nfcb

#########################################################################################
import training.nn_fcns_local as nflc

#########################################################################################
def fcn_run(parm):
	train_steps = nvgb.TRAIN_N_DATA//nvgb.BATCH_SIZE
	validation_steps = nvgb.VAL_N_DATA//nvgb.BATCH_SIZE

	#########################################################################################
	decay_every_epochs = 1.0
	epochs = round(decay_every_epochs*1000)
	decay_steps = round(decay_every_epochs*train_steps)
	decay_rate = 0.90

	#########################################################################################
	lr_min = 1e-4*parm['opt_lr']
	lr_0 = 1e-4*parm['opt_lr']
	m_0 = 0.0	
	lrs_min = 1e-4
	lrs_max = 1e-1

	shuffle_buffer_size = np.minimum(train_steps, 32*nvgb.BATCH_SIZE) # number of elements
	prefetch_buffer_size = nvgb.N_GPUS*np.maximum(1, shuffle_buffer_size//nvgb.BATCH_SIZE) # number of batches (after batch)
	num_parallel_calls = np.minimum(2, nvgb.N_GPUS*2)


	log_update_freq = 32
	log_checkpoint_freq = np.maximum(1024, train_steps//2)
	print_update_freq = np.minimum(1024, 1*train_steps//8)
	test_update_freq = np.maximum(log_update_freq, log_update_freq*(train_steps//(1*log_update_freq)))
	test_n_samples = 8
	validation_freq = 1
	histogram_freq = 1
	reset_metric_freq = log_update_freq

	########################################################s#################################
	parm['weigths_load_path'] = ''

	#########################################################################################
	warmup_steps_0 = 1*train_steps//8
	warmup_steps_0 = np.minimum(256*2, train_steps//2)
	cooldown_steps_0 = np.minimum(decay_steps-warmup_steps_0-8192, train_steps//64)
	cooldown_steps_0 = np.minimum(np.maximum(0, decay_steps-warmup_steps_0-32), 0*decay_steps//1)
	decay_every_epochs_0 = 3*decay_every_epochs
	decay_steps_0 = round(decay_every_epochs_0*train_steps)

	warmup_steps = 0
	cooldown_steps = np.minimum(decay_steps-warmup_steps-32, train_steps//64)

	lrs_m_pow = 1.0
	lrs_lr_pow = 1.0

	#########################################################################################
	parm.update({
		'epochs': epochs,
		'lr_min': lr_min,
		'm_0': m_0,
		'lr_0': lr_0,
		'lrs_min': lrs_min,
		'lrs_max': lrs_max,
		'decay_steps': decay_steps,
		'decay_rate': decay_rate,
		'train_steps': train_steps,
		'validation_steps': validation_steps,
		'warmup_steps_0': warmup_steps_0,
		'cooldown_steps_0': cooldown_steps_0,
		'decay_steps_0': decay_steps_0,
		'warmup_steps': warmup_steps,
		'cooldown_steps': cooldown_steps,
		'lrs_m_pow': lrs_m_pow,
		'lrs_lr_pow': lrs_m_pow,
		'shuffle_buffer_size': shuffle_buffer_size,
		'prefetch_buffer_size': prefetch_buffer_size,
		'num_parallel_calls': num_parallel_calls,
		'log_checkpoint_freq': log_checkpoint_freq,
		'log_update_freq': log_update_freq,
		'print_update_freq': print_update_freq,
		'test_update_freq': test_update_freq,
		'test_n_samples': test_n_samples,
		'validation_freq': validation_freq,
		'histogram_freq': histogram_freq,
		'reset_metric_freq': reset_metric_freq,
		'write_grads': False,
		'initial_epoch': 0,
		'verbose_fit': 2
	})

	#################################### callbacks ##########################################
	parm['callbacks'] = nfcb.fcn_callbacks_train(parm, nflc)

	#########################################################################################
	nflc.fcn_compile_and_fit(parm, bb_print_model_summary=True)

if __name__ == '__main__':
	parm = nflc.fcn_init(bb_parm_search=False)
	fcn_run(parm)