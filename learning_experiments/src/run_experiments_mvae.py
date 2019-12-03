#!/usr/bin/env python
"""
title           :run_experiments_mvae.py
description     :Runs and saves results from multiple experiments in sequence.
author          :Yordan Hristov <yordan.hristov@ed.ac.uk
date            :01/2019
python_version  :2.7.14
==============================================================================
"""

import copy
from itertools import chain
import subprocess
import os
import os.path as osp
import json
import shutil

epochs_arr = ['25']
gammas_obj = ['50000']
gammas_rel = ['50000']
betas = ['10']
alphas = ['1']
dimz_arr = ['8']
batchsizes = ['32']
objects_n_arr = ['2']
n_experiments = 1

BASE_DIR = "photoreal_results"
for folder in os.listdir(BASE_DIR):
	shutil.rmtree(osp.join(BASE_DIR, folder))

experiments = {}
for exp_idx in range(n_experiments):
	experiments[exp_idx] = {}
	alpha = alphas[0] if len(alphas) == 1 else alphas[exp_idx]
	experiments[exp_idx]['--alpha'] = alpha

	beta = betas[0] if len(betas) == 1 else betas[exp_idx]
	experiments[exp_idx]['--beta'] = beta

	gamma_obj = gammas_obj[0] if len(gammas_obj) == 1 else gammas_obj[exp_idx]
	experiments[exp_idx]['--gamma_obj'] = gamma_obj

	gamma_rel = gammas_rel[0] if len(gammas_rel) == 1 else gammas_rel[exp_idx]
	experiments[exp_idx]['--gamma_rel'] = gamma_rel

	epochs = epochs_arr[0] if len(epochs_arr) == 1 else epochs_arr[exp_idx]
	experiments[exp_idx]['--epochs'] = epochs

	batchsize = batchsizes[0] if len(batchsizes) == 1 else batchsizes[exp_idx]
	experiments[exp_idx]['--batchsize'] = batchsize

	dimz = dimz_arr[0] if len(dimz_arr) == 1 else dimz_arr[exp_idx]
	experiments[exp_idx]['--dimz'] = dimz

	objects_n = objects_n_arr[0] if len(objects_n_arr) == 1 else objects_n_arr[exp_idx]
	experiments[exp_idx]['--objects_n'] = objects_n

	experiments[exp_idx]['--output_dir'] = osp.join(BASE_DIR, str(exp_idx))

if __name__ == "__main__":
	for exp_no in range(n_experiments):

		print(exp_no)
		config = experiments[exp_no]

		if not osp.isdir(osp.join(config['--output_dir'])):
			os.makedirs(config['--output_dir'])

		with open(osp.join(config['--output_dir'], "args.json"), 'w') as f:
			json.dump(config, f, indent=2)

		params = ["python", "src/train_mvae.py"] + list(chain(*zip(config.keys(), config.values())))
		subprocess.call(params)
