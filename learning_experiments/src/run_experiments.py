#!/usr/bin/env python
"""
title           :run_experiments.py
description     :Runs and saves results from multiple experiments in sequence.
author          :Yordan Hristov <yordan.hristov@ed.ac.uk
date            :01/2019
python_version  :2.7.14
==============================================================================
"""

import copy
from itertools import chain
import subprocess

epochs = '20'
augment_counter = '0'
gamma = '5000'
beta = '3'
alpha = '1'

experiments = {}
experiments[1] = {}
experiments[1]['--alpha'] = alpha
experiments[1]['--beta'] = beta
experiments[1]['--gamma'] = gamma
experiments[1]['--model'] = 'full'
experiments[1]['--augment_counter'] = augment_counter
experiments[1]['-e'] = epochs

# experiments = {}
experiments[2] = {}
experiments[2]['--alpha'] = '0'
experiments[2]['--beta'] = beta
experiments[2]['--gamma'] = gamma
experiments[2]['--model'] = 'var_classifier'
experiments[2]['--augment_counter'] = augment_counter
experiments[2]['-e'] = epochs

# experiments = {}
experiments[3] = {}
experiments[3]['--alpha'] = alpha
experiments[3]['--beta'] = '1'
experiments[3]['--gamma'] = '0'
experiments[3]['--model'] = 'beta_vae'
experiments[3]['--augment_counter'] = augment_counter
experiments[3]['-e'] = epochs

# experiments = {}
experiments[4] = {}
experiments[4]['--alpha'] = alpha
experiments[4]['--beta'] = '0'
experiments[4]['--gamma'] = '0'
experiments[4]['--model'] = 'autoencoder'
experiments[4]['--augment_counter'] = augment_counter
experiments[4]['-e'] = epochs

# experiments = {}
experiments[5] = {}
experiments[5]['--alpha'] = '0'
experiments[5]['--beta'] = '0'
experiments[5]['--gamma'] = gamma
experiments[5]['--model'] = 'classifier'
experiments[5]['--augment_counter'] = augment_counter
experiments[5]['-e'] = epochs

if __name__ == "__main__":
	for exp_no in range(1, len(experiments) + 1):

		print(exp_no)
		config = copy.deepcopy(experiments[exp_no])
		params = ["python", "src/train_vae.py"] + list(chain(*zip(config.keys(), config.values())))
		subprocess.call(params)
