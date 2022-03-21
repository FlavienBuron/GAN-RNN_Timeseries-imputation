import os

import yaml
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn

import tools
import holdout


# Load config params
params = yaml.load( open(os.getcwd() + '/config.yaml'), yaml.Loader ) 

# Add this for simplicity
params['size_nan'] = int(params['len_input']*params['total_nan_share']) 

# Once params are loaded, load TensorFlow and set its GPU config
tools.set_gpu_configurations(params) 

# Load models by name
seq2seq_name = 'seq2seq_00'
GAN_name = 'GAN_00'
pGAN_name = 'pGAN_00'

seq2seq = tf.keras.models.load_model(os.getcwd() + '/saved_models/' + seq2seq_name +'.h5') 
GAN = tf.keras.models.load_model(os.getcwd() + '/saved_models/' + GAN_name +'.h5') 
pGAN = tf.keras.models.load_model(os.getcwd() + '/saved_models/' + pGAN_name +'.h5') 

# Seq2seq

_, _, _, E_seq2seq, stats_seq2seq = holdout.run_test(
    model = seq2seq, 
    params = params, 
    check_test_performance = True, 
    return_stats = True) 

# GAN's Generator

_, _, _, E_gan, stats_gan = holdout.run_test(
    model = GAN, 
    params = params, 
    check_test_performance = True, 
    return_stats = True) 

# Partial GAN's Generator

_, _, _, E_pgan, stats_pgan = holdout.run_test(
    model = pGAN, 
    params = params, 
    check_test_performance = True, 
    return_stats = True) 