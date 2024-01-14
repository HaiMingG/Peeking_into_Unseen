import os, cv2, pickle
import numpy as np
import matplotlib.pyplot as plt


device_ids = [0]

MODEL_TYPE = 'E2E'            

EXPERIMENT_DATASET = 'kinsv'

dataset_train = EXPERIMENT_DATASET
dataset_eval = EXPERIMENT_DATASET

nn_type = 'resnext'

vc_num = 512
K = 24 #number of templates
context_cluster = 5

### ==== Directories ==== ###

home_dir = '../'
meta_dir = home_dir + 'Models/'
data_dir = home_dir + 'Dataset/'
init_dir = meta_dir + 'ML_{}/'.format(nn_type)
exp_dir = home_dir + 'log/'
trn_dir = home_dir + 'training/'

for d in [exp_dir, trn_dir]:
    if not os.path.exists(d):
        os.mkdir(d)

### ==== Categories ==== ###

categories = dict()
categories['kinsv'] = ['piglet', 'sow']

categories['train'] = categories[dataset_train]
categories['eval']  = categories[dataset_eval]


### ==== Network ==== ###

vMF_kappas = {'vgg_pool4_pascal3d+' : 30, 'resnext_second_pascal3d+' : 65, 'vgg_pool4_kinsv' : 30, 'resnext_second_kinsv' : 50 }

layer = 'second'
feature_num = 1024
feat_stride = 16

vMF_kappa = vMF_kappas['{}_{}_{}'.format(nn_type, layer, dataset_train)]

rpn_configs = {'training_param' : {'weight_decay': 0.0005, 'lr_decay': 0.1, 'lr': 1e-3}, 'ratios' : [0.5, 1, 2], 'anchor_scales' : [8, 16, 32], 'feat_stride' : feat_stride }
