import os
import torchvision.models as models
from Code.model import resnet_feature_extractor

# Setup work
device_ids = [1]
data_path = '../meta/kins/data/'
model_save_dir = '../meta/kins/models/'

dataset = 'KINS'
nn_type = 'resnext'
vMF_kappa=30
vc_num = 512

categories = ['pig']
cat_test = ['pig'] #may include multi-category

layer = 'second' # 'last','second'
extractor=resnet_feature_extractor(nn_type,layer)

extractor.cuda(device_ids[0]).eval()

init_path = model_save_dir+'init_{}/'.format(nn_type)
if not os.path.exists(init_path):
	os.makedirs(init_path)

dict_dir = init_path+'dictionary_{}/'.format(nn_type)
if not os.path.exists(dict_dir):
	os.makedirs(dict_dir)

sim_dir = init_path+'similarity_{}_{}_{}/'.format(nn_type,layer,dataset)

Astride_set = [2, 4, 8, 16, 32]  # stride size
featDim_set = [64, 128, 256, 512, 512]  # feature dimension
Arf_set = [6, 16, 44, 100, 212]  # receptive field size
Apad_set = [2, 6, 18, 42, 90]  # padding size

if layer =='pool4' or layer =='second':
	Astride = Astride_set[3]
	Arf = Arf_set[3]
	Apad = Apad_set[3]
	offset = 3
elif layer =='pool5' or layer == 'last':
	Astride = Astride_set[3]
	Arf = 170
	Apad = Apad_set[4]
	offset = 3
