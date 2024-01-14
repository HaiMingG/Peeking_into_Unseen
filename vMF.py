from Code.vMFMM import *
from config_initialization import vc_num, dataset, categories, data_path, cat_test, device_ids, Astride, Apad, Arf, vMF_kappa, layer,init_path, nn_type, dict_dir, offset, extractor
from Code.helpers import getImg, imgLoader, Imgset, myresize
from DataLoader import KINS_Compnet_Train_Dataset
import torch
from torch.utils.data import DataLoader
import cv2
import glob
import pickle
import os


categories_to_train = ['pig']

img_per_cat = 3000
samp_size_per_img = 20
height_threshold = 75

for vMF_kappa in [50, 60]:
	loc_set = []
	feat_set = []
	imgs = []
	nfeats = 0

	for category in categories_to_train:
		cur_img_num = 0
		imgset = KINS_Compnet_Train_Dataset(height_thrd=height_threshold)
		data_loader = DataLoader(dataset=imgset, batch_size=1, shuffle=True)


		for ii,data in enumerate(data_loader):
			input, demo_img, img_path, true_pad = data
			imgs.append(img_path[0])

			img = demo_img[0].numpy()

			if input.shape[3] < 32:
				continue
			if np.mod(ii,50)==0:
				print('{} / {}'.format(ii,img_per_cat), end='\r')

			if cur_img_num >= img_per_cat:
				break

			with torch.no_grad():
				tmp = extractor(input.cuda(device_ids[0]))[0].detach().cpu().numpy()
			height, width = tmp.shape[1:3]

			tmp = tmp[:,offset:height - offset, offset:width - offset]
			gtmp = tmp.reshape(tmp.shape[0], -1)
			if gtmp.shape[1] >= samp_size_per_img:
				rand_idx = np.random.permutation(gtmp.shape[1])[:samp_size_per_img]
			else:
				rand_idx = np.random.permutation(gtmp.shape[1])[:samp_size_per_img - gtmp.shape[1]]
			tmp_feats = gtmp[:, rand_idx].T

			cnt = 0
			for rr in rand_idx:
				ihi, iwi = np.unravel_index(rr, (height - 2 * offset, width - 2 * offset))
				hi = (ihi+offset)*(input.shape[2]/height)-Apad
				wi = (iwi + offset)*(input.shape[3]/width)-Apad

				loc_set.append([categories_to_train.index(category), ii, hi,wi,hi+Arf,wi+Arf])
				feat_set.append(tmp_feats[cnt,:])
				cnt+=1

			cur_img_num += 1
		print()


	feat_set = np.asarray(feat_set)
	loc_set = np.asarray(loc_set).T

	print(feat_set.shape)
	model = vMFMM(vc_num, 'k++')
	model.fit(feat_set, vMF_kappa, max_it=150)

	S = np.zeros((vc_num, vc_num))
	for i in range(vc_num):
		for j in range(i, vc_num):
			S[i,j] = np.dot(model.mu[i], model.mu[j])
	print('kap {} sim {}'.format(vMF_kappa,np.mean(S+S.T-np.diag(np.ones(vc_num)*2))))

	with open(dict_dir + 'dictionary_{}_{}_kappa{}.pickle'.format(layer,vc_num, vMF_kappa), 'wb') as fh:
		pickle.dump(model.mu, fh)

