import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from glob import glob
from PIL import Image,ImageFile
import torch
ImageFile.LOAD_TRUNCATED_IMAGES = True

class MyDataLoader(Dataset):
	def __init__(self, data_dir, partition, in_ch=3):
		self.in_ch = in_ch
		self.partition = partition
		hr = data_dir+'/HR/'
		lr = data_dir + '/LR/'
		self.hr_file = os.path.join(hr, '*.png')
		self.lr_file = os.path.join(lr, '*.png')
		self.hr_list = sorted(glob(self.hr_file))
		self.lr_list = sorted(glob(self.lr_file))


	def __len__(self):
		return len(self.hr_list) * 50

	def __getitem__(self,idx):
		# input 480 x 270 -> 1920 x 1080
		# input 480 x 270 -> 960 x 540
		img_idx = idx // 50
		if img_idx == 0:
			patch_idx = idx
		else:
			patch_idx = idx % img_idx

		self.hr_patches = []
		self.lr_patches = []
		self.hr_img = self.hr_list[img_idx]
		self.lr_img = self.lr_list[img_idx]

		hr_patch_size = 96
		lr_patch_size = 48
		img = Image.open(self.hr_img)
		img = img.resize((960, 540),resample=Image.BICUBIC)
		img.load()
		data = np.asarray(img, dtype='int32')
		data = data/255.0
		data = data.transpose(2, 1, 0)
		for i in range(data.shape[1]//hr_patch_size):
			for j in range(data.shape[2]//hr_patch_size):
				patch = data[:,hr_patch_size*i:hr_patch_size*(i+1),hr_patch_size*j:hr_patch_size*(j+1)]
				self.hr_patches.append(patch)

		self.hr_patches = np.asarray(self.hr_patches, dtype=np.float32)

		img = Image.open(self.lr_img)
		img = img.resize((480, 270),resample=Image.BICUBIC)
		img.load()
		data = np.asarray(img, dtype='int32')
		data = data/255.0
		data = data.transpose(2, 1, 0)
		for i in range(data.shape[1]//lr_patch_size):
			for j in range(data.shape[2]//lr_patch_size):
				patch = data[:,lr_patch_size*i:lr_patch_size*(i+1),lr_patch_size*j:lr_patch_size*(j+1)]
				self.lr_patches.append(patch)
		self.lr_patches = np.asarray(self.lr_patches, dtype=np.float32)
	
		lr_data = self.lr_patches[patch_idx]
		hr_data = self.hr_patches[patch_idx]
		lr_data = torch.from_numpy(lr_data)
		hr_data = torch.from_numpy(hr_data)
		return lr_data, hr_data



###test

# testset = MyDataLoader('testdata','test',  in_ch = 3)
# lr,hr = testset.__getitem__(20)
# print(lr.shape)
# print(hr.shape)
		
