import argparse
import numpy as np
import sys
import os
# sys.path.append('models/')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.parameter import Parameter 
import time

from dataloader import MyDataLoader
from ESPCN import MyModel

name = 'ESPCN'

def getPSNRLoss():
  mseloss_fn = nn.MSELoss(reduction='none')

  def PSNRLoss(output, target):
    loss = mseloss_fn(output, target)
    loss = torch.mean(loss, dim=(1,2))
    loss = 10 * torch.log10(loss)
    mean = torch.mean(loss)
    return mean

  return PSNRLoss

psnr_func = getPSNRLoss()

def test(model, test_loader, loss_func, device):
	model.eval()
	test_loss = 0.
	test_psnr = 0.
	count = 0.
	time_list = []
	with torch.no_grad():
		for batch_idx, (data, target) in enumerate(test_loader):
			data, target = data.to(device), target.to(device)
			# torch.cuda.synchronize()
			start = time.time()
			output = model(data)
			#print(output)
			# torch.cuda.synchronize()
			end = time.time()
			tmp_time = end - start
			time_list.append(tmp_time)
			# print(tmp_time)
			loss = loss_func(output, target)
			psnr = psnr_func(output, target)
			test_loss += loss.item()*data.size()[0]
			test_psnr += psnr.item()*data.size()[0]
			count += data.size()[0]

	test_loss /= count
	test_psnr /= count
	time_list = np.asarray(time_list)
	avg_time = np.mean(time_list) 

	print('Test Loss: {:.6f}'.format(test_loss),flush=True)
	print('Test PSNR: {:.6f}'.format(test_psnr),flush=True)
	print('Time: {:.4f}'.format(avg_time),flush=True)
	return test_loss

def main():
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")
	# print(device)
	testset = MyDataLoader('data/', 'test', in_ch = 3)
	test_loader = DataLoader(testset, 1, shuffle = True)

	model = MyModel().to(device)
	# model = nn.DataParallel(model)
	checkpoint = torch.load('./checkpoints/'+str(name)+'_Model', map_location=device)
	model.load_state_dict(checkpoint)

	loss_func = nn.MSELoss()
	test_loss = test(model, test_loader, loss_func, device)

if __name__ == '__main__':
	main()
