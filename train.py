import argparse
import numpy as np
import sys
import os
sys.path.append('models/')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.parameter import Parameter 
import time

from dataloader import MyDataLoader
from ESPCN import MyModel

name = 'ESPCNx4'


def calc_psnr(img1, img2):
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))

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

def train(model, train_loader, optim, loss_func, epoch, device):
	model.train()
	#model = model.half()
	scaler = torch.cuda.amp.GradScaler()
	tot_loss = 0.
	count = 0.
	for batch_idx, (data, target) in enumerate(train_loader):
		data, target = data.to(device), target.to(device)
		#data = data.half()
		# target = target.half()
		optim.zero_grad()
		# print(data)
		with torch.cuda.amp.autocast():
			output = model(data)
			#print('before-------')
			#print(output)
			loss = loss_func(output, target)
		
		scaler.scale(loss).backward()
		#print(loss.item())
		scaler.unscale_(optim)
		nn.utils.clip_grad_norm_(model.parameters(),1)
		scaler.step(optim)
		scaler.update()
		tot_loss += loss.item()*data.size()[0]
		count += data.size()[0]


		if batch_idx % 10 == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),100. * batch_idx / len(train_loader), loss.item()), flush=True)
		
	tot_loss /= count
	print('Train Epoch: {} Loss: {:.6f}'.format(epoch, tot_loss),flush=True)
	return tot_loss

def test(model, test_loader, loss_func, epoch, device):
	model.eval()
	test_loss = 0.
	test_psnr = 0.
	count = 0.
	time_list = []
	with torch.no_grad():
		for batch_idx, (data, target) in enumerate(test_loader):
			data, target = data.to(device), target.to(device)
			#data = data.half()
			#model = model.half()
			with torch.cuda.amp.autocast():
				output = model(data)
			#output = output.float()
				loss = loss_func(output, target)
			psnr = psnr_func(output, target)
			test_loss += loss.item()*data.size()[0]
			test_psnr += psnr.item()*data.size()[0]
			count += data.size()[0]

	test_loss /= count
	test_psnr /= count
	print('Test Epoch: {} Loss: {:.6f}'.format(epoch, test_loss),flush=True)
	print('Test Epoch: {} PSNR: {:.6f}'.format(epoch, test_psnr),flush=True)
  # print('Time: {:.4f}'.format(avg_time),flush=True)
	return test_loss

def main():
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")
	torch.manual_seed(0)
	trainset = MyDataLoader('../traindata/', 'train', in_ch=3)
	testset = MyDataLoader('../testdata/', 'test', in_ch = 3)

	train_loader = DataLoader(trainset, 512, shuffle = True,pin_memory=True,  num_workers=2)
	test_loader = DataLoader(testset, 512, shuffle = True, pin_memory=True,num_workers=2)

	model = MyModel().to(device)
	print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
	model = nn.DataParallel(model)
	#model = model.half()
	optimizer = optim.Adam(model.parameters(), lr = 1e-3)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma = 0.9)
	start_epoch = 0
	best_loss = 100000
	EPOCH = 50

	loss_func = nn.MSELoss()

	for epoch in range(start_epoch+1, start_epoch+EPOCH+1):
		train_loss = train(model, train_loader, optimizer, loss_func, epoch, device)
		test_loss = test(model, test_loader, loss_func, epoch, device)
		scheduler.step()

		if test_loss < best_loss:
			if not os.path.isdir('checkpoints/'):
				os.mkdir('checkpoints/')
			#save_model = model.half()
			torch.save(model.state_dict(), './checkpoints/'+str(name)+'_Model')
			best_loss = test_loss
			#model = model.float()

if __name__ == '__main__':
  main()




