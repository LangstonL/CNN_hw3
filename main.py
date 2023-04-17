import numpy as np
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from model import CNNModel
from utils import _load_data, _compute_accuracy, adjust_learning_rate
import wandb

# hyper params
mode = 'train'
epoches = 10
learning_rate = 0.01
decay = 0.5
batch_size = 100
rotation = 15

def main():
	# select gpu or cpu
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")
	print("device: ", device)
	if use_cuda:
		torch.cuda.manual_seed(72)
	
	# load data
	DATA_PATH = "./data/"
	train_loader, test_loader=_load_data(DATA_PATH, batch_size, rotation)

	model = CNNModel()
	model.to(device)

	optimizer = optim.Adam(model.parameters(),lr=learning_rate)
	loss_fun = nn.CrossEntropyLoss()
	
	#  model training
	if mode == 'train':
		model = model.train()
		for epoch in range(epoches):
			adjust_learning_rate(learning_rate, optimizer, epoch, decay)
			
			for i, (x_batch,y_labels) in enumerate(train_loader):
				x_batch,y_labels = Variable(x_batch).to(device), Variable(y_labels).to(device)

				# feed input into model
				output_y = model(x_batch)
				loss = loss_fun(output_y, y_labels)

				# back prop
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				#get pred and accy
				y_pred = torch.argmax(output_y.data, 1)
				accy = _compute_accuracy(y_pred, y_labels)/batch_size
				
				#output to wandb
				if i%100==0:
					print('iter: {} loss: {}, accy: {}'.format(i, loss.item(), accy))
					wandb.log({'iter': iter, 'loss': loss.item()})
					wandb.log({'iter': iter, 'accy': accy})
			
	# model testing
	total=0
	accy_ct=0
	model.eval()
	with torch.no_grad():
		for (imgs, labels) in test_loader:
			imgs, labels = imgs.to(device), labels.to(device)
			outputs = model(imgs)

			y_pred = torch.argmax(outputs.data, 1)
			
			total += labels.size(0)
			accy_ct += _compute_accuracy(y_pred, labels)

	accy = accy_ct/total
	print('testing accy: ', accy)
	
if __name__ == '__main__':
	time_start = time.time()
	with wandb.init(project='MLP', name='MLP_demo'):
		main()
	time_end = time.time()
	print("running time: ", (time_end - time_start)/60.0, "mins")
	

