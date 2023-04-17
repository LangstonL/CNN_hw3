from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

class CNNModel(nn.Module):
	
	def __init__(self):
		super(CNNModel, self).__init__()
		
		self.conv_layers = nn.Sequential( 	
			#1st conv layer
			nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			#2nd conv layer
			nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
			nn.MaxPool2d(2,2),
			nn.BatchNorm2d(20),
			nn.Dropout(0.2),
			#3rd conv layer, output =4-5
			nn.Conv2d(in_channels=20, out_channels=30, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
			nn.MaxPool2d(2,2),
			nn.BatchNorm2d(30),
			nn.Dropout(0.2),
		)
		
		self.fc_layers = nn.Sequential(
			nn.Linear(7*7*30, 256),
			nn.ReLU(),
			nn.Dropout(0.2),
			nn.Linear(256, 10)
		)

	def forward(self, x):
		x = self.conv_layers(x)
		x = x.view(-1, 7*7*30)  
		x = self.fc_layers(x)
		return x