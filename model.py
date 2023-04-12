"""
define moduals of model
"""
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

## MLPModel()
class CNNModel(nn.Module):
	"""docstring for ClassName"""
	
	def __init__(self):
		super(CNNModel, self).__init__()
		##-----------------------------------------------------------
		## define the model architecture here
		## MNIST image input size batch * 28 * 28 (one input channel)
		##-----------------------------------------------------------
		
		## define CNN layers below
		self.conv1 = nn.Sequential( nn.Conv2d(1, 32, kernel_size=3),
									nn.ReLU(),
									nn.MaxPool2d(kernel_size=2),
									nn.Dropout(0.2),
								)
		self.conv2 = nn.Sequential( nn.Conv2d(32, 32, kernel_size=3),
									nn.ReLU(),
									nn.MaxPool2d(kernel_size=2),
									nn.Dropout(0.2),
								)
		self.conv3 = nn.Sequential( nn.Conv2d(32, 64, kernel_size=3),
									nn.ReLU(),
									nn.MaxPool2d(kernel_size=2),
									nn.Dropout(0.2),
								)
		##------------------------------------------------
		## write code to define fully connected layer below
		##------------------------------------------------
		self.fc1 = nn.Sequential( 	nn.Linear(3*3*64, 256),
									nn.ReLU(),
									nn.MaxPool2d(kernel_size=2),
									nn.Dropout(0.2),
								)
		self.fc2 = nn.Linear(256, 10)
		

	'''feed features to the model'''
	def forward(self, x):  #default
		
		##---------------------------------------------------------
		## write code to feed input features to the CNN models defined above
		##---------------------------------------------------------
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		## write flatten tensor code below (it is done)
		#x = torch.flatten(x_out,1) # x_out is output of last layer
		x = x.view(x.size(0), -1)  
		x = self.fc1


		## ---------------------------------------------------
		## write fully connected layer (Linear layer) below
		## ---------------------------------------------------
		result = self.fc2(x)
		
		
		return result, x
        
		
		
	
		