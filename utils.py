import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader

def _load_data(DATA_PATH, batch_size, rotation):
	'''Data loader'''

	print("data_path: ", DATA_PATH)
	train_trans = transforms.Compose([transforms.RandomRotation(rotation),transforms.RandomHorizontalFlip(),\
								transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
	
	train_dataset = torchvision.datasets.MNIST(root=DATA_PATH, download=True,train=True, transform=train_trans)
	train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,num_workers=8)

	test_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
	test_dataset = torchvision.datasets.MNIST(root=DATA_PATH, download=True, train=False, transform=test_trans)
	test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
	
	return train_loader, test_loader

def _compute_accuracy(y_pred, y_batch):
	return (y_pred==y_batch).sum().item()

def adjust_learning_rate(learning_rate, optimizer, epoch, decay):
	"""Sets the learning rate to the initial LR decayed by 1/10 every args.lr epochs"""
	lr = learning_rate
	if (epoch > 5):
		lr = 0.001
	if (epoch >= 10):
		lr = 0.0001
	if (epoch > 20):
		lr = 0.00001
	
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr