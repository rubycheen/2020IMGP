from torchvision import datasets
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets,transforms

transform = transforms.Compose([transforms.Resize([13,13]),transforms.ToTensor()])
def data_loader(root, batch,train=True):
	global transform
	data = datasets.ImageFolder(root,transform=transform)
	loader = torch.utils.data.DataLoader(data,batch_size=batch, shuffle=train, drop_last=False)
	return loader
def make_cnn2(inp,out,pool=True):
	if pool:
		return nn.Sequential(
				nn.Conv2d(inp, out, 3,padding=1,padding_mode='circular'),
				nn.BatchNorm2d(out),
				nn.LeakyReLU(),
				nn.Conv2d(out, out, 3,padding=1,padding_mode='circular'),
				nn.BatchNorm2d(out),
				nn.LeakyReLU(),
				nn.Conv2d(out, out, 3,padding=1,padding_mode='circular'),
				nn.BatchNorm2d(out),
				nn.LeakyReLU(),

				nn.MaxPool2d(kernel_size=2)
				)
	else:
		return nn.Sequential(
				nn.Conv2d(inp, out, 3,padding=1,padding_mode='circular'),
				nn.BatchNorm2d(out),
				nn.Conv2d(out, out, 3,padding=1,padding_mode='circular'),
				nn.BatchNorm2d(out),
				nn.LeakyReLU(),
				#nn.MaxPool2d(kernel_size=2)
				)

class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		inp = 3
		out = 6
		self.conv1 = make_cnn2(3,6)#nn.Sequential(
		# 		nn.Conv2d(inp, out, 3,padding=1,padding_mode='circular'),
		# 		nn.BatchNorm2d(out),
		# 		nn.Conv2d(out, out, 3,padding=1,padding_mode='circular'),
		# 		nn.BatchNorm2d(out),
		# 		nn.LeakyReLU(),
		# 		nn.MaxPool2d(kernel_size=2)
		# 		)
		inp = 6
		out = 12
		self.conv2 = nn.Sequential(
				nn.Conv2d(inp, out, 3,padding=1,padding_mode='circular'),
				nn.BatchNorm2d(out),
				nn.LeakyReLU(),
				nn.Conv2d(out, out, 3,padding=1,padding_mode='circular'),
				nn.BatchNorm2d(out),
				nn.LeakyReLU(),
				nn.MaxPool2d(kernel_size=2)
				)
		inp = 12
		out = 24
		self.conv3 = nn.Sequential(
				nn.Conv2d(inp, out, 3,padding=1,padding_mode='circular'),
				nn.BatchNorm2d(out),
				nn.LeakyReLU(),
				nn.Conv2d(out, out, 3,padding=1,padding_mode='circular'),
				nn.BatchNorm2d(out),
				nn.LeakyReLU(),
				nn.MaxPool2d(kernel_size=2)
				)
		self.fc = nn.Sequential(nn.Linear(24, 18),
								#nn.LeakyReLU(),
								#nn.Dropout(0.2),
								#nn.Linear(54,36),
								#nn.LeakyReLU(),
								#nn.Dropout(0.1),
								#nn.Linear(36,18),
								nn.Softmax(dim=1)
								)
	def forward(self,x):
		bsz = x.size(0)
		out = self.conv1(x)
		out = self.conv2(out)
		out = self.conv3(out)
		out = out.reshape(bsz,-1)
		out = self.fc(out)
		return out


def train(path, bsz,epoch_num,save_path=None):
	loader = data_loader(path + '/train/',bsz,True)
	val_loader = data_loader(path + '/valid/',100,False)
	criterion = nn.CrossEntropyLoss()
	
	model = CNN()
	mode = model.cuda()
	opt = optim.SGD(model.parameters(),lr=0.01)
	for ep in range(epoch_num):
		c = 0.0
		ac = 0.0
		for x,y in loader:
			#print(x.size())
			#print(y)
			x = x.cuda()
			y = y.cuda()
			
			opt.zero_grad()
			pred = model(x)
			loss = criterion(pred,y)
			loss.backward()
			opt.step()
			c += torch.sum(torch.max(pred, 1)[1].cuda().data == y).type(torch.FloatTensor)
			ac += y.size(0)
		print('train',c/ac)
		c = 0.0
		num = 0.0
		for x,y in val_loader:
			print(y)
			x = x.cuda()
			y = y.cuda()
			pred = model(x)
			loss = nn.functional.cross_entropy(pred,y)
			c += torch.sum(torch.max(pred, 1)[1].cuda().data == y).type(torch.FloatTensor)
			num += y.size(0)
		print(ep,loss.data.cpu().numpy(),c/num,c,num)
		if ep % 3 == 0 and save_path != None:
			torch.save( model.state_dict() ,save_path + '/{}.tdic'.format(ep))

if __name__ == '__main__':
	train('.', 2, 1000, './checkpoints')



