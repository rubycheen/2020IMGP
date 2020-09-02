import torch.nn as nn

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

class CNN(nn.Module): # first
	def __init__(self):
		super(CNN, self).__init__()
		inp = 3
		out = 6
		self.conv1 = make_cnn2(3,6)
		#nn.Sequential(
		# 		nn.Conv2d(inp, out, 3,padding=1,padding_mode='circular'),
		# 		nn.BatchNorm2d(out),
		# 		nn.Conv2d(out, out, 3,padding=1,padding_mode='circular'),
		# 		nn.BatchNorm2d(out),
		# 		nn.LeakyReLU(),
				# nn.MaxPool2d(kernel_size=2)
		# 		)
		inp = 6
		out = 12
		self.conv2 = make_cnn2(6,12)
		self.conv3 = make_cnn2(12,24)
		inp = 24
		out = 48
		self.conv4 = nn.Sequential(
				nn.Conv2d(inp, out, 3,padding=1,padding_mode='circular'),
				nn.BatchNorm2d(out),
				nn.LeakyReLU(),
				nn.Conv2d(out, out, 3,padding=1,padding_mode='circular'),
				nn.BatchNorm2d(out),
				nn.LeakyReLU(),
				nn.MaxPool2d(kernel_size=2)
				)
		self.fc = nn.Sequential(nn.Linear(48, 18),
								
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
		out = self.conv4(out)
		out = out.reshape(bsz,-1)
		out = self.fc(out)
		return out
