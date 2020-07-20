from train import CNN, transform
import torch
import numpy as np


import cv2
import numpy as np
from PIL import Image

model = CNN()
model.load_state_dict(torch.load('105.tdic'))

def cv2pil(img):
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	im_pil = Image.fromarray(img)
	return im_pil

def predict(frame):
	global model
	img = cv2pil(frame)
	img =transform(img)
	img = img.unsqueeze(0)
	pred = model(img)

	return pred

def guess(frame):
	pred = predict(frame)

	ans = torch.max(pred,1)[1][0]

	return ans


if __name__ == '__main__':
	for i in range(652):

 		img = cv2.imread("../data/{}.jpg".format(i)) 
 		pred = guess(img).item()
 		cv2.imwrite('{}/{}.jpg'.format(pred,i),img)
 		print(i,pred)



