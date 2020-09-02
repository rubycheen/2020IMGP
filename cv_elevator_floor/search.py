import test
import cv2
#to be done


def get_prob_list(path):
	cap = cv2.VideoCapture(path)
	ret = True
	l = []
	ct = 0
	while ret:
		ret, frame = cap.read()
		#if ct == 1000:
		#	break
		ct += 1
		if ret:
			frame = frame[25 : 45, 1534 : 1562]
			frame = cv2.flip(frame, 1)
			l.append(test.predict(frame).squeeze().tolist())
	return l

def check_validation(l, new_item):
	floors = [1,10,11,12,13,14,15,2,3,4,5,6,7,8,9,0,-1,-2]
	if len(l) == 0:
		return True
	elif new_item == l[-1]:
		return True
	elif abs(floors[new_item] - floors[l[-1]]) > 1:
		return False
	for i in range(min(10,len(l))):
		if l[-1-i] != l[-1]:
			return False
	else:
		return True

def beam(beam_size, prob_list):
	i = 0
	old_list = [[ [], 0 ]]
	prob_list_len = len(prob_list)
	while i < prob_list_len:
		print(i)
		new_list = []
		for l in old_list:
			for j in range(18):
				if check_validation(l[0], j):
					tmp = l[0].copy()
					tmp.append(j)
					#print(l)
					#print(prob_list)
					new_list.append([tmp, l[1] + prob_list[i][j] ])
		new_list.sort(reverse=True, key=lambda x: x[1])
		old_list = new_list[:beam_size]
		i += 1
	return old_list[0]

def get_floor_list(path, beam_size=180):
	prob_list = get_prob_list(path)
	print(prob_list)
	return beam(beam_size,prob_list)

def get_is_open(floor_list):
	start = 0
	i = 1
	isOpen = [0] * len(floor_list)
	while i < len(floor_list):
		if floor_list[i] != floor_list[i-1] or i == len(floor_list) - 1:
			if i - start > 18:
				for j in range(start,i):
					isOpen[j] = 1
			start = i
		i += 1
	return isOpen




if __name__ == '__main__':
	import json
	file_list = ['../0.mov']
	for f in file_list:
		print(f)

		# l = get_floor_list(f)
		# json.dump(l,open(f + '.json','w'))

		l = json.load(open(f + '.json'))

		l = get_isOpen(l[0])
		json.dump(l,open(f + '.open', 'w'))

		











		
		



