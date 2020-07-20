import test
import cv2
#to be done
class node
def get_prob_list(path):
	cap = cv2.VideoCapture(path)
	ret = True
	l = []
	while ret:
		ret, frame = cat.read()
		if ret:
			l.append(pred(frame))
	return l
def check_validation(l, new_item):
	floors = [1,10,11,12,13,14,15,2,3,4,5,6,7,8,9,0,-1,-2]
	if len(l) == 0:
		return True
	elif new_item == l:
		return True
	elif abs(floor[new_item] - floor[l-1]) > 1:
		return False
	for i in range(min(3,len(l))):
		if l[-1-i] != l[-1]:
			return False
	else:
		return True

def beam(beam_size, prob_list):
	i = 0
	old_list = [[[],0]]
	list_len = len(prob_list)
	while i < prob_list_len:
		new_list = []
		for l in old_list:
			for j in range(18):
				if check_validation(l[0], j):
					tmp = l[0].copy()
					tmp.append(j)
					new_list.append([tmp,l[1] + prob_list[i][j] ])
		new_list.sort(reverse=True, key=lambda x: x[1])
		old_list = new_list[:beam_size]
		i++
	return old_list[0]
def get_floor_list(path,beam_size=18):
	prob_list = get_prob_list(path)
	return beam(beam_size,prob_list)

if __name__ == '__main__':
	main()







		
		



