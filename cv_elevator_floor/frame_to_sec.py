#!/usr/bin/env python
# coding: utf-8

# In[28]:


# import json  

# num_filename = ('Data/0.mov.estimate.json')
# floor_filename = ('Data/0.mov.floor.json')  
# open_filename = ('Data/0.mov.open.json')  

# num_frame = json.load(open(num_filename))
# floor_frame = json.load(open(floor_filename)) 
# is_open_frame = json.load(open(open_filename)) 


# In[12]:


# import cv2

# cctv = cv2.VideoCapture("0 - 2020-05-04 12-06-03-362.mov")
# cctv_frame = int(cctv.get(cv2.CAP_PROP_FRAME_COUNT))
# print('CCTV frame:', cctv_frame)
# cctv_start_time = '2020-05-04 12:06:03'
# fps = 7.49


# In[13]:


# print('num',len(num_frame))
# print('floor',len(floor_frame),len(floor_frame[0]),floor_frame[1])
# print('open',len(open_frame))
# video_duration = 4058
# fps = len(num_frame) / video_duration
# print("It's", fps, 'fps') ## it would be a constant


# In[14]:


# video_length = int(cctv_frame / fps)
# print('video_length',video_length)


# In[15]:


def frame_to_sec(source):

    from collections import Counter
    output = []
    is_odd = True
    index = 0
    for i in range(video_length):
        tmp = []
        frame_per_second = 7
        if is_odd == False:
            frame_per_second = frame_per_second + 1
#         print('In', i+1, 'sec', frame_per_second, 'frames')
        try:
            for j in range(frame_per_second):
#                 print('index', index, source[index])
                tmp.append(source[index])
                index += 1
            count = Counter(tmp)
            num = 0
            maximun = -1
            for i in count.keys():
                if(count[i] > maximun):
                    maximun = count[i]
                    num = i
            output.append(num)        
            print(count)
    #         print('num',num)
    #         print('output',output)
            is_odd = not(is_odd)
        except:
            return output
            break


# In[30]:


# people_num_per_sec = frame_to_sec(num_frame)


# In[18]:


# floor_per_sec = frame_to_sec(floor_frame[0])


# In[19]:


# is_open_per_sec = frame_to_sec(is_open_frame)


# In[20]:


def time_stamp(cctv_start_time, length):
    import time
    from datetime import datetime
    time_stamp = []
    start_time = time.strptime(cctv_start_time, "%Y-%m-%d %H:%M:%S")
    timeStamp = int(time.mktime(start_time))

    for i in range(length):
        data_time = time.localtime(timeStamp)
        time_stamp.append(time.strftime("%Y-%m-%d %H:%M:%S", data_time))
        timeStamp += 1
        print(time_stamp[i])
    return time_stamp


# In[31]:


# print(len(time_stamp), len(floor_per_sec), len(is_open_per_sec), len(people_num_per_sec))


# In[22]:


# print(type(floor_per_sec[0]))


# In[23]:


def convert_floor(floor_per_sec):
    for i in range(len(floor_per_sec)):
        if floor_per_sec[i] == 0:
            floor_per_sec[i] = 'B4'
        elif floor_per_sec[i] == 1:
            floor_per_sec[i] = 'B3'
        elif floor_per_sec[i] == 2:
            floor_per_sec[i] = 'B2'
        elif floor_per_sec[i] == 3:
            floor_per_sec[i] = 'B1'
        elif floor_per_sec[i] == 4:
            floor_per_sec[i] = '1'
        elif floor_per_sec[i] == 5:
            floor_per_sec[i] = '2'
        elif floor_per_sec[i] == 6:
            floor_per_sec[i] = '3'
        elif floor_per_sec[i] == 7:
            floor_per_sec[i] = '4'
        elif floor_per_sec[i] == 8:
            floor_per_sec[i] = '5'
        elif floor_per_sec[i] == 9:
            floor_per_sec[i] = '6'
        elif floor_per_sec[i] == 10:
            floor_per_sec[i] = '7'
        elif floor_per_sec[i] == 11:
            floor_per_sec[i] = '8'
        elif floor_per_sec[i] == 12:
            floor_per_sec[i] = '9'
        elif floor_per_sec[i] == 13:
            floor_per_sec[i] = '10'
        elif floor_per_sec[i] == 14:
            floor_per_sec[i] = '11'
        elif floor_per_sec[i] == 15:
            floor_per_sec[i] = '12'
        elif floor_per_sec[i] == 16:
            floor_per_sec[i] = '13'
        elif floor_per_sec[i] == 17:
            floor_per_sec[i] = '14'
        elif floor_per_sec[i] == 18:
            floor_per_sec[i] = '15'


# In[35]:


# #生成csv/json

# # def data
# data = {'Time Stamp':time_stamp, 'Floor':floor_per_sec, 'Is Open':is_open_per_sec, 'Number of People':people_num_per_sec}
# # print(data)

# with open('data.json', 'w') as f:
#     json.dump(data, f)

# from pandas.core.frame import DataFrame
# data_df = DataFrame(data)
# print(data_df)
# data_df.to_csv('data.csv')


# In[ ]:




