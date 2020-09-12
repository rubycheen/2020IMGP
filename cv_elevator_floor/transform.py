#!/usr/bin/env python
# coding: utf-8
from collections import Counter

def frame_to_sec(source, fps):

    output = []
    is_odd = True
    index = 0

    for i in range(fps):
        tmp = []
        frame_per_second = 7
        if is_odd == False:
            frame_per_second = frame_per_second + 1
        try:
            for i in range(frame_per_second):
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
            is_odd = not(is_odd)
        except:
            return output
    return output


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
    return time_stamp


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
    return floor_per_sec