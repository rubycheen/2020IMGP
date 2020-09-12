#!/usr/bin/env python
# coding: utf-8

import search
import numEstimate
import transform
import cv2
import pandas
from pandas.core.frame import DataFrame

#影片輸入, 拿到資料
source = "test.mp4"
floor_frame = search.get_floor_list(source)
is_open_frame = search.get_is_open(floor_frame[0])
people_num_frame = numEstimate.det_people_num(is_open_frame, source = source)
cctv_start_time = '2020-05-04 12:06:03' #這裡要手動輸入
fps = len(is_open_frame)

#單位轉換成秒數
people_num_per_sec = transform.frame_to_sec(people_num_frame , fps)
floor_per_sec = transform.frame_to_sec(floor_frame[0], fps)
is_open_per_sec = transform.frame_to_sec(is_open_frame, fps)

#轉換樓層名稱
floor_per_sec = transform.convert_floor(floor_per_sec)

#產生時間戳
sec = len(floor_per_sec)
time_stamp = transform.time_stamp(cctv_start_time, sec)

# def data
data = {'Time Stamp':time_stamp, 'Floor':floor_per_sec, 'Is Open':is_open_per_sec, 'Number of People':people_num_per_sec}
data_df = DataFrame(data)
data_df.to_csv('output/data.csv')