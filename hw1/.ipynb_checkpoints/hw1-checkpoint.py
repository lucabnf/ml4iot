#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True, help="Input file name")
parser.add_argument("--output", type=str, required=True, help="Output file name")
parser.add_argument("--normalize", type=str, required=False, help="Normalization")
args = parser.parse_args()

#normalization
def normalize(df):
    t_max = 50
    t_min = 0
    h_max = 90
    h_min = 20
    df['temp'] = (df['temp']-tmin)/(tmax-tmin)
    df['hum'] = (df['hum']-hmin)/(hmax-hmin)
    
    return df

if(args.normalize == True):
    df = normalize(df)


# In[ ]:


import tensorflow as tf
import time as time
import datetime
import os
  
with tf.io.TFRecordWriter("prova") as writer:
    for row in df:
        timestamp = row[0] + "," + row[1]
        ts = datetime.datetime.strptime(timestamp, '%d/%m/%Y,%H:%M:%S' )
        posix_ts = int(time.mktime(ts.timetuple()))
        #posix_ts_enc = posix_ts.encode("utf-8")
        timestamp = tf.train.Feature(int64_list = tf.train.Int64List(value = [posix_ts]))
        temperature = tf.train.Feature(int64_list = tf.train.Int64List(value = [row[2]]))
        humidity = tf.train.Feature(int64_list = tf.train.Int64List(value = [row[3]]))
        #print(timestamp, temperature, humidity)
        mapping = {'timestamp': timestamp,
                   'temperature': temperature,
                   'humidity' : humidity}
        example = tf.train.Example(features=tf.train.Features(feature=mapping))
        print("Value of temperature-humidity-timestamp:", list(example.features.feature['temperature'].int64_list.value)[0], "-",                                                          list(example.features.feature['humidity'].int64_list.value)[0], "-",                                                          list(example.features.feature['timestamp'].int64_list.value)[0])
        
        writer.write(example.SerializeToString())

