import pandas as pd
import argparse
import tensorflow as tf
import time
import datetime
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True, help="Input file name")
parser.add_argument("--output", type=str, required=True, help="Output file name")
parser.add_argument("--normalize", default=False, action="store_true", required=False, help="Normalization")
args = parser.parse_args()

try:
    df = pd.read_csv(args.input, header=None, names=['date', 'time', 'temp', 'hum'])
except FileNotFoundError:
    print("Input file '"+args.input+"' does not exist. Shutting down...")
    sys.exit()

def normalize(df):
    t_max = 50
    t_min = 0
    h_max = 90
    h_min = 20
    df['temp'] = (df['temp']-t_min)/(t_max-t_min)
    df['hum'] = (df['hum']-h_min)/(h_max-h_min)
    
    return df

if(args.normalize == True):
    df = normalize(df)

with tf.io.TFRecordWriter(args.output) as writer:
    for row in df.values:
        timestamp = row[0] +","+row[1]
        ts = datetime.datetime.strptime(timestamp, '%d/%m/%Y,%H:%M:%S')
        posix_ts = int(time.mktime(ts.timetuple()))
        timestamp = tf.train.Feature(int64_list = tf.train.Int64List(value = [posix_ts]))

        if args.normalize is True:
            temperature = tf.train.Feature(float_list = tf.train.FloatList(value = [row[2]]))
            humidity = tf.train.Feature(float_list = tf.train.FloatList(value = [row[3]]))
        else:
            temperature = tf.train.Feature(int64_list=tf.train.Int64List(value=[row[2]]))
            humidity = tf.train.Feature(int64_list=tf.train.Int64List(value=[row[3]]))

        mapping = {'d': timestamp,
                   't': temperature,
                   'h' : humidity}
        example = tf.train.Example(features=tf.train.Features(feature=mapping))
        
        writer.write(example.SerializeToString())

print("\n----- OUTPUT -----")
print("{}B".format(os.path.getsize(args.output)))