import pandas as pd
import glob
import numpy as np
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

import seaborn as sns
import matplotlib.pyplot as plt

path = r'testfiles' # use your path
all_files = glob.glob(path + "/*.csv")
li = [] 

#diavasma arxeiwn apo fakelo kai append se mia lista
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

#pernei tin lista poy dimiourgithike pio panw kai vazei ola ta arxeia se ena
frame = pd.concat(li, axis=0, ignore_index=True,sort=True)

frame['statetime'] =pd.to_datetime(frame['statetime'])
frame.set_index(pd.DatetimeIndex(frame['statetime']),inplace=True)

frame['humidity_in'].replace('unknown',0,inplace=True)  #fill uknown with 0
frame['humidity_out'].replace('unknown',0,inplace=True)  
frame['temperature_in'].replace('unknown',0,inplace=True)  #fill uknown with 0
frame['temperature_out'].replace('unknown',0,inplace=True)  

frame['humidity_in'] = frame['humidity_in'].astype(float)
frame['humidity_out'] = frame['humidity_out'].astype(float)
frame['temperature_in'] = frame['temperature_in'].astype(float)
frame['temperature_out'] = frame['temperature_out'].astype(float)
frame['motion_detected'].replace('EA674E',1,inplace=True) 

frame = frame.groupby(pd.Grouper(freq='1Min'))['humidity_in','humidity_out','motion_detected','temperature_in','temperature_out'].mean()


# frame = frame.dropna(thresh=1) #

frame['week'] = frame.index.week  #find week based on statetime
frame['hour'] = frame.index.hour #find week based on statetime
frame['minute'] = frame.index.minute #find week based on statetime
frame['quarter'] = frame.index.quarter #find week based on statetime
frame['month'] = frame.index.month #find week based on statetime
frame['weekday'] = frame.index.weekday #find weekday based on statetime
frame['day'] = frame.index.day  #find week based on statetime

frame['motion_detected'].fillna(0,inplace=True)

#gemisma kenwn me mean stin vdomada kai an den iparxei metrisi g olokliri tin vdomada mean se olokliri tin stili
frame["temperature_in"].fillna(frame.groupby("day")["temperature_in"].transform('mean'), inplace=True)
frame["temperature_in"].fillna(frame["temperature_in"].mean(), inplace=True)
frame["temperature_out"].fillna(frame.groupby("day")["temperature_out"].transform('mean'), inplace=True)
frame["temperature_out"].fillna(frame["temperature_out"].mean(), inplace=True)
frame["humidity_in"].fillna(frame.groupby("day")["humidity_in"].transform('mean'), inplace=True)
frame["humidity_in"].fillna(frame["humidity_in"].mean(), inplace=True)
frame["humidity_out"].fillna(frame.groupby("day")["humidity_out"].transform('mean'), inplace=True)
frame["humidity_out"].fillna(frame["humidity_out"].mean(), inplace=True)
frame['motion_detected'].fillna(0,inplace=True)
# Reset the index of dataf+rame
frame = frame.reset_index()

#dimiourgia bins me morning-afternoon-night klp
bins = [0, 5, 13, 17, 25]
labels = ['1','2','3','4']
hours = frame['statetime'].dt.hour
frame['bin'] = pd.cut(hours-5+24 *(hours<5),bins=bins,labels=labels,right=False)

bintemp = [-10, 0, 15, 25, 33, 50]
labels = [1,2,3,4,5]
frame['temp'] = pd.cut(frame['temperature_out']-0+49 *(frame['temperature_out']<0),bins=bintemp,labels=labels,right=False)
frame['temp'].fillna(3,inplace=True)
frame['statetime']= frame['statetime'].dt.strftime('%Y-%m-%d %H:%M:%S') #metatropi date se str

frame.to_csv("hellotest.csv") #metatrepo se csv arxeio
