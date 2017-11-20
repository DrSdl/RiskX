"""
Download hourly wind data history of dublin airport
Data from http://www.met.ie/climate-request/   => dublin_hly532.csv
Please convert file from UNIX to DOS format if working on Windows
Plot all the results in a 2D color plot
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from datetime import datetime

count=0

# simple csv file reading to see what is in it
with open('./winddata/dublin_hly532.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONE)
    for row in reader:
        if count<10:
            print(row[0], row[13])
        count=count+1
        #print(row)


# read csv file into a pandas dataframe
# date,ind,rain,ind,temp,ind,wetb,dewpt,vappr,rhum,msl,ind,wdsp,ind,wddir,ww,w,sun,vis,clht,clamt
# 01-jan-1987 00:00,0,0.1,0,6.8,0,6.4,5.9,9.3,94,1004.9,2,5,2,130,61,62,0.0,6000,40,8
# in the original data there is the line:
# 26-sep-1998 04:00,0,0.2,0,13.7,0,13.7,13.7, , ,1003.8,2,10,2,50,60,64,0.0,1300,3,8
# 16-jul-2007 18:00,0,0.0,0,17.8,0,14.5,11.9,13.9,68,1005.2,2,10,7, ,15,98,0.5,30000,90,5
# which has two missing values => https://pandas.pydata.org/pandas-docs/stable/missing_data.html

parse_dates = ['date']
dtypes = {'date': str, 'ind': int, 'rain': float, 'temp': float, 'wetb': float, 'dewpt': float, 'vappr': float, 'rhum': float, 'msl': float, 'wdsp': int, 'wddir': int, 'ww': int, 'w': int, 'sun': float, 'vis': int, 'clht': int, 'clamt': int}

wind=pd.read_csv('./winddata/dublin_hly532.csv', delimiter=',', parse_dates=parse_dates, dtype=dtypes)
#wind=pd.read_csv('./winddata/dublin_hly532.csv')
#print(wind)

# pandas dataframe plot option info: https://pandas.pydata.org/pandas-docs/stable/visualization.html
#wind.plot(x='date', y='wdsp')
#plt.show()

Nrows = wind.shape[0] # get number of rows
Ncols = wind.shape[1] # get number of columns

xedges = np.arange(0,25)   # x-axis: number of hour
yedges = np.arange(0,367)  # y-axis: number of day of year

xhrs= np.ndarray([]);  # array to hold the hour-day tuples
yday= np.ndarray([]);
zdat= np.ndarray([]);  # array for the windspeed

for i in range(0,Nrows):
    hxx=wind['date'][i]          # get timestamp
    hrs=hxx.hour                 # get hour of timestamp
    hct=hxx.timetuple().tm_yday  # convert day to interval [1...365]
    wsp=wind['wdsp'][i]          # get windspeed
    xhrs=np.append(xhrs,hrs)     # add values to numpy arrays
    yday=np.append(yday,hct)
    zdat=np.append(zdat,wsp)

H, xed, yed = np.histogram2d(xhrs, yday, bins=(xedges, yedges), weights=zdat)

# colormap definitions:
# https://matplotlib.org/examples/color/colormaps_reference.html
plt.rcParams['image.cmap'] = 'plasma'

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111)
ax.set_title('pcolormesh: wind intensity as function of hour and day per year')
X, Y = np.meshgrid(xed, yed)
ax.pcolormesh(X, Y, H.T)
#ax.set_aspect('equal')
ax.set_xlabel('hour')
ax.set_ylabel('day of year')

plt.show()
