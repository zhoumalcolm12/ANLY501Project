#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 21:57:05 2018

@author: wangguanzhi
"""

import requests
import pandas as pd
import datetime
import time
totaldata=[]

airport=pd.read_csv('airports.csv',header=None)
print(airport)
airport=list(airport[1])
airport[10]='KIAD'
airport[23]='KDCA'
print(airport[23])
#t=int(time.mktime(time.strptime('2018-9-30 9:00:00', '%Y-%m-%d %H:%M:%S')))
username = 'yangxi1994'
apikey = '7038ce657e42de52eaf5af4126a5fbbf0f4f0a43'
url_base = ''.join(['http://',username,':',apikey,'@flightxml.flightaware.com/json/FlightXML2/MetarEx'])
tm1=[]
tm1.append(int(time.mktime(time.strptime('2018-9-30 9:00:00', '%Y-%m-%d %H:%M:%S'))))
tm1.append(int(time.mktime(time.strptime('2018-9-30 21:00:00', '%Y-%m-%d %H:%M:%S'))))
#print(tm1)
for i in range(1,7):
    temp1='2018-10-'+str(i)+' 9:00:00'
    tm1.append(int(time.mktime(time.strptime(temp1, '%Y-%m-%d %H:%M:%S'))))
    temp2='2018-10-'+str(i)+' 21:00:00'
    tm1.append(int(time.mktime(time.strptime(temp2, '%Y-%m-%d %H:%M:%S'))))
#print(tm1)
#print(tm1[13])
for p in range(0,28):
    for q in range(0,14):
        
        url_post = {'airport':airport[p],
                        'startTime':tm1[q],
                        'howMany':'12',
                        'offset':'0'
                        
                        }
        
        
        response=requests.get(url_base, url_post)
        
        jsontxt = response.json()
        #print(jsontxt)
        basicdata=jsontxt['MetarExResult']['metar']
        print(p,q)
        totaldata.append(basicdata)
print(totaldata[0])
df=pd.DataFrame(totaldata[0])
pd.set_option('display.max_columns', None)
print(df)
df=df.append(totaldata[5])
print(df)
print(len(df))
print(len(totaldata))
print(totaldata[391])
for r in range(1,392):
    df=df.append(totaldata[r])
#print(df[:50])
df.index = range(len(df))

df.to_csv('Weather_Output/weather.csv')
