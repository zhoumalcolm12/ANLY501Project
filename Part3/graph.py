#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 14:41:57 2018

@author: zhoumengzhi
"""

import plotly.plotly as py
from plotly.graph_objs import *
import pandas as pd
import plotly 
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import numpy as np
import seaborn as sns

##Set credentials with username and KEY
plotly.tools.set_credentials_file(username='zhoumalcolm12',api_key='K3Qct6JLc22CUsKoomvS')


#data clean
df = pd.read_csv('cleandata.csv', encoding = "ISO-8859-1")
df.drop(['t_diff','cloud_altitude','temp','dewpoint','visibility','wind_speed','gust_speed'], axis = 1, inplace = True, errors = 'ignore')
df.clean = df.dropna()
df.clean = df.clean[df.clean.airline != 'unkown']

#create a new column: dalay
def label_delay_arr (row):
   if row['arr_delay_min'] > 0 :
      return 1
   return '0'

def label_delay_dep (row):
   if row['dep_delay_min'] > 10 :
      return 1
   return '0'

df.clean['delay_arr'] = df.clean.apply (lambda row: label_delay_arr (row),axis=1)
df.clean['delay_dep'] = df.clean.apply (lambda row: label_delay_dep (row),axis=1)

#####################################################
#1st graph
#radar gram(without weather)
attr_list = ['aircraft type', 
                'origin', 'destination', 'filed departure time(week)',
                'filed departure time(hr)', 'estimated arrival time(hr)', 'airline',
                'aircraft manufacturer','aircraft type']
feature_importances=[0.08550837, 0.14906092, 0.1630565,  0.25129193, 0.0653423,  0.16155004,
 0.08253172, 0.04165823]

data = [go.Scatterpolar(
  r = feature_importances,
  theta = attr_list,
  fill = 'toself'
)]

layout = go.Layout(title = 'Radar Graph for Feature Importance',
  polar = dict(
    radialaxis = dict(
      visible = True,
      range = [0, 0.30]
    )
  ),
  showlegend = False
)
fig = go.Figure(data=data, layout=layout)
py.plot(fig, filename = 'radar')
plotly.offline.plot(fig, filename = 'radar_feature.html')



#radar gram(with weather)
attr_list = ['aircraft type', 'filed airspeed(kts)', 'filed altitude', 
                     'origin', 'destination',
                     'filed departuretime(hr)', 'estimated arrival time(hr)', 'airline',
                     'aircraft manufacturer',
                     'cloud altitude','temperature','dew point','visibility',
                     'wind speed','gust speed','aircraft type']
feature_importances=[0.07676359, 0.17079143, 0.10234315, 0.03712042, 0.09973636, 0.05304062,
 0.04998065, 0.04114598, 0.04027695, 0.11216917, 0.04421938, 0.06899675,
 0.00545371, 0.08449094, 0.0134709 ]

data = [go.Scatterpolar(
  r = feature_importances,
  theta = attr_list,
  fill = 'toself'
)]

layout = go.Layout(title = 'Radar Graph for Feature Importance (with weather factor)',
  polar = dict(
    radialaxis = dict(
      visible = True,
      range = [0, 0.20]
    )
  ),
  showlegend = False
)
fig = go.Figure(data=data, layout=layout)
py.plot(fig, filename = 'radar_w')
plotly.offline.plot(fig, filename = 'radar_feature_w.html')

#####################################################
#2nd graph
#histogram
x = df.clean.airline
y = df.clean.delay_dep
z = df.clean.delay_arr

data = [
  go.Histogram(
    histfunc = "count",
    y = y,
    x = x,
    name = "Total Number of Flight"
  ),
  go.Histogram(
    histfunc = "sum",
    y = y,
    x = x,
    name = "Number of Depature Delayed Flight"
  ),
            go.Histogram(
    histfunc = "sum",
    y = z,
    x = x,
    name = "Number of Arrival Delayed Flight"
  )
]

layout = go.Layout(
    title='Histogram for depature flights',
    xaxis=dict(
        title='Airline Company'
    ),
    yaxis=dict(
        title='Count'
    ),
    bargap=0.2,
    bargroupgap=0.1
)
fig1 = go.Figure(data=data, layout=layout)
fig1['layout'].update(title = "some title")
py.plot(data, filename='hist_depature')
plotly.offline.plot(data, filename = 'hist.html')