#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 01:08:06 2018

@author: wangguanzhi
"""
'''
This code use another package's apriori algorithm.
The input data format and output format are different from the 'apriori' package.
'''

import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd  
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori

#Read data
flightdf=pd.read_csv('cleaned_data.csv')
flightdf.head()
columnsize=flightdf.columns.size
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Bin the "filed_airspeed_kts" to A-E. Then add the new column "filed_airspeed_kts_bin".

names=['A', 'B', 'C', 'D', 'E']
bins1=[0, 200, 300, 400, 450, 600]
flightdf['filed_airspeed_kts_bin'] = pd.cut(flightdf['filed_airspeed_kts'], bins1, labels=names)

#Data Proprocessing

no_arr=flightdf.loc[:,['aircrafttype','filed_airspeed_kts_bin','destination',
                       'actualarrivaltime_week',
                       'airline','arr_delay_sig']]
records1 = []
for i in range(0, 11595):
    records1.append([str(no_arr.values[i,j]) for j in range(0, 6)])

no_dep=flightdf.loc[:,['aircrafttype','filed_airspeed_kts_bin','origin',
                       'actualdeparturetime_week',
                       'airline','dep_delay_sig']]

records2 = []
for i in range(0, 11595):
    records2.append([str(no_dep.values[i,j]) for j in range(0, 6)])

#Transform the list to the dataframe
te = TransactionEncoder()
te_ary1 = te.fit(records1).transform(records1)
arrdf= pd.DataFrame(te_ary1, columns=te.columns_)
arrdf

te_ary2 = te.fit(records2).transform(records2)
depdf= pd.DataFrame(te_ary2, columns=te.columns_)
depdf

'''
In this code, we consider the min_support as 0.002, which is lower than all of three thresholds 
in "associate rule.py".

The general processes are do the algorithm, calculate each rule's size, and filter the rules that
satisfied the min_support and required size. Finally, we output these rules in .csv files.
'''
#Arrive part
frequent_itemsetsarr = apriori(arrdf, min_support=0.002, use_colnames=True,max_len=6)

frequent_itemsetsarr['length'] = frequent_itemsetsarr['itemsets'].apply(lambda x: len(x))

frequent_itemsetsarr[ (frequent_itemsetsarr['length'] >= 5) &
                   (frequent_itemsetsarr['support'] >= 0.002) ].sort_values(by=['support'])

#Departure part
frequent_itemsetsdep = apriori(depdf, min_support=0.002, use_colnames=True,max_len=6)
frequent_itemsetsdep['length'] = frequent_itemsetsdep['itemsets'].apply(lambda x: len(x))
frequent_itemsetsdep
frequent_itemsetsdep[ (frequent_itemsetsdep['length'] >= 5) &
                   (frequent_itemsetsdep['support'] >= 0.002) ].sort_values(by=['support'])

#Output the rules
f=open('mlxtend_arr_5.csv','a')
f.write(str(frequent_itemsetsarr[ (frequent_itemsetsarr['length'] == 5) &
                   (frequent_itemsetsarr['support'] >= 0.002) ].sort_values(by=['support'])))
f.close()
f=open('mlxtend_dep_5.csv','a')
f.write(str(frequent_itemsetsdep[ (frequent_itemsetsdep['length'] == 5) &
                   (frequent_itemsetsdep['support'] >= 0.002) ].sort_values(by=['support'])))
f.close()

f=open('mlxtend_arr_6.csv','a')
f.write(str(frequent_itemsetsarr[ (frequent_itemsetsarr['length'] == 6) &
                   (frequent_itemsetsarr['support'] >= 0.002) ].sort_values(by=['support'])))
f.close()

f=open('mlxtend_dep_6.csv','a')
f.write(str(frequent_itemsetsdep[ (frequent_itemsetsdep['length'] == 6) &
                   (frequent_itemsetsdep['support'] >= 0.002) ].sort_values(by=['support'])))
f.close()