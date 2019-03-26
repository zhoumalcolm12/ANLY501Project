#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 17:34:11 2018

@author: wangguanzhi
"""

import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd  
from apyori import apriori 

#Read data#
flightdf=pd.read_csv('cleaned_data.csv')
flightdf.head()
columnsize=flightdf.columns.size

# Bin the "filed_airspeed_kts" to A-E. Then add the new column "filed_airspeed_kts_bin".

names=['A', 'B', 'C', 'D', 'E']
bins1=[0, 200, 300, 400, 450, 600]
flightdf['filed_airspeed_kts_bin'] = pd.cut(flightdf['filed_airspeed_kts'], bins1, labels=names)

#Only consider arrive delay.
'''
===============================================================================================
'''

no_arr=flightdf.loc[:,['aircrafttype','filed_airspeed_kts_bin','destination',
                       'actualarrivaltime_week',
                       'airline','arr_delay_sig']]
#Data Proprocessing
records1 = []
for i in range(0, 11595):
    records1.append([str(no_arr.values[i,j]) for j in range(0, 6)])

'''
we set 3 thresholds.
1. 
min_support=0.005, min_confidence=0.2, min_lift=3, min_length=5
2.
min_support=0.01, min_confidence=0.2, min_lift=3, min_length=5
3.
min_support=0.05, min_confidence=0.2, min_lift=3, min_length=5
'''
#Use the apriori algorithm
association_rules1 = apriori(records1, min_support=0.01, min_confidence=0.2, min_lift=3, min_len=5)  
association_results1 = list(association_rules1) 
#print(type(association_rules1))
#print(association_results1)
#print(len(association_results1))
#print(len(association_results1[0]))

#Transform the association_results data and output to .txt file#
for item in association_results1:
    
    # first index of the inner list
    # Contains base item and add item
    pair = item[0] 
    items = [x for x in pair]
    f=open("arr_0.01.txt","a+")
    f.write("Rule: " + str(items[0:len(items)-1]) + " -> " + str(items[len(items)-1])+'\n')
    f.write("Support: " + str(item[1])+'\n')
    f.write("Confidence: " + str(item[2][0][2])+'\n')
    f.write("Lift: " + str(item[2][0][3])+'\n')
    f.write("=====================================\n")
f.close()

association_rules2 = apriori(records1, min_support=0.05, min_confidence=0.2, min_lift=3, min_len=5)  
association_results2 = list(association_rules2) 
#print(association_results2)
#print(len(association_results2))
#print(len(association_results2[0]))
#print(association_results2[30])
for item in association_results2:
    
    # first index of the inner list
    # Contains base item and add item
    pair = item[0] 
    items = [x for x in pair]
    #print(item[2][0][2])
    #print(items)
    #print(items[0:len(items)-1])
    f=open("arr_0.05.txt","a+")
    f.write("Rule: " + str(items[0:len(items)-1]) + " -> " + str(items[len(items)-1])+'\n')
    f.write("Support: " + str(item[1])+'\n')
    
    f.write("Confidence: " + str(item[2][0][2])+'\n')
    f.write("Lift: " + str(item[2][0][3])+'\n')
    f.write("=====================================\n")
f.close()

association_rules3 = apriori(records1, min_support=0.005, min_confidence=0.2, min_lift=3, min_len=5)  
association_results3 = list(association_rules3) 
#print(association_results3[0])
#print(len(association_results3))
#print(len(association_results3[0]))

for item in association_results3:
    
   
    pair = item[0] 
    items = [x for x in pair]
    f=open("arr_0.005.txt","a+")
    f.write("Rule: " + str(items[0:len(items)-1]) + " -> " + str(items[len(items)-1])+'\n')
    f.write("Support: " + str(item[1])+'\n')
    f.write("Confidence: " + str(item[2][0][2])+'\n')
    f.write("Lift: " + str(item[2][0][3])+'\n')
    f.write("=====================================\n")
f.close()

'''
================================================================================================
'''
#Departure part #
no_dep=flightdf.loc[:,['aircrafttype','filed_airspeed_kts_bin','origin',
                       'actualdeparturetime_week',
                       'airline','dep_delay_sig']]
#Data Proprocessing
records2 = []
for i in range(0, 11595):
    records2.append([str(no_dep.values[i,j]) for j in range(0, 6)])
    
association_rules4 = apriori(records2, min_support=0.01, min_confidence=0.2, min_lift=3, min_len=5)  
association_results4 = list(association_rules4) 
#print(association_results4)
#print(len(association_results4))


for item in association_results4:
    
   
    pair = item[0] 
    items = [x for x in pair]
    f=open("dep_0.01.txt","a+")
    f.write("Rule: " + str(items[0:len(items)-1]) + " -> " + str(items[len(items)-1])+'\n')
    f.write("Support: " + str(item[1])+'\n')
    f.write("Confidence: " + str(item[2][0][2])+'\n')
    f.write("Lift: " + str(item[2][0][3])+'\n')
    f.write("=====================================\n")
f.close()

association_rules5 = apriori(records2, min_support=0.05, min_confidence=0.2, min_lift=3, min_len=5)  
association_results5 = list(association_rules5) 
print(association_results5[0])
#print(len(association_results5))
#print(len(association_results5[0]))

for item in association_results5:
    
   
    pair = item[0] 
    items = [x for x in pair]
    f=open("dep_0.05.txt","a+")
    f.write("Rule: " + str(items[0:len(items)-1]) + " -> " + str(items[len(items)-1])+'\n')
    f.write("Support: " + str(item[1])+'\n')
    f.write("Confidence: " + str(item[2][0][2])+'\n')
    f.write("Lift: " + str(item[2][0][3])+'\n')
    f.write("=====================================\n")
    
f.close()

association_rules6 = apriori(records2, min_support=0.005, min_confidence=0.2, min_lift=3, min_len=5)  
association_results6 = list(association_rules6) 
#print(association_results6[0])
#print(len(association_results6))
#print(len(association_results6[0]))

for item in association_results6:
    
   
    pair = item[0] 
    items = [x for x in pair]
    f=open("dep_0.005.txt","a+")
    f.write("Rule: " + str(items[0:len(items)-1]) + " -> " + str(items[len(items)-1])+'\n')
    f.write("Support: " + str(item[1])+'\n')
    f.write("Confidence: " + str(item[2][0][2])+'\n')
    f.write("Lift: " + str(item[2][0][3])+'\n')
    f.write("=====================================\n")
    
f.close()
'''
===============================================================================================
'''