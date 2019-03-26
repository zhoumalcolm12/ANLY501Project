###########PROJECT PART2
###hypothsesis testing

#preparing for packages
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint
from collections import Counter
import scipy.stats as stats

##load the data
mydata=pd.read_csv('/Users/zmt/Desktop/MachineLearning/cleandata.csv')
##Find 10 most popular airlines
result = Counter(mydata.airline)
result.pop('unkown')
result=sorted(result.items(),key=lambda item:item[1],reverse=True)
airline=[]
for i in range(5):
    airline.append(list(result[i])[0])
print('The 5 most popular airlines are', airline)

#testing the delayrate for 10 popular airlines
delayrate=[]
line=[]
for line in airline:
    air=mydata['dep_delay_sig'][mydata.airline == line]
    delayrate.append(np.mean(air))
plt.bar(range(len(delayrate)), delayrate,color='rgb',tick_label=airline)
plt.xticks(rotation='vertical')
plt.show()
#ANOVA test
d_data = {line:mydata['dep_delay_sig'][mydata.airline == line] for line in airline}
F,p=stats.f_oneway(d_data[airline[0]],d_data[airline[1]],d_data[airline[2]],d_data[airline[3]],d_data[airline[4]])
print('The F statistics is',F,'\n', 'The p-value is',p)


#testing the delayrate for Boeing Aircraft and Airbus Aircraft
#find 4 most popular aircraft manufacturers
result = Counter(mydata.aircraft_manuf)
result.pop('unkown')
result=sorted(result.items(),key=lambda item:item[1],reverse=True)
manu=[]
for i in range(4):
    manu.append(list(result[i])[0])

mydata=mydata.dropna(subset=['aircraft_manuf','arr_delay_sig'])
indA=[v for v in range(len(mydata)) if mydata.iloc[v]['aircraft_manuf']=='Airbus']
indB=[v for v in range(len(mydata)) if mydata.iloc[v]['aircraft_manuf']=='Boeing']
indE=[v for v in range(len(mydata)) if mydata.iloc[v]['aircraft_manuf']=='Embraer']
indC=[v for v in range(len(mydata)) if mydata.iloc[v]['aircraft_manuf']=='Canadair Regional Jet']
Airbus=mydata.iloc[indA]['arr_delay_sig']
Boeing=mydata.iloc[indB]['arr_delay_sig']
Embraer=mydata.iloc[indE]['arr_delay_sig']
CRJ=mydata.iloc[indC]['arr_delay_sig']
A_delay,B_delay,E_delay,C_delay=np.mean(Airbus),np.mean(Boeing),np.mean(Embraer),np.mean(CRJ)
plt.bar(range(4), [A_delay,B_delay,E_delay,C_delay],color='rgb',tick_label=['Airbus','Boeing','Embraer','Canadair Regional Jet'])
plt.show()
F,p=stats.f_oneway(Airbus,Boeing,Embraer,CRJ)
print('The F statistics is',F,'\n', 'The p-value is',p)






