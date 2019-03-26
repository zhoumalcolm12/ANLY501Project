# -*- coding: utf-8 -*-
"""
Name: Boyang Wei
Data Analysis

"""
import datetime
from datetime import datetime as dt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats.stats import pearsonr

df = pd.read_csv('output.csv', encoding = "ISO-8859-1")
df.drop(columns = 'index', inplace = True)

FMT = '%H:%M:%S'

#Clean up the incorrect values (Flight duration is 0 min)
df = df.drop(df[df['actualarrivaltime_time'] == df['actualdeparturetime_time']].index)
df = df.reset_index(drop=True)

#Create column of Flight Duration in datetime and minutes
durationList = []
durationMin = []

for i in range(len(df)):
    time = (dt.strptime(df['actualarrivaltime_time'][i],FMT)
    - dt.strptime(df['actualdeparturetime_time'][i],FMT)).days
    if (time != 0):
        durationList.append(dt.strptime(df['actualarrivaltime_time'][i],FMT)
        - dt.strptime(df['actualdeparturetime_time'][i],FMT) + datetime.timedelta(days=1))
        durationMin.append((dt.strptime(df['actualarrivaltime_time'][i],FMT)
        - dt.strptime(df['actualdeparturetime_time'][i],FMT) + datetime.timedelta(days=1)).seconds/60)
    else:
        durationList.append(dt.strptime(df['actualarrivaltime_time'][i],FMT)
    - dt.strptime(df['actualdeparturetime_time'][i],FMT))
        durationMin.append((dt.strptime(df['actualarrivaltime_time'][i],FMT)
    - dt.strptime(df['actualdeparturetime_time'][i],FMT)).seconds/60)
    
df['Duration'] = durationList
df['DurationMin'] = durationMin


#Create column of Delay/Not-Delay
delayList = [] 
for i in range(len(df)):
    if (dt.strptime(df['actualarrivaltime_time'][i],FMT) - dt.strptime(df['estimatedarrivaltime_time'][i],FMT) > datetime.timedelta(minutes = 0)):
        delayList.append(1)
    else:
        delayList.append(0)
        
df['Class'] = delayList

#Create two other df_2 and df_3 containing numeric and categorical data
colName_2 = [
 'cloud_altitude',
 'temp',
 'dewpoint',
 'visibility',
 'wind_speed',
 'gust_speed',
 'DurationMin']
df_2 = df[colName_2]

colName_3 = [
 'ident',
 'aircrafttype',
 'originCity',
 'destinationCity',
 'airline',
 'aircraft_manuf',
 'Class']
df_3 = df[colName_3]

colName_4 = [
 'temp',
 'wind_speed',
 'gust_speed']
df_4 = df[colName_4]

df_4 = df_4.dropna()
df_2 = df_2.dropna()

#Mean/SD/Median/Mode
meanList = df_2.mean()
medianList = df_2.median()
stdList = df_2.std()
df_num = pd.DataFrame({'mean':meanList, 'median':medianList, 'std':stdList})
print("Statistical Summary of Numerical Attributes: (Mean/Median/Standard Deviation)")
print(df_num)

print("----------------------------------------------------------------------")
modeList = df_3.mode()
modeList = modeList.iloc[0]
print("Statistical Summary of Categorical Attributes: (Mode)")
print(modeList)
print("----------------------------------------------------------------------")


#Outliers
print("Number of outliers in each numerical feature:")
def outliers_iqr(ys):
    quartile_1, quartile_3 = np.percentile(ys, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((ys > upper_bound) | (ys < lower_bound))

for name in list(df_2):
    print('Number of outliers in ' + name +':')
    print(outliers_iqr(df_2[name])[0].size)
print("----------------------------------------------------------------------")

#Missing Values
print('Missing Values for each column:')
df.isna().sum()

df_missing = pd.DataFrame(
    {'Features': df.columns[df.isna().any()].tolist(),
     'Missing Counts':df[df.columns[df.isna().any()].tolist()].isna().sum()
    })
sns.barplot(x="Features", y="Missing Counts", data=df_missing)
plt.title("Count of Missing Values")


####Histogram################################################################## 
#Flight Duration Distribution
#sns.set(rc={'figure.figsize':(15,12)})
sns.distplot(df['DurationMin'], axlabel='Flight Duration (min)').set_title('Flight Duration Distribution in Minutes')

sns.distplot(df_2['visibility'], axlabel='Visibility').set_title('Visility Distribution in levels')

sns.distplot(df_2['gust_speed'], axlabel='Gust Speed').set_title('Guest Speed Distribution')


#Binning Flight Duration
names = ["0 ~ 2 hours", "2 ~ 4 hours", "4 ~ 6 hours", 
         "6 ~ 8 hours", "8 hours above"]
names_1 = range(1,6)
bins1=[0, 120, 240, 360, 480, 1000]
df['Duration Group Catgories'] = pd.cut(df['DurationMin'], bins1, labels=names)
df['Duration Group'] = pd.cut(df['DurationMin'], bins1, labels=names_1)
#Binning Count Plots
sns.countplot(x="Class", hue="Duration Group Catgories",data=df)
plt.title('Count of Flight Duration Time Bins of Non-Dealy vs Dealy')


#Boxplot#######################################################################
sns.boxplot(data=df['DurationMin'])
plt.title('Boxplot of Flight Duration')


#Correlation Matrix for Attributes except for 'Time' and 'Class'
corr = df_2.corr()

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.title("Weather Factors Heat Map")


#Correlation Scatter Plots
g = sns.pairplot(df_4)
g.fig.suptitle("Correlation Plot of WindSpeed/GustSpeed/Temperature")

def findCorrelation():
    colNames = list(df_4)
    for name1 in colNames:
        for name2 in colNames:
            rCor = abs(pearsonr(df_4[name1],df_4[name2])[0])
            print("Correlation r between", name1, "and", name2, ":", rCor)
    print("-----------------------------------------------------------------")

findCorrelation()