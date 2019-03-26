import requests
import pandas as pd
import datetime
import os
##################First API query(using 'Departed')
##define username and apikey
username = 'yangxi1994'
apikey = '7038ce657e42de52eaf5af4126a5fbbf0f4f0a43'
url_base = ''.join(['http://',username,':',
                    apikey,'@flightxml.flightaware.com/json/FlightXML2/Departed'])

##get the airport name from file
airport=pd.read_csv('airports.csv',header=None)
airport=list(airport[1])
url_post = {'airport': airport[0],
                'howmany':'15',
                }
response=requests.get(url_base, url_post)
jsontxt = response.json()

##convert jsontxt into pandas dataframe
df=pd.DataFrame(jsontxt['DepartedResult']['departures'][0],index=[airport[0]])
for i in range(1,len(jsontxt['DepartedResult']['departures'])):
        df2=pd.DataFrame(jsontxt['DepartedResult']['departures'][i],index=[airport[0]])
        df=df.append(df2)

for item in airport[1:29]:
    url_post = {'airport': item,
                'howmany':'15',
                }
    
    # Use the get function of requests library to get data.
    response=requests.get(url_base, url_post)
    
    # The result is convereted to json format
    jsontxt = response.json()
    for i in range(0,len(jsontxt['DepartedResult']['departures'])):
        df2=pd.DataFrame(jsontxt['DepartedResult']['departures'][i],index=[item])
        df=df.append(df2)        


df.to_csv('Departure_Output/noondeparture.csv')








