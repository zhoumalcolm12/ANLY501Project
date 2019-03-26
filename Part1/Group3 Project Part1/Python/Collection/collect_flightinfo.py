import requests
import pandas as pd
import datetime
import os
##################Second API query(using 'FlightInfo')
apikey = '7038ce657e42de52eaf5af4126a5fbbf0f4f0a43'
username = 'yangxi1994'
url_base = ''.join(['http://', username,':',
                    apikey,'@flightxml.flightaware.com/json/FlightXML2/FlightInfo'])

airline=pd.read_csv('Departure_Output/noondeparture.csv')
airline=list(airline['ident'])

url_post = {'ident': airline[0],
                'howmany':'15',
                }
response=requests.get(url_base, url_post)
jsontxt = response.json()
df=pd.DataFrame(jsontxt['FlightInfoResult']['flights'][0],index=[airline[0]])
for i in range(1,len(jsontxt['FlightInfoResult']['flights'])):
        df2=pd.DataFrame(jsontxt['FlightInfoResult']['flights'][i],index=[airline[0]])
        df=df.append(df2)


for item in airline[1:(len(airline)-1)]:
    url_post = {'ident': item,
                'howmany':'15',
                }
    
    # Use the get function of requests library to get data.
    response=requests.get(url_base, url_post)
    
    # The result is convereted to json format
    jsontxt = response.json()
    for i in range(0,len(jsontxt['FlightInfoResult']['flights'])):
        df2=pd.DataFrame(jsontxt['FlightInfoResult']['flights'][i],index=[item])
        df=df.append(df2)