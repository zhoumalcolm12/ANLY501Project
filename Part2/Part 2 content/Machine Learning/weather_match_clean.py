# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 21:58:18 2018

@author: Xi
"""
# Imported packages and libraries
import csv
import math
import numpy as np
import pandas as pd

# Define main function here
def main(argv):
    
    # Open three .csv data files from last part of project
    with open ('data_1.csv', 'r') as d1:
        df_1 = pd.read_csv(d1 , sep=',', encoding='utf-8')
    with open ('data_1.csv', 'r') as d2:
        df_2 = pd.read_csv(d2 , sep=',', encoding='utf-8')
    with open ('data_1.csv', 'r') as d3:
        df_3 = pd.read_csv(d3 , sep=',', encoding='utf-8')
    
    # Open weather data collected from project part 1
    with open ('weather.csv', 'r') as weather:
        df_weather = pd.read_csv(weather , sep=',', encoding='utf-8') 
    
    # Concatenate three three dataframes with flight data
    df_flight = pd.concat([df_1, df_2, df_3])
    df_flight = df_flight[0:10]
    
    # Delete some redundant attributes and check the shape
    df_flight = del_redundant(df_flight)
    print(df_flight.shape)
    
    # Get aircraft manufacturer name from ICAO code using a csv file
    aircraft_dict = dict()
    with open('icao_manufacturer.csv','r') as manuf_input:
        icao_manuf = list(csv.reader(manuf_input))
        
    
    # Store the aircraft manufacturere name and ICAO code data into a dictionary
    for row in icao_manuf:
        icao_info = row[1].split(' ') 
        
        if icao_info[0] == 'Hawker' or icao_info[0] == 'Robin' or icao_info[0] == 'British' \
        or icao_info[0] == 'Fairchild' or icao_info[0] == 'De' or icao_info[0] == 'McDonnell':
            manuf = ' '.join(icao_info[0:2])
        elif icao_info[0] == 'Canadair' or icao_info[0] == 'Government':
            manuf = ' '.join(icao_info[0:3])
        else:
            manuf = icao_info[0]
    
        aircraft_dict[row[0]] = manuf
        
    # Obtaining aircraft manufacturer name from flight ident number using a function and airline
    #   name dictionary.
    df_flight['aircraft_manuf'] = df_flight.apply(lambda x: \
      fun_airline_name(x['aircrafttype'], aircraft_dict),axis=1)
    
    
    # Combine the weather data and flight data using defined concat_weather function
    df_flight['t_diff'], df_flight['cloud_altitude'], \
    df_flight['temp'], df_flight['dewpoint'], df_flight['visibility'],\
    df_flight['wind_speed'], df_flight['gust_speed']\
    = zip(*df_flight.apply(lambda row: \
      concat_weather(row['origin'], row['actualdeparturetime'], df_weather),axis=1))
    
    #print("Valid: ", sum(df_flight['t_diff'].isna() ==False))
    
    # Write the concantenated dataframe
    with open('cleaned_data.csv','w') as output:
        df_flight.to_csv(output, sep=',',index = True)
    
# This function uses the aircraft manufacturer dataframe to return aircraft
#   manufacturer dataframe.
def fun_airline_name(icao, a_dict):
    
    try:
        return a_dict[icao]
    except:
        return "unkown"

# This function is used to match weather data and flight data
def concat_weather(dp_icao, dp_time, df_w):
    # Firstlly, determine weather data based on airport
    index_list = df_w.index[df_w['airport'] == dp_icao].tolist()
    
    # Start empty lists to collect possible matched weather data
    t_diff_list = []
    cloud_alt_list = []
    temp_list = []
    dew_list = []
    vis_list = []
    wind_list = []
    gust_list = []
    
    # Search from the weather dataframe to match weather data based on time
    for ind in index_list:
        t_diff = dp_time - df_w.iloc[ind]['time']

        if t_diff < 0:
            t_diff = float('nan')
        
        # Append the possible corresponding weather data
        t_diff_list.append(t_diff)
        cloud_alt_list.append(df_w.iloc[ind]['cloud_altitude'])
        temp_list.append(df_w.iloc[ind]['temp_air'])
        dew_list.append(df_w.iloc[ind]['temp_dewpoint'])
        vis_list.append(df_w.iloc[ind]['visibility'])
        wind_list.append(df_w.iloc[ind]['wind_speed'])
        gust_list.append(df_w.iloc[ind]['wind_speed_gust'])
    
    #print(t_diff_list)
              
    # Show the code is runnning
    print('go')
    nan = np.repeat(float('nan'), 7)
    
    # Some weather data are not avialable so we use try/except structure
    try:
        # Deterine the most related weather data and get the values
        min_loc = t_diff_list.index(min(t_diff_list))
        cld = cloud_alt_list[min_loc]
        temp = temp_list[min_loc]
        dew = temp_list[min_loc]
        vis = vis_list[min_loc]
        wind = wind_list[min_loc]
        gust = gust_list[min_loc]
        
        
        #print(t_diff_list)
        # Return the matched weather data or if not weather is matched, return nan s.
        if math.isnan(min(t_diff_list)) == True:
            return nan.tolist()
        else:
            return min(t_diff_list), cld, temp, dew, vis, wind, gust
    except:
        return nan.tolist()

# This function is used to delete redundant or unneeded attributes
def del_redundant(df):
    del df['Unnamed: 0']
    del df['diverted']
    
    del df['filed_time_year']
    del df['filed_time_month']
    del df['filed_time_week']
    del df['filed_time_day']
    del df['filed_time_time']
    
    del df['filed_departuretime_year']
    del df['filed_departuretime_month']
    del df['filed_departuretime_day']
    
    del df['actualdeparturetime_year']
    del df['actualdeparturetime_month']
    del df['actualdeparturetime_day']
    
    del df['estimatedarrivaltime_year']
    del df['estimatedarrivaltime_month']
    del df['estimatedarrivaltime_day']
    
    del df['actualarrivaltime_year']
    del df['actualarrivaltime_month']
    del df['actualarrivaltime_day']
    
    return df
    
    
# Let the main function run
if __name__=="__main__":
    # Execute only if run as script
    main(sys.argv)