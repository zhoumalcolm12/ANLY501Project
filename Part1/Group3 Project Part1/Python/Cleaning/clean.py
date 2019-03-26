# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 12:44:03 2018

@author: Xi
"""
# Anly 501 Group 01 Project Part 1
# This python file clean the raw flight data we collected

# Import the packages we need.
import numpy as np
import pandas as pd
import csv
import time

from pprint import pprint


def main():
    input_output("Input/data1.csv", "Output/OUTPUT1.csv")
    input_output("Input/data2.csv", "Output/OUTPUT2.csv")
    input_output("Input/data3.csv", "Output/OUTPUT3.csv")

def input_output(inputName, outputName):
    # Open the raw data file here
    with open (inputName, 'r') as raw:
        df = pd.read_csv(raw , sep=',', encoding='latin1')
    
    #df = df[:10]
    
    # Call the function to decode UNIX time and separate the date and time objects
    fun_sep_time(df,'filed_time', decode = True)
    fun_sep_time(df,'filed_departuretime', decode = True)
    fun_sep_time(df,'actualdeparturetime', decode = True)
    fun_sep_time(df,'estimatedarrivaltime', decode = True)
    fun_sep_time(df,'actualarrivaltime', decode = True)
    
    # Print the first five rows for visualization
    pprint(df[:5])
    
    # Get airline name from ICAO code using a csv file
    airline_dict = dict()
    with open('airline_ICAO.csv','r') as icao_input:
        icao_data = list(csv.reader(icao_input))
    
    # Store the airline name and ICAO code data into a dictionary
    for row in icao_data:
        airline_dict[row[0]] = row[1]
    
    # Obtaining airline name from flight ident number using a function and airline
    #   name dictionary.
    df['airline'] = df.apply(lambda y: \
      fun_airline_name(y['ident'], airline_dict),axis=1)
    
    # Print out how many missing actual arrival and departure time in the data set.
    print("We have",sum(df['actualarrivaltime_year']=="1969"),
          "Missing actual arrival time.")
    print("\nWe have",sum(df['actualdeparturetime_year']=="1969"),
          "Missing actual departure time.")
    
    # Delete the rows with missing actual arrival and departure time
    df = df[df['actualarrivaltime_year'] != "1969"]
    df = df[df['actualdeparturetime_year'] != "1969"]
    print("\nWe have ", len(df), " valid rows of data in this dataset.\n")
    
    # Delete the redundant columns that we don;t need
    del df['Unnamed: 0']
    del df['filed_airspeed_mach']
    del df['originName']
    del df['route']
    del df['destinationName']
    
    # Using the cal_arrival_delay and cal_dep_delay to calculate arrival
    #   departure delay for every row in the dataset.
    df['arr_delay_sig'], df['arr_delay_min']  = zip(*df.apply(cal_arrival_delay,axis=1))
    df['dep_delay_sig'], df['dep_delay_min']  = zip(*df.apply(cal_dep_delay,axis=1))
    
    # Print out the basic result of delay
    print("We have ", sum(df['dep_delay_sig']), " delay (departure) flights in this dataset.\n")
    print("We have ", sum(df['dep_delay_sig']<0), " early (departure) flights in this dataset.\n")
    
    print("We have ", sum(df['arr_delay_sig']), " delay (arrival) flights in this dataset.\n")
    print("We have ", sum(df['arr_delay_min']<0), " early (arrival) flights in this dataset.\n")
    
    # Print out the missing planned travel time in the dataset.
    print("We have ", sum(df['filed_ete'].isna()), 
          " missing estimated travel time in this dataset.\n")
    # Use the cal_diff_fl time to calculate difference between planned fly time
    #   and actual fly time.
    df['diff_flt'] = df.apply(cal_diff_fl,axis=1)
    
    # Delete the raw time data which is in UNIX form
    del df['filed_time']
    del df['filed_departuretime']
    del df['actualdeparturetime']
    del df['estimatedarrivaltime']
    del df['actualarrivaltime']
    
    # Open a csv file to write the result
    with open(outputName,'w') as output:
        df.to_csv(output, sep=',',index = True)

# This function separates the date and time objects after decoding and store
#   result into new columns in the data set.
def fun_sep_time(data_frame,col_name, decode = False):
    
    data_frame[''.join([col_name, '_week'])], \
    data_frame[''.join([col_name, '_month'])], \
    data_frame[''.join([col_name, '_day'])], \
    data_frame[''.join([col_name, '_time'])], \
    data_frame[''.join([col_name, '_year'])] \
    = zip(*data_frame[col_name].apply(lambda x: decode_time(x, decode)))

    #del data_frame[col_name]

# This function decode UNIX and and return a list containng separated
#   date and time objects.
def decode_time(uncode, sig):
    
    decode = []
    
    if sig == True:
        uncode = int(uncode)
        if uncode < 0:
            uncode = 0
        time_str = time.ctime(uncode)
    else:
        time_str = uncode
    
    decode.append(time_str[0:3])
    decode.append(time_str[4:7])
    decode.append(time_str[8:10])
    decode.append(time_str[11:19])
    decode.append(time_str[20:])
    
    return decode
        

# This function using the airline name and airline ICAO code dictionary to
#   obtain actual airline names.
def fun_airline_name(ident, a_dict):
    icao = ident[0:3]
    
    try:
        return a_dict[icao]
    except:
        return "unkown"

# This function calculate departure delay based on planned departure time
#   and actual departure time. It returns two variable: one boolean variable
#   that show if the flight is deplayed, and a float variable that store the 
#   delay time.
def cal_dep_delay(data):
    e_sss = data['filed_departuretime']
    a_sss = data['actualdeparturetime']
    
    dep_min = (a_sss - e_sss)/60
    
    if dep_min > 0:
        sig = True
    elif dep_min <=0:
        sig = False
            
    return list([sig, dep_min])

# This function calculate arrival delay based on planned arrival time
#   and actual arrival time. It returns two variable: one boolean variable
#   that show if the flight is deplayed, and a float variable that store the 
#   delay time.
def cal_arrival_delay(data):
    e_sss = data['estimatedarrivaltime']
    a_sss = data['actualarrivaltime']
    
    delay_min = (a_sss - e_sss)/60
    
    if delay_min > 0:
        sig = True
    elif delay_min <=0:
        sig = False
            
    return list([sig, delay_min])

# This function calculate difference between planned fly time and actual fly time
def cal_diff_fl(data):
    filed_ete = data['filed_ete']
    
    act_arr = int(data['actualarrivaltime'])
    act_dep = int(data['actualdeparturetime'])
    
    # Some planned fly time data are missing
    #   So we use try/except to write the difference value as nan
    #   for rows with missing planned fly time.
    try:
        ete_hh, ete_mm, ete_ss = map(int, filed_ete.split(':'))
        ete_min = ete_hh*60 + ete_mm + ete_ss/60
        act_min = (act_arr - act_dep)/60
    
        act_ete_diff = act_min - ete_min
        
    except:
        act_ete_diff = float('nan')
    
    return act_ete_diff

    

main()