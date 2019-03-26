ANLY501 Project (Flight Delay Rate Analysis and Prediction)

Python Files (Python):
    Collection Part (Python\Collection):
        collect_departed.py
            This file helps collect the flights information based on the
	    top 30 airports in the U.S. from local csv file called 
            'airports.csv' and saved the output in csv format under
            directory called 'Python\Collection\Departure_Output'
	
	collect_flightinfo.py	
	    This file helps collect the flights information based on the
	    top identations of flights in the U.S. from result csv file
            generated from collection_departed.py in 'Python\Collection\Departure_Output'
            and saved the output in csv format under directory called
           'Python\Collection\FlightInfo_Output'

	weather.py
	    This file helps collect the weather information based on the
	    top 30 airports in the U.S. from Sep30th to October6th from 
            local csv file called 'airports.csv' and saved the output in 
            csv format under directory called 'Python\Collection\Weather_Output'

    Cleaning Part (Python\Cleaning):
	clean.py
	    This file has many functions that clean the results that we get from
	    the python files in collection part and output the result in the 
	    directory called 'Python\Cleaning\Output'. The input files are under
	    'Python\Collection\Weather_Input' which are the csv files generated
	    from collect_flightinfo.py and collect_departed.py

Input File
    airports.csv
	This file contains the top 30 airports in the U.S. and is used to passed
	as parameters in collect_departed.py to generate the raw outputs
	
    airline_ICAO.csv
	This file contains the criteria that helps clean the flight series data 
	into correct formats

Report File
    PartOne Report.docx/pdf
	Report that explains the part one steps and procedures that we  used to
	collect and clean our flight datasets. This report also introduce the 
	background and reasons we chose this topic for investigation