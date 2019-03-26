There are 8 files in total in this machine learning part folder:
1. data_1.csv
	Input data for weather_match_clean.py
	This is part of collected and cleaned data frome Project part 1 
	and served as input file for this part.

2. data_2.csv
	Input data for weather_match_clean.py
	This is part of collected and cleaned data frome Project part 1 
	and served as input file for this part.

3. data_3.csv
	Input data for weather_match_clean.py
	This is part of collected and cleaned data frome Project part 1 
	and served as input file for this part.

4. weather.csv
	Input data for weather_match_clean.py
	This is part of collected weather data frome Project part 1 
	and served as input file for this part.

5. weather_match_clean.py
	This pythos file input data_1.csv, data_2.csv, data_3.csv, and weather.csv
	It concatenate three flight data files and match corresponding weather data.
	Further clean for missing and incorrect values are performed in this file.
	It output: cleaned_data.csv

6. cleaned_data.csv
	Output data from weather_match_clean.py
	Cleaned data for Machine Learning

7. machine_learning_binaryclass.py
	This python file reads in cleaned_data.csv
	It creats binary delay type data. Using different classifiers to predict
		departure or arrival delay. There is an option to choose whether
		to include weather data or not.
	Logistic regression, and Recursive Feature Elemination is implmented.
	k-folds cross-validation during traning, accuracy socre, confusion matrices,
		testing report, and ROC curves are generated.

8. machine_learning_multiclass.py
	This python file reads in cleaned_data.csv
	It creats multi-type delay level data. Using different classifiers to predict
		departure or arrival delay level. There is an option to choose whether
		to include weather data or not.
	k-folds cross-validation during traning, accuracy socre, confusion matrices,
		testing report, and multi-type class ROC curves are generated.