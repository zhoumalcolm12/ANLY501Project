# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 18:48:18 2018

@author: Jimny
"""

# Import packages and classes to use
import warnings
warnings.filterwarnings("ignore")

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from operator import itemgetter 

# Currently using sklearn v 0.20
from sklearn.multiclass import OneVsRestClassifier
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
import sklearn.tree as tree

from sklearn.metrics import roc_curve, auc
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

# Define main function here
def main(argv):
    
    # Open cleaned file and read to a dataframe
    with open ('cleaned_data.csv', 'r') as raw:
        df = pd.read_csv(raw , sep=',', encoding='utf-8')
        
    # Input wheter to include weather data
    weather_sig = input('Do you want to include weather attributes : y or n?  ')
    
    # Decisioin structure to determine which attributes to include for machine learning
    if weather_sig == 'y':
        
        df = df[df['t_diff'].isna() == False]
        
        attr_list = ['aircrafttype', 'filed_airspeed_kts', 'filed_altitude', 
                     'origin', 'destination', 'filed_departuretime_week',
                     'filed_departuretime_hr', 'estimatedarrivaltime_hr', 'airline',
                     'aircraft_manuf', 
                     'cloud_altitude','temp','dewpoint','visibility',
                     'wind_speed','gust_speed']
        change_list = ['aircrafttype', 'origin', 'destination',
                       'filed_departuretime_week','airline','aircraft_manuf']
        
        
    elif weather_sig == 'n':
        
        attr_list = ['aircrafttype', 'filed_airspeed_kts', 'filed_altitude', 
                     'origin', 'destination', 'filed_departuretime_week',
                     'filed_departuretime_hr', 'estimatedarrivaltime_hr', 'airline',
                     'aircraft_manuf']
        change_list = ['aircrafttype', 'origin', 'destination',
                       'filed_departuretime_week','airline','aircraft_manuf']
        
        
    # Get the departure and arrival hour from time in hr:mm:ss format
    df['filed_departuretime_hr'] = df.apply(lambda x: \
      get_hr(x['filed_departuretime_time']),axis=1)
    
    df['estimatedarrivaltime_hr'] = df.apply(lambda x: \
      get_hr(x['estimatedarrivaltime_time']),axis=1)
    
    # Further cleaning before machine learning
    df = df[df['aircrafttype'].isna() == False]
    #df_ml = df_ml[df_ml['diff_flt'].isna() == False]
    df = df[df['filed_airspeed_kts'] != 0]
    
    # From the attribute list, prepare the dataframe for machine learning
    df_ml = df.loc[:,attr_list]

    # Using the type_to_num function to change all attributes with types to numbers
    df_ml = type_to_num(change_list, df_ml)
    
    # Chosse to predict departure delay or arrival delay
    delay_type = input('Departure Delay please type d or Arrival delay please type a: ')
    
    # Decision structure to choose which kind of delay to predict
    if delay_type == 'd':
        df_ml['delay'] = df.apply(lambda x: \
             int(x['dep_delay_sig']),axis=1)
    elif delay_type == 'a':
        df_ml['delay'] = df.apply(lambda x: \
             int(x['arr_delay_sig']),axis=1)

    # Save the dataframe for machine learning before normalization to check
    #with open('df_ml_wo_weather.csv','w') as output:
    #    df_ml.to_csv(output, sep=',',index = True)
    
    # Histogram
    df_ml.hist(figsize=(15,15))
    plt.tight_layout()
    plt.show()
    
    ######## Machine Learning ############
    # Change the dataframe into a numpy array
    data_arr = df_ml.values #returns a numpy array
    
    # The values of different columns are not in same scale
    # So we choose to normalize it by change every value into [0,1] scale
    #   according to it's original value
    min_max_scaler = preprocessing.MinMaxScaler()
    norm_arr = min_max_scaler.fit_transform(data_arr)
    
    # Call the machine learning function to run machine learning
    machine_learning(norm_arr, list(range(len(df_ml.columns))))
    
    
    # feature extraction by Logistic Regression
    print('\n\n####### Feature Selection ###########:')
    X = norm_arr[:, 0 : (len(df_ml.columns)-1)]
    Y = norm_arr[:, (len(df_ml.columns)-1)]
    
    # Using RFE to rank attributes
    log_reg = LogisticRegression()
    rfe = RFE(log_reg, 6)
    fit = rfe.fit(X, Y)
    
    # Selecte most relevant attributes for machien learning
    fit_list = fit.support_.tolist()
    indexes = [index for index in range(len(fit_list)) if fit_list[index] == True]
    
    # Print out attributes selected and ranking
    print('\nAttributes selected are: ', itemgetter(*indexes)(attr_list))
    print('\nAttributes Ranking: ', fit.ranking_)
    
    indexes.append(len(df_ml.columns)-1)
    
    # Call machine learning function after feature selection
    machine_learning(norm_arr, indexes)
    
    if weather_sig == 'n':
        # Logistic Regression report
        logit_model=sm.Logit(Y,X)
        result=logit_model.fit()
        print(result.summary2())
    
# This function is used to get obly hr from time attributes in hr:mm:ss format
def get_hr(time):
    hr = int(time[0:2])
    return hr

# This function takes in categorical attributes and convert them into numbers
def type_to_num(name_list, df):
    for name in name_list:
        a_type = df[name].value_counts().keys().tolist()
        a_replace = list(range(len(a_type)))
    
        df[name].replace(a_type, a_replace, inplace = True)
    
    return df

# Machine Learning function is defined here
def machine_learning(array, attr):
    
    # Prepare attributes array and class array
    X = array[:, attr[0 : len(attr)-1]]
    Y = array[:, attr[-1]]
    
    # Define test size and random seed
    test_size = 0.30
    seed = 50
    
    # Splitting training and validation dataset
    X_train, X_validate, Y_train, Y_validate = \
    model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
    
    # Setup 10-fold cross validation to estimate the accuracy of different models
    # Split data into 10 parts
    # Test options and evaluation metric
    num_folds = 10
    #num_instances = len(X_train)
    seed = 10
    scoring = 'accuracy'
    
    # Add each algorithm and its name to the model array
    models = []
    
    models.append(('Decision Tree', tree.DecisionTreeClassifier()))
    models.append(('Random Forest', RandomForestClassifier()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('Gaussian Naive Bayes', GaussianNB()))
    models.append(('SVM', SVC()))
    models.append(('Gradient Boosting Classifier', GradientBoostingClassifier()))

    # Evaluate each model, add results to a results array,
    # Print the accuracy results (remember these are averages and std
    results = []
    names = []
    print("\nAccuracy during training:\n")
    for name, model in models:
        kfold = model_selection.KFold( n_splits=num_folds, 
                                      random_state=seed)
        cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
        
    
    print("\n###### Prediction and ROC Curve: ######\n")
    # Make predictions on validation dataset
    plt.rcParams.update({'font.size': 15})
    plt.figure(figsize=(10,7))
    
    # Prediction and ROC curve
    for name, model in models:
        model.fit(X_train, Y_train)
        prediction = model.predict(X_validate)
        
        print("\nTest results using ", name, " is :\n")
        print_validate(Y_validate, prediction)
        
        if name == 'Gradient Boosting Classifier' or name == 'SVM':
            draw_roc(X, Y, OneVsRestClassifier(model), name)
        else:
            draw_roc(X, Y, OneVsRestClassifier(model), name, proba = True)
    

    plt.plot([0, 1], [0, 1], 'k--', lw = 4)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for binary class')
    plt.legend(loc="lower right")
    plt.show()

# This function is used to draw ROC curves
def draw_roc(X, Y, classifier, clf_name, proba = False):
    #print(roc_auc_score(Y_validate, predictions))
    random_state = np.random.RandomState(0)

    # shuffle and split training and test sets
    X_train, X_test, Y_train, Y_test = \
    model_selection.train_test_split(X, Y, test_size=.5,random_state=random_state)
    
    # Learn to predict each class against the other
    #   Some classifiers does not have decision_function so we have to use
    #   predict_proba instead
    if proba == True:
        Y_score = classifier.fit(X_train, Y_train).predict_proba(X_test)
    else:
        Y_score = classifier.fit(X_train, Y_train).decision_function(X_test)
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    
    if proba == True:
        fpr, tpr, _ = roc_curve(Y_test, Y_score[:,1])
    else:
        fpr, tpr, _ = roc_curve(Y_test, Y_score)
    
    roc_auc = auc(fpr, tpr)
    
    #color = itertools.cycle(['aqua', 'darkorange', 'cornflowerblue', 
    #                          'darkred', 'blue', 'darkgreen'])
    plt.plot(fpr, tpr,  lw=4,
                 label='ROC curve of {0} (area = {1:0.2f})'
                 ''.format(clf_name, roc_auc))
    

    ################################

# This function is used to print validation/test resuls
def print_validate(Y_val, predictions):
    print("\nAccuracy score: ",accuracy_score(Y_val, predictions))
    print("\nConfusion Matrix: \n",confusion_matrix(Y_val, predictions))
    print(classification_report(Y_val, predictions))
    
# Let the main function run
if __name__=="__main__":
    # Execute only if run as script
    main(sys.argv)