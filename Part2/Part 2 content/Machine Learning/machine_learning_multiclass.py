# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 23:58:46 2018

@author: Jimny
"""

# Import packages and classes to use
import warnings
warnings.filterwarnings("ignore")

import sys
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interp

from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import sklearn.tree as tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

from sklearn.metrics import roc_curve, auc

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
                     'origin', 'destination',
                     'filed_departuretime_hr', 'estimatedarrivaltime_hr', 'airline',
                     'aircraft_manuf',
                     'cloud_altitude','temp','dewpoint','visibility',
                     'wind_speed','gust_speed']
        change_list = ['aircrafttype', 'origin', 'destination',
                       'airline','aircraft_manuf']
        
        dep_delay_bin = [-30, 0, 10, 30, 400]
        arr_delay_bin = [-250, -10, -5, 0.1, 10]
        
    elif weather_sig == 'n':
        
        attr_list = ['aircrafttype', 'filed_airspeed_kts', 'filed_altitude', 
                     'origin', 'destination', 'filed_departuretime_week',
                     'filed_departuretime_hr', 'estimatedarrivaltime_hr', 'airline',
                     'aircraft_manuf']
        change_list = ['aircrafttype', 'origin', 'destination',
                       'filed_departuretime_week','airline','aircraft_manuf']
        
        dep_delay_bin = [-3800, 0, 10, 30, 1000]
        arr_delay_bin = [-850, 0, 10, 60, 500]
    
     # Get the departure and arrival hour from time in hr:mm:ss format
    df['filed_departuretime_hr'] = df.apply(lambda x: \
      get_hr(x['filed_departuretime_time']),axis=1)
    
    df['estimatedarrivaltime_hr'] = df.apply(lambda x: \
      get_hr(x['estimatedarrivaltime_time']),axis=1)
    
    # Further cleaning before machine learning
    df = df[df['aircrafttype'].isna() == False]
    #df_ml = df_ml[df_ml['diff_flt'].isna() == False]
    df = df[df['filed_airspeed_kts'] != 0]
    
    #'dep_delay_min','diff_flt',
    # From the attribute list, prepare the dataframe for machine learning  
    df_ml = df.loc[:,attr_list]
    
    # Using the type_to_num function to change all attributes with types to numbers
    df_ml = type_to_num(change_list, df_ml)
    
    # Binning delay level
    #print('Min = ', min(df['dep_delay_min']))
    #print('Max = ', max(df['dep_delay_min']))
    
    # Chosse to predict departure delay or arrival delay
    delay_type = input('Departure Delay please type d or Arrival delay please type a: ')
    
    # Decision structure to choose which kind of delay to predict
    if delay_type == 'd':
        # Bin the departure delay level
        delay_level = pd.cut(df['dep_delay_min'], 
                         dep_delay_bin, labels = range(len(dep_delay_bin)-1))
    elif delay_type == 'a':
        # Bin the arrival delay level
        delay_level = pd.cut(df['arr_delay_min'], 
                         arr_delay_bin, labels = range(len(arr_delay_bin)-1))
    delay_level = delay_level.tolist()

    
    # Save dataframe used for machine learning into a csv file
    #with open('df_ml.csv','w') as output:
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
    
    # Apeend the categorical delay level to normalized array for machine learning
    norm_arr = np.c_[norm_arr, delay_level]
    
    # Make a list containing columns index of the dataframe
    attr = list(range(len(df_ml.columns)))
    attr.append(len(df_ml.columns))
    
    # Call the machine learning function
    machine_learning(norm_arr, attr, attr_list)
    
    
# This function is used to get obly hr from time attributes in hr:mm:ss format 
def get_hr(time):
    hr = int(time[0:2])
    return hr

# This function takes in categorical attributes and convert them into numbers
def type_to_num(name_list, df):
    for name in name_list:
        a_type = df[name].value_counts().keys().tolist()
        a_replace = list(range(len(a_type)))
    
        #print(df[name].value_counts())
    
        df[name].replace(a_type, a_replace, inplace = True)
    
    return df

# Machine Learning function is defined here
def machine_learning(array, attr , attr_names):
    
    # Prepare attributes array and class array
    X = array[:, attr[0 : len(attr)-1]]
    Y = array[:, attr[-1]]
    
    
    # Define test size and random seed
    test_size = 0.20
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
    print("Accuracy during training")
    for name, model in models:
        kfold = model_selection.KFold( n_splits=num_folds, 
                                      random_state=seed)
        cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
        
        
    # Make predictions on validation dataset
    # Prediction and ROC curve
    for name, model in models:
        model.fit(X_train, Y_train)
        prediction = model.predict(X_validate)
        
        print("\nTest results using ", name, " is :\n")
        print_validate(Y_validate, prediction)
        
        if name == 'Gradient Boosting Classifier' or name == 'SVM':
            draw_roc(X, Y, OneVsRestClassifier(model))
        else:
            draw_roc(X, Y, OneVsRestClassifier(model), proba = True)
    
# This function is used to draw ROC curves
def draw_roc(X, Y, classifier, proba = False):
    #print(roc_auc_score(Y_validate, predictions))
    
    # Make binarized multi classes labels
    Y = label_binarize(Y, classes =  [3,2,1,0])
    n_classes = Y.shape[1]

    # Define Random state seed
    random_state = np.random.RandomState(0)
    
    # Get the shape of attributes
    n_samples, n_features = X.shape
    #X = np.c_[X, random_state.randn(n_samples,  n_features)]
    
    # shuffle and split training and test sets
    X_train, X_test, Y_train, Y_test = \
    model_selection.train_test_split(X, Y, test_size=.5,random_state=random_state)
    
    # Learn to predict each class against the other
    if proba == True:
        Y_score = classifier.fit(X_train, Y_train).predict_proba(X_test)
    else:
        Y_score = classifier.fit(X_train, Y_train).decision_function(X_test)
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], Y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(Y_test.ravel(), Y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    
    lw = 4
    plt.rcParams.update({'font.size': 15})
    plt.figure(figsize=(10,7))
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
                   color='deeppink', linestyle=':', linewidth=lw)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
                   color='navy', linestyle=':', linewidth=lw)
    
    colors = itertools.cycle(['aqua', 'darkorange', 'cornflowerblue', 'darkred'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()

# This function is used to print validation/test resuls
def print_validate(Y_val, predictions):
    print("\nAccuracy score: ",accuracy_score(Y_val, predictions))
    print("\nConfusion Matrix: \n",confusion_matrix(Y_val, predictions))
    print(classification_report(Y_val, predictions))
    
# Let the main function run
if __name__=="__main__":
    # Execute only if run as script
    main(sys.argv)