# -*- coding: utf-8 -*-
"""
Created on Fri May 20 09:30:26 2022
Objective: To find the prediction of next-day new case
@author: Nur Izyan Kamarudin
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.utils import plot_model
from modules import ExploratoryDataAnalysis,DataVisualization,DataPreprocessing
from modules import ModelCreation,ModelEvaluation

#%% Paths
LOG_PATH = os.path.join(os.getcwd(),'log')
DATASET_TRAIN_PATH = os.path.join(os.getcwd(), 'datasets', 
                                  'cases_malaysia_train.csv')
DATASET_TEST_PATH = os.path.join(os.getcwd(), 'datasets', 
                                 'cases_malaysia_test.csv')
MMS_PATH = os.path.join(os.getcwd(), 'saved_model', 'mms_scaler.pkl')
MODEL_SAVE_PATH = os.path.join(os.getcwd(), 'saved_model', 'model.h5')
#%% EDA
# STEP 1: Data Loading
df_train = pd.read_csv(DATASET_TRAIN_PATH)
df_test = pd.read_csv(DATASET_TEST_PATH)
df_train = df_train['cases_new']
df_test = df_test['cases_new']

#%% # STEP 2: Data Interpretation and Data Cleaning

# a) to replace non-numerical value and blank spaces into nan
df_train = pd.to_numeric(df_train, errors='coerce')

# b) replace nan with last observed value using ffill approach
eda = ExploratoryDataAnalysis()
eda.fill_nan(df_train)
eda.fill_nan(df_test)
'''replacing nan with last observed value because we don't have a big sequence 
of nan values, we only a very small number of nan values.'''

graph = DataVisualization()
graph.plot_initial_trend(df_train)
''' Observation: the graph_1.png shows there is a dramatic increment of new 
cases, and a downward trend at the end of the data.'''

#%% STEP 3: Data Preprocessing

# a) scale the data using min max scaler
dp = DataPreprocessing()
mms, train_df_scaled, test_df_scaled = dp.min_max_scaler(df_train,df_test,
                                                         MMS_PATH)

#%% STEP 4: Model Creation

window_size = 30
len_train = len(df_train)
len_test = window_size + len(df_test)

# a) split TRAINING set
mc = ModelCreation()
x_train, y_train = mc.split_data(train_df_scaled, window_size, len_train)

# b) split TESTING set
dataset_full = np.concatenate((train_df_scaled,test_df_scaled), axis=0)
test_data = dataset_full[-len_test:]
x_test, y_test = mc.split_data(test_data, window_size, len_test)

# c) build LSTM model
model = mc.lstm_model(x_train)
plot_model(model)

# d) train the model
mc.train_model(LOG_PATH, model, x_train, y_train, epochs=50)

#%% STEP 5: Model Evaluation

# a) predict the model on x test
me = ModelEvaluation()
y_true, y_pred = me.model_pred(model, x_test, y_test, mms)

# b) to view the performance of model
graph.plot_performance(y_true,y_pred)

# c) to view the mean percentage absolute error
print('The Mean Percentage Absolute Error',
      (mean_absolute_error(y_true,y_pred)/sum(abs(y_true))) *100,'%')

# d) save the model
model.save(MODEL_SAVE_PATH)

