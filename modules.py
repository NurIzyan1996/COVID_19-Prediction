# -*- coding: utf-8 -*-
"""
Created on Fri May 20 09:46:56 2022
This file contains functions to deploy on train.py file.
@author: Nur Izyan Kamarudin
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense,Dropout,LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard

class DataVisualization():
    
    def __init__(self):
        pass
    
    def plot_initial_trend(self, data):
        plt.figure()
        plt.plot(data)
        return plt.show()
    
    def plot_performance(self, data_1, data_2):
        
        plt.figure()
        plt.plot(data_1, color='r', label='Actual New Case')
        plt.plot(data_2, color='b', label='Predicted New case')
        plt.legend(['Actual','Predicted'])
        return plt.show()

class ExploratoryDataAnalysis():
    def __init__(self):
        pass
    
    def fill_nan(self, data):
        return data.fillna(method='ffill', inplace=True)


class DataPreprocessing():
    
    def __init__(self):
        pass
    
    def min_max_scaler(self, data_1, data_2, path):
        mms = MinMaxScaler()
        data_1 = mms.fit_transform(np.expand_dims(data_1,-1))
        data_2 = mms.transform(np.expand_dims(data_2,-1))
        pickle.dump(mms, open(path, 'wb'))
        return mms, data_1, data_2
       
class ModelCreation():
    def __init__(self):
        pass

    def split_data(self, data, size_1, size_2):
        data_x = np.array([data[i-size_1:i,0] for i in range(size_1,size_2)])
        data_x = np.expand_dims(data_x,-1)
        data_y = np.array([data[i,0] for i in range(size_1,size_2)])
        return data_x, data_y
    
    def lstm_model(self,data_1):
        model = Sequential()
        model.add(LSTM(64, activation='tanh',
                       return_sequences=True, 
                       input_shape=(data_1.shape[1:]))) 
        model.add(Dropout(0.2))
        model.add(LSTM(64, activation='tanh'))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.summary()
        model.compile(optimizer='adam',loss='mse',metrics='mse')       
        return model
    
    def train_model(self, path, model, data_1, data_2, epochs):
        log_files = os.path.join(path,datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
        tensorboard_callback = TensorBoard(log_dir=log_files, histogram_freq=1)
        return model.fit(data_1, data_2, epochs=epochs, 
                         callbacks=tensorboard_callback)


class ModelEvaluation():
    def __init__(self):
        pass

    def model_pred(self, model, data_1, data_2, scaler):
        predicted = np.array([model.predict(np.expand_dims(test,axis=0)) 
                          for test in data_1])
        new_data_1 = scaler.inverse_transform(np.expand_dims(data_2,axis=-1))
        new_data_2 = scaler.inverse_transform(predicted.reshape(len(predicted),
                                                                1))
        return new_data_1, new_data_2
