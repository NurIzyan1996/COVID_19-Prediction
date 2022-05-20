# COVID_19 Case Prediction in Malaysia
Predicting COVID-19 new cases in Malaysia using a deep learning model.


# Description
This repository contains 2 python files (new_case.py, modules.py).
new_case.py contains the codes to build a deep learning model and train on the dataset.
modules.py contains the codes where there are class and functions to be used in train.py.

#How run Tensorboard

1. Clone this repository and use the model.h5, mms_scaler.pkl (inside saved_model folder) to deploy on your dataset.
2. Run tensorboard at the end of training to see how well the model perform via Anaconda prompt. Activate the correct environment.
3. Type "tensorboard --logdir "the log path"
4. Paste the local network link into your browser and it will automatically redirected to tensorboard local host and done! Tensorboard is now can be analyzed.

# The Architecture of Model
![The Architecture of Model](model_architecture.png)

# The Performance of model
![The Performance of model](MAPE.PNG)
![The Performance of model](actual_pred_graph.png)

# Tensorboard screenshot from my browser
![Tensorboard](tensorboard.PNG)

# Discussion
Based on the assignment given. I have succcessfully produce an LSTM model that trains on Malaysia COVID-19 new cases dataset and produces  an MAPE value around 0.15%. 

Throughtout the process, I manage to clean the data, build a one-layer LSTM model with 64 nodes and train the model with 50 epochs. 

In conclusion, the deployment of this prediction model may assist in future decision making in travel bans by Government to make sure the spread of COVID-19 can be minimized efficiently. 

# Credit
Big thanks to the owner of the datasets GitHub - MoH-Malaysia/covid19-public: Official data on the COVID-19 epidemic in Malaysia. Powered by CPRC, CPRC Hospital System, MKAK, and MySejahtera.
