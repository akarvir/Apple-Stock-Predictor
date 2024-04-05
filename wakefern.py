import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt  
import seaborn as sns  
 
import tensorflow as tf
from tensorflow import keras
from keras import Input
from keras.models import Sequential
from keras.layers import Dense, Dropout, SimpleRNN, LSTM, GRU
 
import yfinance as yf  
from sklearn.model_selection import train_test_split # 


company = 'AAPL'  
stock_data = yf.Ticker(company)

 
stock_history = stock_data.history(period='max',interval='1d')  

closing_prices = stock_history['Close']



training_data, testing_data = train_test_split(stock_history,test_size=0.2,random_state=42,shuffle=True) # Splitting the data. Decided to keep 20 percent testing
# training data
X_train = training_data.drop(['Close','Volume','Dividends','Stock Splits'],axis=1)
y_train = training_data['Close']
# testing data
 
X_test = testing_data.drop(['Close','Volume','Dividends','Stock Splits'],axis=1)
y_test = testing_data['Close']

ann_model = Sequential()
ann_model.add(Dense(25,activation='relu',kernel_initializer='he_normal',bias_initializer='ones',input_shape=(X_train.shape[1],1)))
ann_model.add(Dropout(rate=0.2))
ann_model.add(Dense(50,activation='relu',kernel_initializer='he_normal',bias_initializer='ones'))
ann_model.add(Dropout(rate=0.2))
ann_model.add(Dense(25,activation='relu',kernel_initializer='he_normal',bias_initializer='ones'))
ann_model.add(Dense(1,activation='relu',kernel_initializer='he_normal',bias_initializer='ones'))

# ann_model.summary() 

optimizer = keras.optimizers.SGD(learning_rate=0.03)
loss_function = keras.losses.MeanSquaredError()
ann_model.compile(optimizer=optimizer,loss=loss_function)

history = ann_model.fit(X_train,y_train,batch_size=32,epochs=50,shuffle=True)

testing_loss = ann_model.evaluate(X_test,y_test,batch_size=32,verbose=0)
print(f'Testing Loss in Dollars: ${np.sqrt(testing_loss)}') # About 41 dollars.

# RNN next


raw_sequence_data_X = []
raw_sequence_data_Y = []
data_list = stock_history.drop(['Volume','Dividends','Stock Splits'],axis=1).values
 # padding the image
for i in range(len(data_list)):
  sequence = [[0]*4] * 10 #  

 
  if i < 10:
    for x in range(i):
      sequence[x] = list(data_list[x])
  else:
    index = 0
    for x in range(i-10,i):
      sequence[index] = list(data_list[x])
      index += 1
  raw_sequence_data_X.append(sequence)
  raw_sequence_data_Y.append(data_list[i][3])

# Changing the lists into numpy arrays
sequence_X = np.array(raw_sequence_data_X)
sequence_Y = np.array(raw_sequence_data_Y)

#splitting the sequence data now.

X_train, X_test, y_train, y_test = train_test_split(sequence_X,sequence_Y,test_size=0.2,random_state=42,shuffle=True)


rnn_model = Sequential()

 
rnn_model.add(SimpleRNN(50,activation='tanh',bias_initializer='ones',return_sequences=False,input_shape=(X_train.shape[1],X_train.shape[2])))

 
rnn_model.add(Dense(25,activation='relu',kernel_initializer='he_normal',bias_initializer='ones'))
rnn_model.add(Dense(1,activation='relu',kernel_initializer='he_normal',bias_initializer='ones')) #Adding layers

optimizer = keras.optimizers.SGD(learning_rate=0.001)
loss_function = keras.losses.MeanSquaredError()
rnn_model.compile(optimizer=optimizer,loss=loss_function)

#training the RNN now
history = rnn_model.fit(X_train,y_train,batch_size=32,epochs=5,shuffle=True)

testing_loss = rnn_model.evaluate(X_test,y_test,batch_size=32,verbose=0)
print(f'Testing Loss in Dollars: ${np.sqrt(testing_loss)}') #About 30 dollars. way better than the ANN !

# THE LSTM - Long Short-Term Network

lstm_model = Sequential()
lstm_model.add(LSTM(30,activation='tanh',recurrent_activation='sigmoid',bias_initializer='ones',return_sequences=False,input_shape=(X_train.shape[1],X_train.shape[2])))

# At the end of our LSTM, we will get a 50 dimensional vector as our output.
# We need a single number so we will add an ANN on top of this RNN.
lstm_model.add(Dense(25,activation='relu',kernel_initializer='he_normal',bias_initializer='ones'))
lstm_model.add(Dense(1,activation='relu',kernel_initializer='he_normal',bias_initializer='ones'))

#Writing the configuration for how the model learns

optimizer = keras.optimizers.SGD(learning_rate=0.001)
loss_function = keras.losses.MeanSquaredError()
lstm_model.compile(optimizer=optimizer,loss=loss_function)

testing_loss = lstm_model.evaluate(X_test,y_test,batch_size=32,verbose=0)
print(f'Testing Loss in Dollars: ${np.sqrt(testing_loss)}') # About 29 dollars. Slightly better than the RNN
