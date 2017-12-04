
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.optimizers import Adam
from keras import regularizers
from keras import backend as K


# In[2]:

# classify all data points in 2017 into validation set, which is 42 available weeks
n_valid = 42


# In[3]:

def scaling(df):
    ''' This function is to standardize all columns in the data frame
    Args:
    df: data frame with all x and y
    
    Returns:
    scaled: the standardized data
    scaler: the object that stores the scaling informaiton which will be used to 
    inverse the standardized values to original values
    '''
    # ensure all data is float
    values = df.values
    values = values.astype('float32')
    # standardize features
    scaler = StandardScaler()
    scaled = scaler.fit_transform(values)
    return (scaled, scaler)


# In[4]:

# Some of the codes below refer to this article: 
# https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
def add_lags(df, n_lag=1, n_y=4):
    ''' This function is to add week lags to the features used to predict mortgage rate
    Args:
    df: data frame with all x and y
    n_lag: degree of lags used on all predictors
    n_y: number of dependent variables in the data frame
    
    Returns:
    agg: a data frame with all lagged predictors added
    '''
    n_vars = df.shape[1]-n_y
    feats = pd.DataFrame(df).iloc[:,:-n_y]
    y = pd.DataFrame(df).iloc[:,-n_y:]
    cols, names = list(), list()
    # add lags to predictors (t-i, ... t-1)
    for i in range(n_lag, -1, -1):
        cols.append(feats.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # combine all lagged predictors and y
    agg = pd.concat((pd.concat(cols, axis=1), y), axis=1)
    agg.columns = names + [('mortdiff(t+%d)' % (i+1)) for i in range(n_y)]
    # drop rows with NaN values
    agg.dropna(inplace=True)
    return agg


# In[5]:

def train_test_splitter(reframed_data, n_weeks, n_features, n_valid):
    ''' This function is to split the traing and validation set '''
    values = reframed_data.values
    n_train_weeks = values.shape[0]-n_valid
    train_X = values[:n_train_weeks, :(n_weeks*n_features)]
    train_y = values[:n_train_weeks, -4:]
    test_X = values[n_train_weeks:, :(n_weeks*n_features)]
    test_y = values[n_train_weeks:, -4:]
    # reshape input to 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], n_weeks, n_features))
    test_X = test_X.reshape((test_X.shape[0], n_weeks, n_features))
    return (train_X, train_y, test_X, test_y)


# In[6]:

def build_rnn(train_X, n_output, lr=0.01, decay=0.02):
    ''' This function is to build the RNN neural network structure
    Args:
    n_output: the dimension of output of the RNN layer
    '''
    model = Sequential()
    model.add(GRU(n_output, activation='tanh', return_sequences=False, 
                  input_shape=(train_X.shape[1], train_X.shape[2]),
                  kernel_regularizer=regularizers.l2(0.0005), recurrent_regularizer=regularizers.l2(0.0005)))
    model.add(Dense(4))
    adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay)
    model.compile(loss='mae', optimizer=adam)
    return model


# In[7]:

def get_embedding(model, input0, layer):
    ''' This function is to get the embedding vector (the output from the RNN layer)'''
    get_layer_output = K.function([model.layers[0].input],
                                  [model.layers[layer].output])
    layer_output = get_layer_output([input0])[0]
    return layer_output


# In[8]:

def invert_y(mort, scaler, test_X, test_y, yhat, n_features, n_weeks):
    ''' This function is to invert the standardized values to original mortgage rate values
    Returns:
    inv_y: the actual mortgage rate values for next 4 weeks
    inv_yhat: the rnn forecast of the original mortgage rate values for next 4 weeks
    nb: the naive bayes forecast of the original mortgage rate values for next 4 weeks
    '''
    test_X = test_X.reshape((test_X.shape[0], n_weeks*n_features))
    # invert the standardized forecasts
    inv_yhat = concatenate((test_X[:, -n_features:], yhat), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,-4:]
    # invert the standardized actual Y
    inv_y = concatenate((test_X[:, -n_features:], test_y), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,-4:]
    # add the differenced value back to the original mortgage rate to
    # create forecasts of actual mortgage rate values
    for i in xrange(4):
        if i == 0:
            inv_y[:,i] += np.array(mort[len(mort)-n_valid-4:len(mort)-4])
            inv_yhat[:,i] += np.array(mort[len(mort)-n_valid-4:len(mort)-4])
        else:
            inv_y[:,i] += inv_y[:,i-1]
            inv_yhat[:,i] += inv_yhat[:,i-1]
    # generate the Naive Bayes forecast
    nb = np.tile(np.array(mort[len(mort)-n_valid-4:len(mort)-4]),(4,1)).transpose()
    return inv_y, inv_yhat, nb


# In[9]:

def model_eval(y, yhat):
    ''' This function is to evaluate the results on validation dataset
    Returns:
    mae: the average MAE over all validation data
    mape: the average MAPE over all validation data
    '''
    # calculate MAE and MAPE
    mae = np.mean(abs(y - yhat))
    mape = np.mean(abs(y - yhat)/y)
    print ('Test MAE: %.3f' % mae)
    print ('Test MAPE: %.3f' % mape)
    return mae, mape


# In[10]:

def process1(features, mort, n_feats, n_weeks, n_valid, n_output, lr=0.01, decay=0.02, epochs=50, batch_size=52):
    ''' This function is to build the complete modeling process including:
    scaling and building variables, split training and validation set,
    build the rnn model, train and evaluation the model
    Returns:
    mae: the average mae over all validation data
    mape: the average mape over all validation data    
    '''
    scaled, scaler = scaling(features)
    reframed = add_lags(scaled, n_weeks-1)
    train_X, train_y, test_X, test_y = train_test_splitter(reframed, n_weeks, n_feats, n_valid)
    model = build_rnn(train_X, n_output, lr=lr, decay=decay)
    fit = model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, validation_data=(test_X, test_y), verbose=0, shuffle=False)
    yhat = model.predict(test_X)
    inv_y, inv_yhat, nb = invert_y(mort, scaler, test_X, test_y, yhat, n_feats, n_weeks)
    # to achieve the forecasts, just comment out this line and modify the return to inv_yhat
    mae, mape = model_eval(inv_y, inv_yhat)
    return mae, mape


# In[28]:

def process2(features, mort, n_feats, n_weeks, n_valid, n_embed, n_output, lr1=0.01, lr2=0.005, epochs=50, batch_size=52):
    ''' This function is to build the complete modeling process with a sophisticated RNN structure
    Returns:
    mae: the average mae over all validation data
    mape: the average mape over all validation data    
    '''
    scaled, scaler = scaling(features)
    reframed = add_lags(scaled, n_weeks-1)
    # create a copy of train & test X for scaler inversion
    train_X, train_y, test_X, test_y = train_test_splitter(reframed, n_weeks, n_feats, n_valid)
    train_X1, train_y1, test_X1, test_y1 = train_test_splitter(reframed, n_weeks, n_feats-1, n_valid)
    model1 = build_rnn(train_X1, n_embed, lr=lr1)
    fit1 = model1.fit(train_X1, train_y1, epochs=epochs, batch_size=batch_size, validation_data=(test_X1, test_y1), verbose=0, shuffle=False)
    embed1 = get_embedding(model1, np.concatenate([train_X1,test_X1]), 0)
    yhat1 = get_embedding(model1, np.concatenate([train_X1,test_X1]), 1)
    res1 = np.concatenate([train_y1,test_y1]) - yhat1
    feats2 = np.concatenate([embed1,features.iloc[(n_weeks-1):,-5:-4],res1],axis=1)
    embed2 = add_lags(feats2, n_weeks-1)
    train_X2, train_y2, test_X2, test_y2 = train_test_splitter(embed2, n_weeks, embed1.shape[1]+1, n_valid)
    model2 = build_rnn(train_X2, n_output, lr=lr2)
    fit2 = model2.fit(train_X2, train_y2, epochs=epochs, batch_size=batch_size, validation_data=(test_X2, test_y2), verbose=0, shuffle=False)
    yhat2 = model2.predict(test_X2)+yhat1[-n_valid:,:]
    inv_y, inv_yhat, nb = invert_y(mort, scaler, test_X, test_y, yhat2, n_feats, n_weeks)
    # to achieve the forecasts, just comment out this line and modify the return to inv_yhat
    mae, mape = model_eval(inv_y, inv_yhat)
    return mae, mape


# In[38]:

# read the data files generated by the R code
mort30 = pd.read_csv('mort30.csv', index_col=0)
mort15 = pd.read_csv('mort15.csv', index_col=0)
mort5 = pd.read_csv('mort5.csv', index_col=0)
morts = pd.read_csv('morts.csv')


# In[13]:

# select the most effective predictors
morts30 = mort30.loc[:,['T10YIE', 'VXTYN', 'DGS10', 'MORTGAGE30US', 'week1', 'week2', 'week3', 'week4']]
morts15 = mort15.loc[:,['T10YIE', 'MORTGAGE30US', 'DGS10', 'week1', 'week2', 'week3', 'week4']]
morts5 = mort5.loc[:,['VXTYN', 'DFII10', 'week1', 'week2', 'week3', 'week4']]


# Belows are illustrations of one-time training and validation process for the three models.
# 
# The results would be different every time the function is called due to the random starts of the weights unless a seed is set.
# 
# The result in the report is the average returned value given by repeating this process 10 times.

# In[15]:

process1(morts30, morts.MORTGAGE30US, morts30.shape[1]-4, n_weeks=6, n_valid=n_valid, n_output=12)


# In[17]:

process1(morts15, morts.MORTGAGE15US, morts15.shape[1]-4, n_weeks=8, n_valid=n_valid, n_output=12)


# In[19]:

process1(morts5, morts.MORTGAGE5US, morts5.shape[1]-4, n_weeks=8, n_valid=n_valid, n_output=10)


# Belows are illustrations of the more complicated recurrent neural network structure inspired by the Uber article.
# 
# We can see that the performance are even poorer than the single-layer RNN.
# 
# The reason of this is explained in the report.

# In[49]:

process2(morts30, morts.MORTGAGE30US, morts30.shape[1]-4, n_weeks=6, n_valid=n_valid, n_embed=12, n_output=6)


# In[46]:

morts15_2 = mort15.loc[:,['T10YIE', 'MORTGAGE30US', 'DGS10', 'MORTGAGE15US', 'week1', 'week2', 'week3', 'week4']]
process2(morts15_2, morts.MORTGAGE15US, morts15_2.shape[1]-4, n_weeks=8, n_valid=n_valid, n_embed=12, n_output=6)


# In[47]:

morts5_2 = mort5.loc[:,['VXTYN', 'DFII10', 'MORTGAGE5US', 'week1', 'week2', 'week3', 'week4']]
process2(morts5_2, morts.MORTGAGE5US, morts5_2.shape[1]-4, n_weeks=8, n_valid=n_valid, n_embed=10, n_output=6)

