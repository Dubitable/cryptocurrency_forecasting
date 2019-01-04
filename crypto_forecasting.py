# -*- coding: utf-8 -*-
"""
Author: Logan Emery
Python Version: 2.7
Created: 12/14/2018
Last Updated: 01/04/2018
"""




# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~ The following scripts are used by both train/test validation and future prediction. ~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""    
Import the data from csv and make some small changes for computation.  Specific changes noted below.
Inputs:
    csv (as string): input csv, will always be 'consolidated_coin_data.csv' for this project
Returns:
    df (as DataFrame): finalized dataframe containing all the given data
"""
def prep_data(csv):
    import pandas as pd
    df = pd.read_csv(csv)
    # Changes the raw date text to datetime format.
    df = df.assign(Date=pd.to_datetime(df['Date']))
    # Replaces raw text nulls in Volume with 0, and changes to int.
    df.loc[df['Volume'] == '-', 'Volume'] = 0
    df['Volume'] = df['Volume'].astype('int64')
    return(df)


"""
Take the mean of the Close and Open data for the designated currency and convert to column vector, in chronological order.
Inputs:
    df (as DataFrame): data prepped from prep_data
    currency (as string): name of the desired currency
    scale (as int): number of most recent observations to take.  Recommended to set to 100.
Returns:
    data (as numpy array of shape (scale, 1)): data ready to be input to the transformations or algorithm.  Note that the data must be reshaped to 2d for keras.
"""
def prep_currency(df, currency, scale):
    import numpy as np
    currencydf = df[df['Currency'] == currency].sort_values(by='Date').reset_index(drop=True)
    # Mean of the Close and Open prices.
    currencyprice = (currencydf['Close'] + currencydf['Open']) / 2
    # Take only the most recent observations, with (scale)-many.  Ensures that the model does not emphasize fitting the distant past.
    currencyprice = currencyprice[-scale:]
    currencydata = np.array(currencyprice)
    # Reshape data for keras input.
    data = currencydata.reshape((len(currencydata), 1))
    return(data)


"""
Smooth a training set using exponential moving average.  This helps with making predictions less volatile.
Inputs:
    data_array (as numpy array of shape (scale, 1)): vector containing prices prepped from prep_currency
    alpha: larger alpha results in smoothing that emphasizes the current data value, smaller emphasizes past data values
        - empirically, alpha = 0.4 performs best for this project
Returns:
    data_array (as numpy array of shape (scale, 1)): vector containing the smoothed prices
""" 
def exp_smooth(data_array, alpha):
    import numpy as np
    workinglst = []
    # Turns 2D array into a list for ease of coding.
    for i in data_array:
        workinglst.append(i[0])
    # Initialize the smoothing equation.
    EMA = workinglst[0]
    for j in range(len(workinglst)-1):
        # Exponential moving average.  Just a weighted average of the current value and the previous (smoothed) value.
        EMA = alpha * workinglst[j+1] + (1 - alpha) * EMA
        workinglst[j+1] = EMA
    # Reshape data for keras input.
    data_array = np.array(workinglst).reshape(len(workinglst), 1)
    return(data_array)


"""
Square roots and differences data to make more stationary.
Inputs:
    data (as numpy array of shape (scale, 1)): vector containing prices
Returns:
    transformdata (as numpy array of shape (scale, 1)): vector containinged transformed prices
    initial_root (as float): the first rooted price.  Necessary when attempting to undifference the training data.
    last_root (as float): the last rooted price.  Necessary when attempting to undifference the predicted pricing.
"""
def stationarity_transform(data):
    import numpy as np
    import pandas as pd
    # Note that when transforming, root first then difference.  When un-transforming, difference first, then square.
    rootdata = np.sqrt(data)
    # Take the first and last roots of the data before differencing for use later when undifferencing.
    initial_root = rootdata.reshape(len(rootdata))[0]
    last_root = rootdata.reshape(len(rootdata))[-1]
    # Must turn np array into Series to use built in differencing method.
    rootseries = pd.Series(rootdata.reshape(len(rootdata)))
    diffseries = rootseries.diff()[1:]
    # Reshape data for keras input.
    transformdata = np.array(diffseries).reshape((len(diffseries), 1))
    return(transformdata, initial_root, last_root)      


"""
Normalize the inputs using sklearn, and keep the scaler for later use.
Inputs:
    data (as numpy array of shape (scale, 1)): data, whether transformed or not
Returns:
    data (as numpy array of shape (scale, 1)): scaled data
    scaler: the factors used to normalize the features, used later to unscale
"""
def normalize(data):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data = scaler.fit_transform(data)
    return(data, scaler)




# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~ The following scripts are used only for training/testing validation. ~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""
Splits the time series data into training and testing sets.
Note that this is intended to be used on data that is already scaled.
Inputs:
    data_array (as numpy array of size (scale, 1)): data set to be split
    test_size (as float): the percent size of the testing split
Returns:
    train (as numpy array of size ((1-test_size)*scale, 1)): training set
    test (as numpy array of size (test_size*scale, 1)): testing set
Note that the training and testing sets for time series are sequential, meaning that the training set is
necessarily chronologically before the testing set.
"""
def split_data(data_array, test_size=0.25):
    train_size = int(len(data_array) * (1 - test_size))
    train, test = data_array[0:train_size,:], data_array[train_size:len(data_array),:]
    return(train, test)
    
    
"""
Staggers inputs to create target variable and reshapes for LSTM.
Inputs:
    train (as numpy array of above shape): training data
    test (as numpy array of above shape): testing data
    batch (as int): batch size, number of observations to consider at one time in keras
    step_size (as int, default 1): number of additional features to create by using successive observations in the time series
Returns:
    trainx (as numpy array of size (train-step_size, 1, step_size)): the training observations
    trainy (as numpy array of size (train-step_size,)): the training target
    testx (as numpy array of size (test-step_size, 1, step_size)): the testing observations
    testy (as numpy array of size (test-step_size,)): the testing target
"""
def reshape_inputs_split(train, test, batch, step_size=1):
    from crypto_forecasting import stagger
    import numpy as np
    # As stated above, staggering creates more features by using multiple data points (if staggered by more than one step),
    # but it is necessary to stagger at least a step_size of 1.  This creates the target variable y, which is the next observation
    # in the time series.
    trainx, trainy = stagger(train, step_size)
    testx, testy = stagger(test, step_size)
    # Reshape inputs for keras.
    trainx = np.reshape(trainx, (trainx.shape[0], 1, trainx.shape[1]))
    testx = np.reshape(testx, (testx.shape[0], 1, testx.shape[1]))
    return(trainx, trainy, testx, testy)
    # Note: due to the stagger function, this removes step_size + 1 from total length of the data.


"""
Fit the LSTM model.
Inputs:
    trainx (as numpy array of aforementioned size): the training observations
    trainy (as numpy array of aforementioned size): the training target
    epochs (as int): the number of times the model processes the training data
    batch (as int): the number of observations for the model to consider at one time, must be a divisor of the size of the data set
    neurons (as int): the number of nodes in the hidden layer of the LSTM
    step_size (as int, default 1): same as before
Returns:
    model (as keras model): the trained model ready to predict
"""
def fit_model(trainx, trainy, epochs, batch, neurons, step_size=1):
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    # Initialize a feed-forwarde like model.
    model = Sequential()
    # The line below adds another layer to the network.  This yielded minimal accuracy improvements for a large amount of computation time.
    #model.add(LSTM(neurons, input_shape=(1, step_size), batch_input_shape=(batch, 1, 1), stateful=True, return_sequences=True))
    # Add the LSTM layer.
    model.add(LSTM(neurons, input_shape=(1, step_size), batch_input_shape=(batch, 1, 1), stateful=True))
    # Add the output layer.
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    # Resetting states is necessary when running the model multiple times, to avoid carrying over past learned trends.
    model.reset_states()
    model.fit(trainx, trainy, epochs=epochs, batch_size=batch, verbose=0)
    return(model)


"""
Calculates the predictions along with RMS error scores using the testing set.
Inputs:
    model (as keras model): the model trained using fit_model
    trainx (as numpy array of aforementioned size): the training observations
    testx (as numpy array of aforementioned size): the testing observations
    trainy (as numpy array of aforementioned size): the training target
    testy (as numpy array of aforementioned size): the testing target
    scaler (as before): the scaler generated in the normalization function
    batch (as int): the number of observations for the model to consider at one time, must be a divisor of the size of the data set
Returns:
    trainpredict (as numpy matrix of size (trainy,)): the predictions of the model on the training set
    trainy (as numpy matrix of aforementioned size): the training target, now inverse scaled to be graphed
    trainscore (as float): the RMS error of the training predictions versus the training set
    testpredict (as numpy matrix of size (testy,)): the predictions of the model on the testing set
    testy (as numpy matrix of aforementioned size): the testing target, now inverse scaled to be graphed
    testscore (as float): the RMS error of the testing predictions versus the testing set
"""
def calc_train(model, trainx, testx, trainy, testy, scaler, batch):
    from sklearn.metrics import mean_squared_error
    import math
    # Use the model to predict currency price.
    trainpredict = model.predict(trainx, batch_size=batch)
    testpredict = model.predict(testx, batch_size=batch)
    # Reinvert matrices.
    trainpredict = scaler.inverse_transform(trainpredict)
    trainy = scaler.inverse_transform([trainy])
    testpredict = scaler.inverse_transform(testpredict)
    testy = scaler.inverse_transform([testy])
    # Calculate RMS error scores.
    trainscore = math.sqrt(mean_squared_error(trainy[0], trainpredict[:,0]))
    testscore = math.sqrt(mean_squared_error(testy[0], testpredict[:,0]))
    print(trainscore, testscore)
    return(trainpredict, trainy, trainscore, testpredict, testy, testscore)


"""
Plots the RAW data and predictions on the same plot for comparison.
Inputs:
    data_array (as numpy array of size (data,)): the original, untrasformed pricing data
    trainpredict (as numpy array of size (trainy,)): the predictions on the training set generated by the model
    testpredict (as numpy array of size (testy,)): the predictions on the testing set generated by the model
    scaler (as before): the scaler generated in the normalization function, used to un-normalize the data
    batch (as int): must be the same batch size as previously used in the model
Returns:
    trainpredictplot (as numpy array of size (data,)): the plotting data containing the predictions on the training set for the first
        part of the graph, equal to as many points in the training set, and no points for the rest
    testpredictplot (as numpy array of size (data,)): the plotting data containing the predictions on the testing set for the last
        part of the graph, equal to as many points in the testing set, and no points for the rest
    Note that the above two components, when graphed on the same plot, form one complete prediction of the entire data set.
"""
def plot_predicts(data_array, trainpredict, testpredict, scaler, batch):
    import matplotlib.pyplot as plt
    import numpy as np
    # Shift train prediction back.
    trainpredictplot = np.empty_like(data_array)
    trainpredictplot[:, :] = np.nan
    trainpredictplot[:len(trainpredict), :] = trainpredict
    # Shift test prediction back.
    testpredictplot = np.empty_like(data_array)
    testpredictplot[:, :] = np.nan
    testpredictplot[len(trainpredict)+2:len(trainpredict)+len(testpredict)+2, :] = testpredict
    # Plot both predictions and actual data on the same plot.
    plt.plot(scaler.inverse_transform(data_array))
    plt.plot(trainpredictplot)
    plt.plot(testpredictplot)
    plt.show()
    return(trainpredictplot, testpredictplot)


"""
Plots the TRANSFORMED data and predictions on the same plot for comparison.
Inputs:
    diff (as numpy array of size (data-1,)): the transformed data
    trainpredict (as numpy array of size (trainy,)): the predictions on the training set generated by the model
    testpredict (as numpy array of size (testy,)): the predictions on the testing set generated by the model
    scaler (as before): the scaler generated in the normalization function, used to un-normalize the data
    batch (as int): must be the same batch size as previously used in the model
Returns:
    trainpredictplot (as numpy array of size (data,)): the plotting data containing the predictions on the training set for the first
        part of the graph, equal to as many points in the training set, and no points for the rest
    testpredictplot (as numpy array of size (data,)): the plotting data containing the predictions on the testing set for the last
        part of the graph, equal to as many points in the testing set, and no points for the rest
    Note that the above two components, when graphed on the same plot, form one complete prediction of the entire data set.
"""
def plot_predicts_diff(diff, trainpredict, testpredict, scaler, batch):
    import matplotlib.pyplot as plt
    import numpy as np
    # Shift train prediction back.
    trainpredictplot = np.empty_like(diff)
    trainpredictplot[:, :] = np.nan
    trainpredictplot[:len(trainpredict), :] = trainpredict
    # Shift test prediction back.
    testpredictplot = np.empty_like(diff)
    testpredictplot[:, :] = np.nan
    testpredictplot[len(trainpredict)+2:len(trainpredict)+len(testpredict)+2, :] = testpredict
    # Plot both predictions and actual data on plot.
    plt.plot(scaler.inverse_transform(diff))
    plt.plot(trainpredictplot)
    plt.plot(testpredictplot)
    plt.savefig('diff.png')
    plt.show()
    return(trainpredictplot, testpredictplot)


"""
Plots the UNTRANSFORMED data and predictions on the same plot for comparison.
Inputs:
    data (as numpy array): the raw, untransformed data
    diff (as numpy array of size (data-1,)): the transformed data
    trainpredict (as numpy array of size (trainy,)): the predictions on the training set generated by the model
    testpredict (as numpy array of size (testy,)): the predictions on the testing set generated by the model
    initial_root (as float): the square root of the first data point, generated by the stationarity_transform function, necessary for undifferencing the data
    scaler (as before): the scaler generated in the normalization function, used to un-normalize the data
    batch (as int): must be the same batch size as previously used in the model
Returns:
    trainpredictplot (as numpy array of size (data,)): the plotting data containing the predictions on the training set for the first
        part of the graph, equal to as many points in the training set, and no points for the rest
    testpredictplot (as numpy array of size (data,)): the plotting data containing the predictions on the testing set for the last
        part of the graph, equal to as many points in the testing set, and no points for the rest
    Note that the above two components, when graphed on the same plot, form one complete prediction of the entire data set.
"""
def plot_predicts_undiff(data, diff, trainpredict, testpredict, initial_root, scaler, batch):
    import matplotlib.pyplot as plt
    import numpy as np
    from crypto_forecasting import undiff
    # Undifference the training prediction.
    trainpredict_undiff = np.array(undiff(initial_root, trainpredict)).reshape(len(trainpredict), 1)
    #trainpredict_unlog = np.exp(trainpredict_undiff)
    trainpredict_unroot = np.square(trainpredict_undiff)
    # Shift train prediction back.
    trainpredictplot = np.empty_like(diff)
    trainpredictplot[:, :] = np.nan
    trainpredictplot[:len(trainpredict_unroot), :] = trainpredict_unroot
    # Undifference the testing prediction.
    initial = trainpredict_undiff[-1][0]
    testpredict_undiff = np.array(undiff(initial, testpredict)).reshape(len(testpredict), 1)
    #testpredict_unlog = np.exp(testpredict_undiff)
    testpredict_unroot = np.square(testpredict_undiff)
    # Shift test prediction back.
    testpredictplot = np.empty_like(diff)
    testpredictplot[:, :] = np.nan
    testpredictplot[len(trainpredict_unroot)+4:len(trainpredict_unroot)+len(testpredict_unroot)+6, :] = testpredict_unroot
    # Plot both predictions and actual data on plot.
    plt.plot(data)
    plt.plot(trainpredictplot)
    plt.plot(testpredictplot)
    plt.savefig('undiff.png')
    plt.show()
    return(trainpredictplot, testpredictplot)


"""
Runs the entire model process using all the above functions and graphs the TRANSFORMED results, specifically for training/testing validation.
For documentation on the inputs and outputs of this function, please refer to the above functions.
"""
def run_all(df, currency, scale, batch, epochs, neurons):
    from crypto_forecasting import prep_currency
    from crypto_forecasting import normalize
    from crypto_forecasting import split_data
    from crypto_forecasting import reshape_inputs_split
    from crypto_forecasting import fit_model
    from crypto_forecasting import calc_train
    from crypto_forecasting import plot_predicts
    data = prep_currency(df, currency, scale=scale, transform=False)
    data, scaler = normalize(data)
    train, test = split_data(data)
    trainx, trainy, testx, testy = reshape_inputs_split(train, test, batch)
    model = fit_model(trainx, trainy, epochs, batch, neurons)
    trainpredict, trainy, trainscore, testpredict, testy, testscore = calc_train(model, trainx, testx, trainy, testy, scaler, batch)
    trainpredictplot, testpredictplot = plot_predicts(data, trainpredict, testpredict, scaler, batch)


"""
Runs the entire model process using all the above functions and graphs the UNTRANSFORMED results, specifically for training/testing validation.
For documentation on the inputs and outputs of this function, please refer to the above functions.
"""
def run_all_diff(df, currency, alpha, scale, batch, epochs, neurons):
    from crypto_forecasting import prep_currency
    from crypto_forecasting import exp_smooth
    from crypto_forecasting import stationarity_transform
    from crypto_forecasting import normalize
    from crypto_forecasting import split_data
    from crypto_forecasting import reshape_inputs_split
    from crypto_forecasting import fit_model
    from crypto_forecasting import calc_train
    from crypto_forecasting import plot_predicts_diff
    from crypto_forecasting import plot_predicts_undiff
    data = prep_currency(df, currency, scale)
    smoothdata = exp_smooth(data, alpha)
    transformdata, initial_root = stationarity_transform(smoothdata)
    transformdata, scaler = normalize(transformdata)
    train, test = split_data(transformdata)
    trainx, trainy, testx, testy = reshape_inputs_split(train, test, batch)
    model = fit_model(trainx, trainy, epochs, batch, neurons)
    trainpredict, trainy, trainscore, testpredict, testy, testscore = calc_train(model, trainx, testx, trainy, testy, scaler, batch)
    trainpredictplot_diff, testpredictplot_diff = plot_predicts_diff(transformdata, trainpredict, testpredict, scaler, batch)
    trainpredictplot_undiff, testpredictplot_undiff = plot_predicts_undiff(data, transformdata, trainpredict, testpredict, initial_root, scaler, batch)
    



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~ The following scripts are used only for prediction. ~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""
Staggers inputs to create target variable and reshapes for LSTM.
Inputs:
    train (as numpy array of (data,)): training data, may be untransformed or transformed
    step_size (as int, default 1): number of additional features to create by using successive observations in the time series
Returns:
    trainx (as numpy array of size (train-step_size, 1, step_size)): the training observations
    trainy (as numpy array of size (train-step_size,)): the training target
"""
def reshape_inputs_predict(train, step_size=1):
    from crypto_forecasting import stagger
    import numpy as np
    # As stated above, staggering creates more features by using multiple data points (if staggered by more than one step),
    # but it is necessary to stagger at least a step_size of 1.  This creates the target variable y, which is the next observation
    # in the time series.
    trainx, trainy = stagger(train, step_size)
    # Reshape the inputs for keras.
    trainx = np.reshape(trainx, (trainx.shape[0], 1, trainx.shape[1]))
    return(trainx, trainy)


"""
Calculates the predictions along with RMS error scores using the testing set.
Inputs:
    model (as keras model): the model trained using fit_model
    trainx (as numpy array of aforementioned size): the training observations
    scaler (as before): the scaler generated in the normalization function
    batch (as int): the number of observations for the model to consider at one time, must be a divisor of the size of the data set
    desired_days (as int, default 30): the number of desired days to predict
Returns:
    future (as list): the desired future predictions
"""
def calc_future(model, trainx, scaler, batch, desired_days=30):
    import numpy as np
    # Create the predictions on the training set.
    predictions = model.predict(trainx, batch_size=batch)
    future = []
    # Reshape the current batch into the required shape for keras.
    currentstep = np.reshape(predictions[-batch:,:], (batch, 1, 1))
    #To predict further than the training set, predict one day in the future iteratively.
    for i in range(int(desired_days/batch)):
        currentstep = np.reshape(model.predict(currentstep), (batch, 1, 1))
        for i in currentstep:
            future.append(currentstep[0][0])
    return(future)


"""
Untransforms the future predictions.  Should result in a segment of a smooth parabola due to inverting the square root.
Inputs:
    future (as list): the predicted future values from the model
    last_root (as float): the square root of the final training observation, necessary to undifference the predictions
    scaler (as before): the scaler generated in the normalization function
Returns:
    unroot_future (as numpy array of shape (future,)): the untransformed future prediction values
"""   
def untransform_future(future, last_root, scaler):
    import numpy as np
    # Reshape the future values into the standard shape.
    future = np.reshape(future, (len(future), 1))
    # Un-normalize the future values.
    unscaled_future = scaler.inverse_transform(future)
    # Undifference the future values.
    undiff_future = undiff(last_root, unscaled_future)
    # Unroot the future values by squaring.
    unroot_future = np.square(undiff_future)
    return(unroot_future)
    

"""
Plots the training data and the future predictions on the same plot for visual analysis.
Inputs:
    data (as numpy array of shape (data, 1)): the original currency price data, from prep_currency
    unroot_future (as numpy array of shape (unroot_future,)): the untransformed future prediction values from the model
Returns:
    x1 (as range): the range of the training data, used for the x-axis
    y1 (as numpy array of shape (data, 1)): the original training data, used for plotting
    x2 (as range): the range of the predictions, used for the remaining x-axis
    y2 (as numpy array of shape (unroot_future, 1)): the predicted values, used for plotting
"""
def plot_future(data, unroot_future):
    import matplotlib.pyplot as plt
    import numpy as np
    # Reshape the future values, just in case.
    unroot_future = np.reshape(unroot_future, (len(unroot_future), 1))
    y1 = data
    y2 = unroot_future
    # Create the x-axis from the x-axis of both the training and predicted sets.
    x1 = range(len(y1)+len(y2))[:len(y1)]
    x2 = range(len(y1)+len(y2))[len(y1):]
    # Plot all values on the same axis.
    plt.plot(x1, y1)
    plt.plot(x2, y2)
    return(x1, y1, x2, y2)


"""
Runs the entire model process using all the above functions and graphs the UNTRANSFORMED results, specifically for future prediction.
For documentation on the inputs and outputs of this function, please refer to the above functions.
"""
def run_all_future(df, currency, scale, epoch, neuron, alpha=0.4, batch=1):
    from crypto_forecasting import prep_currency
    from crypto_forecasting import exp_smooth
    from crypto_forecasting import stationarity_transform
    from crypto_forecasting import normalize
    from crypto_forecasting import reshape_inputs_predict
    from crypto_forecasting import fit_model
    from crypto_forecasting import calc_future
    from crypto_forecasting import untransform_future
    from crypto_forecasting import plot_future
    data = prep_currency(df, currency, scale)
    smoothdata = exp_smooth(data, alpha)
    transformdata, initial_root, last_root = stationarity_transform(smoothdata)
    normdata, scaler = normalize(transformdata)
    trainx, trainy = reshape_inputs_predict(normdata)
    model = fit_model(trainx, trainy, epochs=epoch, batch=batch, neurons=neuron)
    future = calc_future(model, trainx, scaler, batch=batch)
    unroot_future = untransform_future(future, last_root, scaler)
    #x1, y1, x2, y2 = plot_future(data, unroot_future)
    return(unroot_future)


"""
If desired, this function can be used to run the model multiple times for predictions, and average all the results.
This can be used to combat outliers in the predictions.
For documentation on the inputs and outputs of this function, please refer to the above functions.
""" 
def mean_future(df, currency, scale, epoch, neuron, runs=5):
    from crypto_forecasting import run_all_future
    from crypto_forecasting import plot_future
    from crypto_forecasting import prep_currency
    import pandas as pd
    import numpy as np
    all_futures = []
    for i in range(runs):
        current_future = run_all_future(df, currency, scale, epoch, neuron)
        all_futures.append(current_future)
    all_futures_df = pd.DataFrame()
    j = 1
    for i in range(len(all_futures)):
        titlestring = "Future #" + str(j)
        all_futures_df[titlestring] = all_futures[i]
        j += 1
    all_futures_df['Mean Future'] = all_futures_df.mean(axis=1)
    mean_future_data = np.array(all_futures_df['Mean Future'])
    data = prep_currency(df, currency, scale)
    x1, y1, x2, y2 = plot_future(data, mean_future_data)
    return(mean_future_data)




# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~ These functions use the above functions to run all currencies and provide the final answer. ~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
"""
Predicts the future values for ALL currencies, and stores their results in a dictionary.
Note that this function is NOT ADVISED, as run times can be exceedingly long.
Inputs:
    df (as DataFrame): the original dataframe used to store all currency data
    scale (as int): the number of desired days to use for training
    epoch (as int): the number of epochs to use in the LSTM model
    neuron (as int): the number of nodes to be used in the hidden layer of the LSTM model
Returns:
    future_dict (as dictionary): the dictionary containing all predicted future values of every currency.  Each currency string is the key to a list containing the values
"""
def run_all_currencies(df, scale, epoch, neuron):
    from crypto_forecasting import run_all_future
    # In order to remove currencies with insufficient data, add a column to the DataFrame containing the frequency of the currency.
    df['Currency Frequency'] = df.groupby('Currency')['Currency'].transform('count')
    # Filter out said currencies with insufficient amount of data.
    passing_df = df[df['Currency Frequency'] > 99]
    passing_currencies = passing_df.drop_duplicates(subset='Currency')
    passing_currency_list = passing_currencies['Currency'].values
    future_dict = {}
    # i will be used as an index to print the status of the script.
    i = 1
    # Iterate over each of the currencies with enough data, run the model and append the dictionary.
    for current_currency in passing_currency_list:
        currency_future = run_all_future(df, current_currency, scale, epoch, neuron)
        future_dict[current_currency] = currency_future
        # Print the completion message for the currency, and update the index.
        index_text = "Completed iteration on currency:" + str(i) + " " + current_currency
        print(index_text)
        i += 1
    return(future_dict)


"""
Finds the currencies with sufficient data (as above), and splits them into smaller chunks to reduce run times.
Inputs:
    df (as DataFrame): the original currency data
Returns:
    list(i) (as list): the lists of all currencies with sufficient data, split into six pieces.
"""
def split_currencies(df):
    # In order to remove currencies with insufficient data, add a column to the DataFrame containing the frequency of the currency.
    df['Currency Frequency'] = df.groupby('Currency')['Currency'].transform('count')
    # Filter out said currencies with insufficient amount of data.
    passing_df = df[df['Currency Frequency'] > 99]
    passing_currencies = passing_df.drop_duplicates(subset='Currency')
    passing_currency_list = passing_currencies['Currency'].values
    # Split the list into six smaller lists.
    # Note: this function can be modified using modulus to provide a more flexible partition.
    list1 = passing_currency_list[:25]
    list2 = passing_currency_list[25:50]
    list3 = passing_currency_list[50:75]
    list4 = passing_currency_list[75:100]
    list5 = passing_currency_list[100:125]
    list6 = passing_currency_list[125:]
    return(list1, list2, list3, list4, list5, list6)


"""
Runs the model for one of the six partitions created by split_currencies.
Note: When using this function, you must initialize future_dict on the first call with an empty dictionary.
Inputs:
    df (as DataFrame): the original currency data
    currency_list (as list): the list of ONE partition of currencies with sufficient data
    future_dict (as dictionary): the dictionary containing the running total of all currencies and their predicted future values
    scale (as int): the number of desired days to use for training
    epoch (as int): the number of epochs to use in the LSTM model
    neuron (as int): the number of nodes to be used in the hidden layer of the LSTM model
Returns:
    future_dict (as dictionary): the running total of all currencies and predictions, updated to include the latest partition
"""
def run_some_currencies(df, currency_list, future_dict, scale, epoch, neuron):
    from crypto_forecasting import run_all_future
    i = 1
    for currency in currency_list:
        currency_future = run_all_future(df, currency, scale, epoch, neuron)
        future_dict[currency] = currency_future
        index_text = "Completed iteration on currency:" + str(i) + " " + currency
        print(index_text)
        i += 1
    return(future_dict)


"""
If desired, the function below can be used to run multiple partitions in one command.
Inputs:
    df (as DataFrame): the original currency data
    currency_list (as list): the list of ONE partition of currencies with sufficient data
    scale (as int): the number of desired days to use for training
    epoch (as int): the number of epochs to use in the LSTM model
    neuron (as int): the number of nodes to be used in the hidden layer of the LSTM model
Returns:
    future_dict (as dictionary): the running total of all currencies and predictions, updated to include the latest partition
"""
def run_currency_chunks(df, currency_list, scale, epoch, neuron):
    from crypto_forecasting import split_currencies
    from crypto_forecasting import run_some_currencies
    list1, list2, list3, list4, list5, list6 = split_currencies(df)
    list_index = [list1, list2, list3, list4, list5, list6]
    future_dict = {}
    for element in list_index:
        future_dict = run_some_currencies(df, element, future_dict, scale, epoch, neuron)
    return(future_dict)
        

"""
Finds the top 3 % gains of all cryptocurrency predictions and their projected best close out day.
Inputs:
    df (as DataFrame): the original raw dataframe containing cryptocurrency price data
    future_dict (as dictionary): the dictionary containing all future predictions for currencies given by the model
Returns:
    top_3_df (as DataFrame): a small dataframe containing information on the top 3 performing currencies
"""
def find_solution(df, future_dict):
    import pandas as pd
    import numpy as np
    # Reobtain the list of all currencies with sufficient data.
    df['Currency Frequency'] = df.groupby('Currency')['Currency'].transform('count')
    passing_df = df[df['Currency Frequency'] > 99]
    passing_currencies = passing_df.drop_duplicates(subset='Currency')
    passing_currency_list = passing_currencies['Currency'].values
    # Obtain the final price of each currency to calculate maximum % gain over the futures.
    last_price_df = passing_df.sort_values(by=['Currency', 'Close']).drop_duplicates(subset='Currency', keep='last')
    last_price_df = pd.concat([last_price_df['Currency'], last_price_df['Close']], axis=1)
    last_price_list = last_price_df.values
    # Create a dictionary of the final prices of each currency.
    last_price_dictionary = {}
    for row in last_price_list:
        last_price_dictionary[row[0]] = row[1]
    # Find the % gain between the last price and the max prediction for each currency, along with the index (day) of that max price.
    final_solution_list = []
    for currency in passing_currency_list:
        max_price = np.amax(future_dict[currency])
        max_price_percent = (max_price - last_price_dictionary[currency]) / last_price_dictionary[currency]
        max_price_day = np.argmax(future_dict[currency]) + 1
        last_close_price = last_price_dictionary[currency]
        final_solution_list.append([currency, last_close_price, max_price, max_price_percent, max_price_day])
    # Find the top 3 performers in the list of all max price increases.
    final_solution_df = pd.DataFrame(final_solution_list)
    top_3_df = final_solution_df.nlargest(30, 3).reset_index(drop=True)
    top_3_df[3] = pd.Series(["{0:.2f}%".format(val * 100) for val in top_3_df[3]], index = top_3_df.index)
    top_3_df.columns = ['Currency', 'Current Closing Price', 'Maximum Price', '% Gain', 'Suggested Close Day']
    return(top_3_df)
    
    
"""
Plot the top 3 performing currencies and save the figures.
Inputs:
    df (as DataFrame): the original raw currency data
    future_dict (as dictionary): the dictionary containing all future predictions for currencies given by the model
    top_3_df (as DataFrame): the top 3 performers data as given by find_solution
"""
def graph_winners(df, future_dict, top_3_df):
    from crypto_forecasting import prep_currency
    import matplotlib.pyplot as plt
    winners_list = top_3_df['Currency'].values
    for currency in winners_list:
        graph_title = currency + "_prediction_graph.png"
        data = prep_currency(df, currency, 100)
        future = future_dict[currency]
        x1 = range(len(data)+len(future))[:len(data)]
        x2 = range(len(data)+len(future))[len(data):]
        plt.plot(x1, data)
        plt.plot(x2, future)
        plt.savefig(graph_title)
        plt.clf()
        

"""
Run the entire project from start to finish.
Inputs:
    csv (as string): the file name of the raw .csv data, for this project 'consolidated_coin_data.csv'
    scale (as int): the number of most recent data observations to consider in the model, for this project 100
    epoch (as int): the number of epochs in the LSTM, for this project 1500
    neuron (as int): the number of neurons in the hidden layer of the LSTM, for this project 4
"""
def run_entire_project(csv, scale, epoch, neuron):
    from crypto_forecasting import prep_data
    from crypto_forecasting import run_all_currencies
    from crypto_forecasting import find_solution
    from crypto_forecasting import graph_winners
    df = prep_data(csv)
    future_dict = run_all_currencies(df, scale, epoch, neuron)
    top_3_df = find_solution(df, future_dict)
    graph_winners(df, future_dict, top_3_df)
    



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~ The following are "helper" functions, used as pieces to other functions. ~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 
    
"""
Uses ADFuller test to check for difference stationarity.
Inputs:
    data (as numpy array of shape (data, 1)): the raw time series data
Returns:
    Boolean: if true, the time series exhibits difference stationarity; if false, then the series must be differenced
"""
def adf_test(data):
    from statsmodels.tsa.stattools import adfuller
    import pandas as pd
    data = pd.Series(data.reshape(len(data)))
    adftest = adfuller(data)
    if adftest[0] < adftest[4]['1%']:
        return(True)
    else:
        return(False)


"""
Uses KPSS test to check for trend stationarity.
Inputs:
    data (as numpy array of shape (data, 1)): the raw time series data
Returns:
    Boolean: if true, the time series exhibits trend stationarity; if false, then the series must be transformed (usually log or root)
"""
def kpss_test(data):
    from statsmodels.tsa.stattools import kpss
    import pandas as pd
    data = pd.Series(data.reshape(len(data)))
    kpsstest = kpss(data)
    if kpsstest[0] < kpsstest[3]['10%']:
        return(True)
    else:
        return(False)


"""
The stagger function; changes the observation points to be tuples of size='step_size',
so the model uses adjacent observations as multiple features.
Inputs:
    data (as numpy array of shape (data, 1)): the time series data
    step_size (as int, default 1): the desired number of features to be obtained
Returns:
    dataX (as numpy array of size (data-step_size, step_size)): the tuples to be used as observations
    dataY (as numpy array of size (data-step_size, step_size)): the tuples to be used as target variables
Note: this process removes step_size+1 length from the total length of the data.
This is very important for consideration when choosing the batch size in the algorithm,
as the total number of data observations must be divisible by the batch size.
"""
def stagger(data, step_size=1):
    import numpy as np
    dataX, dataY = [], []
    for i in range(len(data)-step_size-1):
        step = data[i:(i+step_size), 0]
        dataX.append(step)
        dataY.append(data[i + step_size, 0])
    return np.array(dataX), np.array(dataY)


"""
Undifferences the previously differenced data.
Inputs:
    initial (as float): the first datapoint of the NONDIFFERENCED time series.  This is necessary to obtain the original sequence.
    diff (as numpy array of shape (data-1, 1)): the differenced data to be undifferenced
Returns:
    undiffs (as numpy array of shape (data, 1)): the undifferenced data
"""
def undiff(initial, diff):
    next_value = initial + diff[0][0]
    undiffs = list()
    undiffs.append(next_value)
    for i in range(len(diff)-1):
        next_value += diff[i+1][0]
        undiffs.append(next_value)
    return(undiffs)
