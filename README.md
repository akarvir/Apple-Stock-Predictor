· Developed a DL Model to predict if a particular day is a min(green),inflection(yellow),or max(red) point to buy or short Apple stock accordingly.

· Used the open-source yfinance library to access Apple’s stock history data, used numpy array manipulation to convert the panda dataframe into a time-series sequence for every data point(1 day).

· Recognizing the sequential nature of the data, implemented a recurrent neural network with an LSTM architecture to capture long-term dependencies in data.
