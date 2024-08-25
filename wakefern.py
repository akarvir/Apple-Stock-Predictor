
 #Predicting local min and local max of Apple stock to buy or short stock accordingly. 


import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# Simple strategy. If green point, we buy more stock. If not, we sell.
microsoft_ticker = yf.Ticker("MSFT")

stock_history = microsoft_ticker.history(period = 'max',interval = '1d')

# for each day, check the day ahead of it. This is for the model training data.

# The input parameters will be - volume, 3 day, 10 day, 20 day regression coefficient, normalized stock price. These are technical indicators

# 1 implies is a buying point, 0 implies not.


# now, creating the new dataset.


columns = ['Volume','3dayreg','10dayreg','20dayreg','normalizedsp','is_buyingpoint'] # the technical indicators serving as the input parameters.

dataset = pd.DataFrame(columns = columns)

for i in range(20,len(stock_history)-1):

  currentdayprice = stock_history.iloc[i]['Close']

  nextdayprice = stock_history.iloc[i+1]['Close']

  if(nextdayprice>currentdayprice): buying_point = 1
  else: buying_point = 0

  threexvalues = []

  for j in range(1,4):
    threexvalues.append([j])

  # getting y values of last 3 days.
  threeyvalues = []

  for m in range(i-3,i):

    threeyvalues.append(stock_history.iloc[m]['Close'])


  threedayregression = LinearRegression().fit(threexvalues,threeyvalues)

  threedaycoef = threedayregression.coef_ # got the 3-day regression coefficient.

  tenxvalues = []

  tenyvalues = []

  for f in range(1,11):
    tenxvalues.append([f]) # got the x values for the 10-day Linear regression

  for v in range(i-10,i):
    tenyvalues.append(stock_history.iloc[v]['Close'])

  tendayregression = LinearRegression().fit(tenxvalues,tenyvalues)

  tendaycoef = tendayregression.coef_

  twentyxvalues = []

  twentyyvalues = []

  for c in range(1,21): twentyxvalues.append([c])

  for x in range(i-20,i):
    twentyyvalues.append(stock_history.iloc[x]['Close'])

  twentydayregression = LinearRegression().fit(twentyxvalues, twentyyvalues)

  twentydaycoef = twentydayregression.coef_

  day_volume = stock_history.iloc[i]['Volume']

  if((stock_history.iloc[i]['High'] - stock_history.iloc[i]['Low'])==0): normalizedstockprice = 0
  else: normalizedstockprice = (stock_history.iloc[i]['Close'] - stock_history.iloc[i]['Low'])/(stock_history.iloc[i]['High'] - stock_history.iloc[i]['Low'])



  row = {'Volume' : day_volume, '3dayreg' : threedaycoef,'10dayreg':tendaycoef, '20dayreg':twentydaycoef,'normalizedsp': normalizedstockprice,'is_buyingpoint' : buying_point}


  dataset.loc[len(dataset)] = row


dataset.dropna(subset=['normalizedsp'], inplace=True)



# Implementing the model and optimizing the threshold.


from sklearn.metrics import accuracy_score

y_values = dataset['is_buyingpoint']

x_values = dataset.drop('is_buyingpoint',axis = 1)

x_train,x_test,y_train,y_test = train_test_split(x_values,y_values,test_size = 0.2, random_state = 42, shuffle = True)




model = LogisticRegression()

model.fit(x_train,y_train) # training the model

y_probs = model.predict_proba(x_test)[:, 1] # getting the respective probabilities. Threshold will determine the outcome sequence.

print(len(y_probs))

thresholds = np.arange(0.71,0.0, -0.01).tolist() # all thresholds to test.

noofbuyingpoints = 0

missbuyingpoints = 0




for threshold in thresholds:

  incorrect_counter = 0

  y_pred = (y_probs>=threshold).astype(int) # a different outcome sequence for each threshold

  misclassification_counter = 0

  for i in range(len(y_pred)):


    if(y_test.iloc[i]==1): noofbuyingpoints +=1
    if(y_test.iloc[i]==1 and y_pred[i]==0): missbuyingpoints +=1
    if(y_test.iloc[i]==0 and y_pred[i]==1): misclassification_counter +=1
    if(y_test.iloc[i]!=y_pred[i]): incorrect_counter +=1


  print("The missclassification counter for this threshold ",threshold," is ",misclassification_counter)
  print("With this threshold we make the right prediction ",((incorrect_counter)/(len(y_pred)))*100,"% of the time")
  print("Out of the times we make missclassification errors, we buy high and sell low ",((misclassification_counter)/(incorrect_counter))*100,"% of the errors! ")
  print("The model misses buying points",(missbuyingpoints/noofbuyingpoints)*100,"% of the time")
  for m in range(2): print()























