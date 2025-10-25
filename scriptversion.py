# Yfinance library (for the data)
import yfinance as yf

# Pandas library (For manipulating the data)
import pandas as pd

# Numpy for computing/data analysis
import numpy as np

# Sklearn for the model
## Importing Linear Regression model
from sklearn.linear_model import LinearRegression

## Importing train_test_split function
from sklearn.model_selection import train_test_split

## Importing evaluation metrics
from sklearn.metrics import mean_squared_error, r2_score


# Matplot for displaying visualizations
import matplotlib.pyplot as plt




# Defining which ticker we're wanting to analyze (rn we're using apple)
#ticker = 'NVDA'
ticker = input("Enter the ticker for the stock you are wanting to analyze: ")
# Importing the data from yfinance API
data = yf.download(ticker, start="2015-01-01",  end="2025-10-01")
# Displaying the first 5 rows of the data ("head" = first 5 rows. You can put whatever number you want in the parentheses to see more rows or use "tail" to see the last rows)
data.head()






# First we gotta create our target column (What we're trying to predict). In this case, we're trying to predict the next days closing price
data['Target'] = data['Close'].shift(-1)

# Drop the last row since it will have a NaN value for the target
data = data.dropna()

# Next we gotta define some simple features and target
X = data[['Open', 'High', 'Low', 'Close', 'Volume']]
y = data['Target']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False) 
# we're using shuffle=False because this is time series data and we want to maintain the chronological order




model = LinearRegression()
model.fit(X_train, y_train)


# Predictions
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')


plt.figure(figsize=(10,5))
plt.plot(y_test.values, label='Actual Prices', color='blue')
plt.plot(y_pred, label='Predicted Prices', color='red')
plt.title(f"{ticker} Stock Price Prediction (Linear Regression)")
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.show()



# Use the most recent data row as input:
latest_data = data[['Open', 'High', 'Low', 'Close', 'Volume']].iloc[-1].values.reshape(1, -1)
next_day_prediction = model.predict(latest_data)[0]

print(f"Predicted next day's closing price for {ticker}: ${next_day_prediction:.2f}")