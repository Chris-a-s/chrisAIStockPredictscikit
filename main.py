import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

# Load the data
stocky = input("Please enter your ticker: ")
df = pd.read_csv(f"/Users/christianschroeder/projects/py_stock_predict/stock_data/{stocky}.csv", parse_dates=['Date'])
df.set_index('Date', inplace=True)

# Feature Engineering
df['Target'] = df['Close'].shift(-1)  # Next day's closing price
df['SMA_10'] = df['Close'].rolling(window=10).mean()  # 10-day Simple Moving Average
df['SMA_30'] = df['Close'].rolling(window=30).mean()  # 30-day Simple Moving Average

# Drop rows with NaN values resulting from rolling mean
df.dropna(inplace=True)

# Prepare features and target
X = df[['Close', 'SMA_10', 'SMA_30']]
y = df['Target']

# Time Series Split
tscv = TimeSeriesSplit(n_splits=5)
model = make_pipeline(MinMaxScaler(), LinearRegression())

# Run time series cross-validation
for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    # Calculate RMSE and print it
    rmse = mean_squared_error(y_test, predictions, squared=False)
    print(f'Test RMSE: {rmse}')

# Final Model Training on all data
model.fit(X, y)

# Define thresholds
slightly_threshold = 0.02  # 2%
significantly_threshold = 0.05  # 5%

# Calculate the percentage difference between the predicted and actual closing prices
df['Perc_Diff'] = (predictions - df['Close'].iloc[test_index]) / df['Close'].iloc[test_index]

# Generate the signals based on the percentage difference
def get_investment_signal(perc_diff):
    if perc_diff >= significantly_threshold:
        return "STRONG BUY"
    elif perc_diff >= slightly_threshold:
        return "BUY"
    elif -slightly_threshold <= perc_diff <= slightly_threshold:
        return "HOLD"
    elif perc_diff <= -significantly_threshold:
        return "STRONG SELL"
    else:
        return "SELL"

df['Signal'] = df['Perc_Diff'].apply(get_investment_signal)

# Show the dataframe with the signals
df[['Close', 'Perc_Diff', 'Signal']].tail()

# Plot the last fold with the signal label
plt.figure(figsize=(14, 7))
plt.plot(df.index[test_index], y_test, label='Actual Prices')
plt.plot(df.index[test_index], predictions, label='Predicted Prices (Linear Regression)')
plt.legend()
plt.title(f"Stock Price Prediction for {stocky}")
plt.xlabel('Date')
plt.ylabel('Price')

# Get the latest signal for annotation
latest_signal = df['Signal'].iloc[test_index[-1]]
plt.annotate(f'LR says: {latest_signal}', xy=(1, 1), xytext=(8, -8), fontsize=12,
             xycoords='axes fraction', textcoords='offset points',
             bbox=dict(facecolor='white', alpha=0.8),
             horizontalalignment='right', verticalalignment='top')

plt.show()
