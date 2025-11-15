import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

#  Load the combined stock dataset
data = pd.read_csv('all_stocks_3mo.csv')   

# Data Preprocessing
# data cleaning
data['Date'] = pd.to_datetime(data['Date'])
# type conversion
for col in ['Close', 'Open', 'High', 'Low', 'Volume']:
    data[col] = pd.to_numeric(data[col], errors='coerce')
# sorting
data = data.sort_values(by=['Ticker', 'Date'])
data.dropna(inplace=True)

# Calculate 7-day moving average of 'Close'
data['7_day_MA'] = data.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=7).mean())
# Calculate daily returns for volatility calculation
data['daily_return'] = data.groupby('Ticker')['Close'].pct_change()
# Calculate 7-day rolling volatility (standard deviation of returns)
data['volatility'] = data.groupby('Ticker')['daily_return'].transform(lambda x: x.rolling(window=7).std())
# Select and reorder the columns to display
columns_to_display = ['Date', 'Ticker', 'Low', 'High', 'Open', 'Close', '7_day_MA', 'volatility', 'Volume']
table = data[columns_to_display]
# Display the results for each company
tickers = data['Ticker'].unique()
for ticker in tickers:
    print(f"\nTable for {ticker}:")
    display_table = table[table['Ticker'] == ticker]
    print(display_table.to_string(index=False))

# Feature Engineering
data['Prev_Close'] = data.groupby('Ticker')['Close'].shift(1)
data['Prev_Volume'] = data.groupby('Ticker')['Volume'].shift(1)
data['Next_Close'] = data.groupby('Ticker')['Close'].shift(-1)

#  Drop rows with missing values due to shifting
data.dropna(inplace=True)

# One-hot encode the Ticker column
tickers_dummies = pd.get_dummies(data['Ticker'])
data = pd.concat([data, tickers_dummies], axis=1)

# Prepare features and target
feature_cols = ['Prev_Close', 'Prev_Volume', 'Open', 'High', 'Low', 'Volume'] + list(tickers_dummies.columns)
X = data[feature_cols]
y = data['Next_Close']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

#  Predict and evaluate
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Global Model Results:")
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}\n")

#  Analyze per company performance 
X_test_with_ticker = X_test.copy()
X_test_with_ticker['Actual'] = y_test
X_test_with_ticker['Predicted'] = y_pred

for ticker in tickers_dummies.columns:
    ticker_data = X_test_with_ticker[X_test_with_ticker[ticker] == 1]
    if not ticker_data.empty:
        mse_ticker = mean_squared_error(ticker_data['Actual'], ticker_data['Predicted'])
        r2_ticker = r2_score(ticker_data['Actual'], ticker_data['Predicted'])
        print(f"Results for {ticker}:")
        print(f"Mean Squared Error: {mse_ticker}")
        print(f"R^2 Score: {r2_ticker}\n")
        # Plot actual vs predicted
        plt.figure(figsize=(8, 4))
        plt.scatter(ticker_data['Actual'], ticker_data['Predicted'])
        plt.xlabel('Actual Next Day Close Price')
        plt.ylabel('Predicted Next Day Close Price')
        plt.title(f'Actual vs Predicted Closing Price for {ticker} (Global Model)')
        plt.show()

# Closing price distribution for each company
for ticker in tickers:
    data_ticker = data[data['Ticker'] == ticker]
    plt.figure(figsize=(8, 4))
    plt.hist(data_ticker['Close'], bins=30, alpha=0.7)
    plt.title(f"Closing Price Distribution for {ticker}")
    plt.xlabel("Closing Price")
    plt.ylabel("Frequency")
    plt.show()

# Pivot table for closing prices
pivot_close = data.pivot(index='Date', columns='Ticker', values='Close')
corr_matrix = pivot_close.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix of Closing Prices")
plt.show()

for ticker in tickers:
    data_ticker = data[data['Ticker'] == ticker]
    plt.figure(figsize=(10, 5))
    plt.plot(data_ticker['Date'], data_ticker['Close'], label='Close Price')
    plt.plot(data_ticker['Date'], data_ticker['7_day_MA'], label='7-Day MA')
    plt.title(f"{ticker} Closing Price & 7-Day Moving Average")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()








