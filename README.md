## Stock Market Analysis and Prediction
This project downloads historical stock price data for multiple companies and performs data analysis, feature engineering, and prediction of the next day closing prices using a linear regression model.

## Features
- Downloads historical daily stock data for selected tickers over the last 3 months.
- Combines data from all tickers into a single dataset.
- Cleans and preprocesses the dataset (handling missing values, converting data types).
- Calculates technical indicators such as 7-day moving average and rolling 7-day volatility.
- Performs exploratory data analysis including:
    * Distribution plots of closing prices by company
    * Correlation heatmap of closing prices among stocks
    * Trend plots of closing prices and      moving averages
- Feature engineering for next-day price prediction including lag features and one-hot encoding of stocks.
- Uses linear regression to predict next-day close prices.
- Evaluates model performance globally and individually per ticker.
- Visualizes actual vs predicted closing prices.

## Installation
- Create python virtual environment:
   python -m venv venv
   .\venv\Scripts\activate
- Install the rquired libraries:
    pip install pandas numpy matplotlib seaborn scikit-learn yfinance

## Usage
- Run download_dataset.py to download and save the latest three months of stock data for predefined tickers (AAPL, MSFT, NFLX, GOOG).

- Run stock_analysis.py to perform analysis, train the model, evaluate performance, and generate plots.

## Files
- download_dataset.py: Downloads and saves stock market data.

- stock_analysis.py: Performs data preprocessing, analysis, modeling, evaluation, and visualization.
