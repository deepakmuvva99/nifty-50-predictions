import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import yfinance as yf
from tqdm import tqdm

# 1. Data Collection
def get_nifty100_symbols():
    # This is a simplified list. In practice, you'd want to fetch this dynamically.
    return ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS']  # Add more symbols as needed

def get_stock_data(ticker, start_date, end_date):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        if df.empty:
            print(f"No data fetched for {ticker}")
            return None
        return df
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

# 2. Data Preprocessing
def preprocess_data(df):
    df['Returns'] = df['Close'].pct_change()
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df.dropna(inplace=True)
    return df

# 3. Feature Engineering
def create_features(df):
    df['Target'] = df['Close'].shift(-1)  # Next day's closing price
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 'SMA_5', 'SMA_20']
    X = df[features]
    y = df['Target']
    X = X[:-1]  # Remove the last row
    y = y[:-1]  # Remove the last row
    return X, y

# 4. Model Implementation
def train_knn_model(X_train, y_train, n_neighbors=5):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    knn = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn.fit(X_train_scaled, y_train)
    return knn, scaler

# 5. Model Evaluation
def evaluate_model(model, X_test, y_test, scaler):
    X_test_scaled = scaler.transform(X_test)
    predictions = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    return mse, mae, predictions

# 6. Visualization
def plot_predictions(y_test, predictions, ticker):
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test.values, label='Actual')
    plt.plot(y_test.index, predictions, label='Predicted')
    plt.title(f'Actual vs Predicted Stock Prices for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def plot_error_distribution(errors):
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True)
    plt.title('Distribution of Prediction Errors')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.show()

# Main simulation
def run_simulation():
    start_date = "2020-01-01"
    end_date = "2024-06-25"
    symbols = get_nifty100_symbols()
    results = {}

    for ticker in tqdm(symbols, desc="Processing stocks"):
        # Get and preprocess data
        df = get_stock_data(ticker, start_date, end_date)
        if df is None or len(df) < 100:
            print(f"Insufficient data for {ticker}. Skipping.")
            continue
        
        df = preprocess_data(df)
        X, y = create_features(df)

        # Use TimeSeriesSplit for a more realistic evaluation
        tscv = TimeSeriesSplit(n_splits=5)
        mse_scores = []
        mae_scores = []

        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # Train model
            knn_model, scaler = train_knn_model(X_train, y_train)

            # Evaluate model
            mse, mae, predictions = evaluate_model(knn_model, X_test, y_test, scaler)
            mse_scores.append(mse)
            mae_scores.append(mae)

        # Store results
        results[ticker] = {
            'MSE': np.mean(mse_scores),
            'MAE': np.mean(mae_scores)
        }

        # Final prediction using the entire dataset
        knn_model, scaler = train_knn_model(X, y)
        last_data_point = scaler.transform(X.iloc[-1].values.reshape(1, -1))
        next_day_prediction = knn_model.predict(last_data_point)
        results[ticker]['Next Day Prediction'] = next_day_prediction[0]

        # Visualize results for this stock
        _, _, final_predictions = evaluate_model(knn_model, X, y, scaler)
        plot_predictions(y, final_predictions, ticker)
        plot_error_distribution(y - final_predictions)

    # Print overall results
    print("\nOverall Results:")
    for ticker, metrics in results.items():
        print(f"\n{ticker}:")
        print(f"  Average MSE: {metrics['MSE']:.4f}")
        print(f"  Average MAE: {metrics['MAE']:.4f}")
        print(f"  Next Day Prediction: {metrics['Next Day Prediction']:.2f}")

if __name__ == "__main__":
    run_simulation()