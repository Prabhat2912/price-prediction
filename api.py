from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from datetime import datetime

app = Flask(__name__)

# Load historical data and train simulation models (simplified from notebook)
def load_and_train_models():
    try:
        # Load data (adjust paths as needed)
        df1 = pd.read_csv("soapnutshistory.csv")
        df2 = pd.read_csv("woolballhistory.csv")
        df = pd.concat([df1, df2], ignore_index=True)
        df['Report Date'] = pd.to_datetime(df['Report Date'])
        df = df.sort_values('Report Date')
        df['Product Price'] = df['Product Price'].fillna(df['Product Price'].median())
        df['Organic Conversion Percentage'] = df['Organic Conversion Percentage'].fillna(0)
        df['Ad Conversion Percentage'] = df['Ad Conversion Percentage'].fillna(0)
        df['Total Sales'] = df['Total Sales'].fillna(0)
        df['Predicted Sales'] = df['Predicted Sales'].fillna(df['Total Sales'].mean())

        # Compute medians
        median_sales = df['Total Sales'].median()
        median_org_conv = df['Organic Conversion Percentage'].median()
        median_ad_conv = df['Ad Conversion Percentage'].median()
        median_price = df['Product Price'].median()

        # Train simulation models
        historical_df = df[df['Total Sales'].notnull()].copy()
        historical_df['day_of_week'] = historical_df['Report Date'].dt.weekday
        historical_df['month'] = historical_df['Report Date'].dt.month
        df_features = pd.get_dummies(historical_df[['Product Price', 'day_of_week', 'month']], 
                                     columns=['day_of_week', 'month'])
        
        sales_model = LinearRegression().fit(df_features, historical_df['Total Sales'])
        org_conv_model = LinearRegression().fit(df_features, historical_df['Organic Conversion Percentage'])
        ad_conv_model = LinearRegression().fit(df_features, historical_df['Ad Conversion Percentage'])

        return sales_model, org_conv_model, ad_conv_model, df_features.columns, median_sales, median_org_conv, median_ad_conv, median_price
    except Exception as e:
        print(f"Error loading data or training models: {e}")
        return None, None, None, None, None, None, None, None

# Simulation functions
def simulate_sales(price, date, sales_model, feature_columns):
    features = pd.DataFrame({'Product Price': [price], 'day_of_week': [date.weekday()], 'month': [date.month]})
    features = pd.get_dummies(features, columns=['day_of_week', 'month'])
    for col in feature_columns:
        if col not in features.columns:
            features[col] = 0
    features = features[feature_columns]
    return max(0, sales_model.predict(features)[0])

def simulate_org_conv(price, date, org_conv_model, feature_columns):
    features = pd.DataFrame({'Product Price': [price], 'day_of_week': [date.weekday()], 'month': [date.month]})
    features = pd.get_dummies(features, columns=['day_of_week', 'month'])
    for col in feature_columns:
        if col not in features.columns:
            features[col] = 0
    features = features[feature_columns]
    return min(max(org_conv_model.predict(features)[0], 0), 100)

def simulate_ad_conv(price, date, ad_conv_model, feature_columns):
    features = pd.DataFrame({'Product Price': [price], 'day_of_week': [date.weekday()], 'month': [date.month]})
    features = pd.get_dummies(features, columns=['day_of_week', 'month'])
    for col in feature_columns:
        if col not in features.columns:
            features[col] = 0
    features = features[feature_columns]
    return min(max(ad_conv_model.predict(features)[0], 0), 100)

# Load Q-table and simulation models once when the app starts
def load_q_table():
    try:
        with open("q_table.pkl", "rb") as f:
            q_table = pickle.load(f)
        print(f"Q-table Loaded: Shape = {q_table.shape}")
        return q_table
    except FileNotFoundError:
        print("Error: Q-table file 'q_table.pkl' not found. Please train the model first.")
        return None
    except Exception as e:
        print(f"Error loading Q-table: {e}")
        return None

# Price mapping
price_min, price_max, price_step = 10, 20, 0.5
possible_prices = np.arange(price_min, price_max + price_step, price_step)
price_to_index = {p: i for i, p in enumerate(possible_prices)}

# Load resources
q_table = load_q_table()
sales_model, org_conv_model, ad_conv_model, feature_columns, median_sales, median_org_conv, median_ad_conv, median_price = load_and_train_models()

def discretize_price(price):
    price = max(price_min, min(price_max, price))
    return round(price / price_step) * price_step

# Prediction function
def predict_price(state, q_table, date=None):
    if q_table is None:
        return None
    try:
        state = float(state)
        state_price = discretize_price(state)
        state_idx = price_to_index[state_price]
        action_idx = np.argmax(q_table[state_idx])
        predicted_price = possible_prices[action_idx]
        return predicted_price
    except Exception as e:
        print(f"Error in predict_price: {e}")
        return None

@app.route('/predict', methods=['GET', 'POST'])
def predict_api():
    if request.method == 'POST':
        data = request.get_json()
        if data is None or 'price' not in data:
            return jsonify({"error": "No price provided in JSON payload"}), 400
        state = data['price']
        date_str = data.get('date', '2025-02-20')  # Default date if not provided
    else:
        state = request.args.get('price', None)
        date_str = request.args.get('date', '2025-02-20')  # Default date if not provided
        if state is None:
            return jsonify({"error": "Price parameter is missing in query string"}), 400

    try:
        state = float(state)
    except ValueError:
        return jsonify({"error": "Invalid price value, must be numeric"}), 400

    if q_table is None or sales_model is None:
        return jsonify({"error": "Model resources not loaded"}), 500

    try:
        date = datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        return jsonify({"error": "Invalid date format, use YYYY-MM-DD"}), 400

    predicted_price = predict_price(state, q_table, date)
    if predicted_price is None:
        return jsonify({"error": "Prediction failed"}), 500

    return jsonify({
        "input_price": state,
        "predicted_price": float(predicted_price)  # Ensure JSON compatibility
    })

if __name__ == '__main__':
    app.run(debug=True)