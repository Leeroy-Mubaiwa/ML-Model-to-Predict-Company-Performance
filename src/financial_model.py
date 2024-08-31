import csv
import pandas as pd
import numpy as np
import json
import requests
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import os


def load_data():
    # Load datasets from the 'data' directory
    income_statements = pd.read_csv('data/income_statements.csv')
    balance_sheets = pd.read_csv('data/balance_sheets.csv')
    cash_flow_statements = pd.read_csv('data/cashflows.csv')

    return income_statements, balance_sheets, cash_flow_statements

def preprocess_data(income_statements, balance_sheets, cash_flow_statements):
    # Merge datasets based on Company Name and Year
    merged_data = pd.merge(income_statements, balance_sheets, on=['Company Name', 'Year'])
    merged_data = pd.merge(merged_data, cash_flow_statements, on=['Company Name', 'Year'])
    
    # Feature Engineering - Create new financial ratios
    merged_data['Profit Margin'] = merged_data['Net Income'] / merged_data['Revenue']    
    merged_data['Current Ratio'] = merged_data['Current Assets'] / merged_data['Current Liabilities']    
    merged_data['Debt-to-Equity Ratio'] = merged_data['Total Liabilities'] / merged_data['Shareholders\' Equity']
    merged_data['Cash Flow to Debt Ratio'] = merged_data['Operating Cash Flow'] / merged_data['Total Liabilities']
    merged_data['Return on Assets (ROA)'] = merged_data['Net Income'] / merged_data['Total Assets']
    
    # Handling missing values by filling with the median value
    numeric_cols = merged_data.select_dtypes(include='number').columns
    merged_data[numeric_cols] = merged_data[numeric_cols].fillna(merged_data[numeric_cols].median())
    
    return merged_data

def train_model(data, target_column):
    # Define features (X) and the target variable (y)
    features = ['Profit Margin', 'Current Ratio', 'Debt-to-Equity Ratio', 
                'Cash Flow to Debt Ratio', 'Return on Assets (ROA)']
    X = data[features]
    y = data[target_column]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train a Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\nModel Evaluation for {target_column}:\nMean Squared Error: {mse}\nR2 Score: {r2}")
    
    return model

def predict_current_year(model, historical_data):
    # Prepare feature data for prediction
    features = ['Profit Margin', 'Current Ratio', 'Debt-to-Equity Ratio', 
                'Cash Flow to Debt Ratio', 'Return on Assets (ROA)']
    X_current = historical_data[features].tail(1)  # Use the most recent year's data
    
    # Make prediction for the current year
    prediction = model.predict(X_current)
    return prediction

def main():
    income_statements, balance_sheets, cash_flow_statements = load_data() 
    processed_data = preprocess_data(income_statements, balance_sheets, cash_flow_statements)

    # Train models for different financial metrics
    income_model = train_model(processed_data, 'Net Income')
    equity_model = train_model(processed_data, 'Shareholders\' Equity')
    cash_flow_model = train_model(processed_data, 'Operating Cash Flow')

    # Predict current year financials
    income_prediction = predict_current_year(income_model, processed_data)
    equity_prediction = predict_current_year(equity_model, processed_data)
    cash_flow_prediction = predict_current_year(cash_flow_model, processed_data)

    # Print predictions
    print("\nPredicted Financials for Current Year:")
    print(f"Net Income: {income_prediction}")
    print(f"Shareholders' Equity: {equity_prediction}")
    print(f"Operating Cash Flow: {cash_flow_prediction}")

if __name__ == "__main__":
    main()
