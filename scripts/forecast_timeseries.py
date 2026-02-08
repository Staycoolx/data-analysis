#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
import holidays
import os

# Set style
plt.style.use('ggplot')
sns.set_palette("husl")
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def forecast_timeseries(file_path, target_col, datetime_col='Datetime', output_dir=None):
    """
    Perform time series analysis and forecasting using Prophet.
    Includes hourly trend, holiday analysis, and component decomposition.
    """
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    if output_dir is None:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_dir = os.path.join(os.getcwd(), f"{base_name}_forecast")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating forecast in: {output_dir}")

    # 1. Preprocessing
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.sort_values(datetime_col)

    # Prophet requires columns named 'ds' and 'y'
    prophet_df = df.rename(columns={datetime_col: 'ds', target_col: 'y'})

    # 2. Hourly Trend Analysis (EDA)
    print("Analyzing hourly trends...")
    df['Hour'] = df[datetime_col].dt.hour
    hourly_avg = df.groupby('Hour')[target_col].mean()

    plt.figure(figsize=(12, 6))
    hourly_avg.plot(kind='line', marker='o')
    plt.title('Average Hourly Energy Consumption')
    plt.ylabel(target_col)
    plt.xlabel('Hour of Day')
    plt.grid(True, alpha=0.3)
    plt.xticks(range(0, 24))
    plt.tight_layout()
    plt.savefig(f'{output_dir}/hourly_trend.png')

    # 3. Holiday Analysis
    print("Analyzing holiday effects...")
    us_holidays = holidays.US()
    df['Date'] = df[datetime_col].dt.date
    df['IsHoliday'] = df['Date'].apply(lambda x: x in us_holidays)

    holiday_stats = df.groupby('IsHoliday')[target_col].agg(['mean', 'count', 'std'])
    print("\nHoliday vs Non-Holiday Consumption:")
    print(holiday_stats)

    plt.figure(figsize=(8, 6))
    sns.boxplot(x='IsHoliday', y=target_col, data=df)
    plt.title('Energy Consumption: Holiday vs Non-Holiday')
    plt.xticks([0, 1], ['Non-Holiday', 'Holiday'])
    plt.tight_layout()
    plt.savefig(f'{output_dir}/holiday_effect.png')

    # 4. Prophet Modeling
    print("\nTraining Prophet model (this may take a minute)...")
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
    model.add_country_holidays(country_name='US')
    model.fit(prophet_df)

    # Forecast 30 days (720 hours)
    future = model.make_future_dataframe(periods=720, freq='H')
    forecast = model.predict(future)

    # Plot Components (Trend, Weekly, Daily, Yearly)
    print("Plotting forecast components...")
    fig1 = model.plot_components(forecast)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/forecast_components.png')

    # Plot Forecast
    fig2 = model.plot(forecast)
    plt.title(f'Forecast for {target_col}')
    plt.xlabel('Date')
    plt.ylabel(target_col)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/forecast_plot.png')

    print(f"\nAnalysis complete. Results saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Time Series Forecasting with Prophet")
    parser.add_argument("file_path", help="Path to the CSV file")
    parser.add_argument("target_col", help="Target column to forecast")
    parser.add_argument("--datetime_col", default="Datetime", help="Name of datetime column")
    parser.add_argument("--output", "-o", help="Output directory")

    args = parser.parse_args()

    forecast_timeseries(args.file_path, args.target_col, args.datetime_col, args.output)
