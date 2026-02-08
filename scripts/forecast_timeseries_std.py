#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from statsmodels.tsa.seasonal import seasonal_decompose

# Set style
plt.style.use('ggplot')
sns.set_palette("husl")
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def forecast_timeseries_std(file_path, target_col, datetime_col='Datetime', output_dir=None):
    """
    Perform time series analysis using standard libraries (pandas, statsmodels).
    Includes hourly trend, holiday analysis, and STL decomposition.
    """
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    if output_dir is None:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_dir = os.path.join(os.getcwd(), f"{base_name}_forecast_std")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating analysis in: {output_dir}")

    # 1. Preprocessing
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.sort_values(datetime_col).set_index(datetime_col)

    # Handle duplicates if any
    if df.index.duplicated().any():
        print("Warning: Duplicate timestamps found. Taking mean.")
        df = df.groupby(level=0).mean()

    # Set frequency (Hourly)
    df = df.asfreq('H', method='ffill')

    # 2. Hourly Trend Analysis
    print("Analyzing hourly trends...")
    df['Hour'] = df.index.hour
    hourly_avg = df.groupby('Hour')[target_col].mean()

    plt.figure(figsize=(12, 6))
    hourly_avg.plot(kind='line', marker='o', linewidth=2, color='#1f77b4')
    plt.title('Average Hourly Energy Consumption Profile')
    plt.ylabel('MW')
    plt.xlabel('Hour of Day (0-23)')
    plt.grid(True, alpha=0.3)
    plt.xticks(range(0, 24))

    # Highlight peaks
    peak_hour = hourly_avg.idxmax()
    peak_val = hourly_avg.max()
    plt.annotate(f'Peak: {peak_hour}:00', xy=(peak_hour, peak_val), xytext=(peak_hour, peak_val+500),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    plt.tight_layout()
    plt.savefig(f'{output_dir}/hourly_trend.png')

    # 3. Weekend Analysis (Proxy for Holiday)
    print("Analyzing weekend effects...")
    # 0=Monday, 5=Saturday, 6=Sunday
    df['IsWeekend'] = df.index.weekday >= 5

    weekend_mean = df[df['IsWeekend']][target_col].mean()
    weekday_mean = df[~df['IsWeekend']][target_col].mean()

    print(f"\nWeekend Avg: {weekend_mean:.2f} MW")
    print(f"Weekday Avg: {weekday_mean:.2f} MW")
    print(f"Diff: {(weekend_mean - weekday_mean):.2f} MW ({((weekend_mean - weekday_mean)/weekday_mean)*100:.1f}%)")

    plt.figure(figsize=(8, 6))
    sns.boxplot(x='IsWeekend', y=target_col, data=df, palette='Set2')
    plt.title('Energy Consumption: Weekend vs Weekday')
    plt.xticks([0, 1], ['Weekday', 'Weekend'])
    plt.ylabel('MW')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/weekend_effect.png')

    # 4. Decomposition (Trend & Seasonality)
    print("\nDecomposing time series (Trend/Seasonal/Resid)...")
    # Decompose using additive model
    decomposition = seasonal_decompose(df[target_col], model='additive', period=24*365) # Yearly seasonality

    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)

    decomposition.observed.plot(ax=axes[0], legend=False)
    axes[0].set_ylabel('Observed')
    axes[0].set_title('Time Series Decomposition')

    decomposition.trend.plot(ax=axes[1], legend=False, color='orange')
    axes[1].set_ylabel('Trend (Long-term)')

    decomposition.seasonal.plot(ax=axes[2], legend=False, color='green')
    axes[2].set_ylabel('Seasonal (Yearly)')

    decomposition.resid.plot(ax=axes[3], legend=False, color='grey', alpha=0.5)
    axes[3].set_ylabel('Residual')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/decomposition.png')

    # 5. Weekly Pattern
    print("Analyzing weekly patterns...")
    df['Weekday'] = df.index.weekday
    df['DayName'] = df.index.day_name()
    weekly_avg = df.groupby('DayName')[target_col].mean().reindex(
        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    )

    plt.figure(figsize=(10, 6))
    weekly_avg.plot(kind='bar', color='#2ca02c', alpha=0.8)
    plt.title('Average Weekly Energy Consumption')
    plt.ylabel('MW')
    plt.ylim(weekly_avg.min() * 0.9, weekly_avg.max() * 1.05) # Zoom in
    plt.tight_layout()
    plt.savefig(f'{output_dir}/weekly_trend.png')

    print(f"\nAnalysis complete. Results saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Time Series Analysis with Statsmodels")
    parser.add_argument("file_path", help="Path to the CSV file")
    parser.add_argument("target_col", help="Target column to analyze")
    parser.add_argument("--datetime_col", default="Datetime", help="Name of datetime column")
    parser.add_argument("--output", "-o", help="Output directory")

    args = parser.parse_args()

    forecast_timeseries_std(args.file_path, args.target_col, args.datetime_col, args.output)
