#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Set style
plt.style.use('ggplot')
sns.set_palette("husl")
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def analyze_groups(file_path, group_col, target_col, agg_funcs=['mean', 'sum', 'count'], bins=None, top_n=10, output_dir=None):
    """
    Generic script to analyze a target variable grouped by another variable.
    Supports automatic binning for numerical group columns.
    """
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    if output_dir is None:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_dir = os.path.join(os.getcwd(), f"{base_name}_group_analysis")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating group analysis in: {output_dir}")

    report_file = os.path.join(output_dir, "Group_Analysis_Report.md")

    # 1. Data Preprocessing
    # Handle missing values in group_col
    df = df.dropna(subset=[group_col])

    # Check if group_col is numeric and needs binning
    is_numeric_group = pd.api.types.is_numeric_dtype(df[group_col])
    original_group_col = group_col

    if is_numeric_group and df[group_col].nunique() > 20:
        if bins is None:
            bins = 10
        print(f"Binning numeric column '{group_col}' into {bins} bins...")
        group_col = f"{original_group_col}_binned"
        try:
            df[group_col] = pd.cut(df[original_group_col], bins=int(bins))
        except:
             df[group_col] = pd.cut(df[original_group_col], bins=int(bins), duplicates='drop')

    # 2. Group & Aggregate
    print(f"Grouping by '{group_col}' and aggregating '{target_col}'...")

    # Map string agg_funcs to actual functions if needed, or rely on pandas string aliases
    grouped = df.groupby(group_col)[target_col].agg(agg_funcs).reset_index()

    # Rename columns for clarity
    grouped.columns = [group_col] + [f"{target_col}_{func}" for func in agg_funcs]

    # Sort by the first aggregation metric (usually mean or sum)
    sort_metric = grouped.columns[1]
    grouped_sorted = grouped.sort_values(sort_metric, ascending=False)

    # 3. Generate Report
    with open(report_file, 'w') as f:
        f.write(f"# Group Analysis: {target_col} by {original_group_col}\n\n")

        f.write("## 1. Top Groups\n\n")
        f.write(f"Top {top_n} groups sorted by {sort_metric}:\n\n")
        f.write("```\n" + grouped_sorted.head(top_n).to_string(index=False) + "\n```\n\n")

        f.write("## 2. Bottom Groups\n\n")
        f.write(f"Bottom {top_n} groups sorted by {sort_metric}:\n\n")
        f.write("```\n" + grouped_sorted.tail(top_n).to_string(index=False) + "\n```\n\n")

    # 4. Visualization
    # Plot top N groups
    plt.figure(figsize=(12, 6))

    # If too many groups, just show top N and bottom N
    if len(grouped_sorted) > 30:
        plot_data = pd.concat([grouped_sorted.head(top_n), grouped_sorted.tail(top_n)])
    else:
        plot_data = grouped_sorted

    sns.barplot(x=group_col, y=sort_metric, data=plot_data, palette='viridis')
    plt.title(f'{target_col} ({sort_metric.split("_")[-1]}) by {original_group_col}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "group_analysis.png"))
    plt.close()

    print(f"Group analysis complete: {report_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generic Group & Aggregate Analysis")
    parser.add_argument("file_path", help="Path to the CSV file")
    parser.add_argument("group_col", help="Column to group by (e.g., Region, Category)")
    parser.add_argument("target_col", help="Target column to aggregate (e.g., Sales, Price)")
    parser.add_argument("--agg", nargs='+', default=['mean', 'sum', 'count'], help="Aggregation functions (mean, sum, count, min, max)")
    parser.add_argument("--bins", type=int, default=10, help="Number of bins for numeric group columns")
    parser.add_argument("--top", type=int, default=10, help="Number of top groups to show")
    parser.add_argument("--output", "-o", help="Output directory for report")

    args = parser.parse_args()

    analyze_groups(args.file_path, args.group_col, args.target_col, args.agg, args.bins, args.top, args.output)
