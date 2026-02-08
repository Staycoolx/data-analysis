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

def analyze_data(file_path, output_dir=None, target_col=None):
    """
    Perform automatic EDA on a dataset.
    """
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    if output_dir is None:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_dir = os.path.join(os.getcwd(), f"{base_name}_eda_report")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating report in: {output_dir}")

    report_file = os.path.join(output_dir, "EDA_Report.md")

    with open(report_file, 'w') as f:
        f.write(f"# EDA Report: {os.path.basename(file_path)}\n\n")

        # 1. Basic Info
        f.write("## 1. Data Overview\n\n")
        f.write(f"- **Rows**: {df.shape[0]:,}\n")
        f.write(f"- **Columns**: {df.shape[1]}\n")
        f.write(f"- **Missing Values**: {df.isnull().sum().sum():,}\n")
        f.write(f"- **Duplicate Rows**: {df.duplicated().sum():,} ({df.duplicated().mean()*100:.2f}%)\n\n")

        f.write("### Column Types\n")
        f.write("```\n" + df.dtypes.to_string() + "\n```\n\n")

        # 2. Descriptive Statistics
        f.write("## 2. Descriptive Statistics\n\n")
        f.write("```\n" + df.describe().T.to_string() + "\n```\n\n")

        # 3. Target Variable Analysis (if provided)
        if target_col and target_col in df.columns:
            f.write(f"## 3. Target Variable Analysis: `{target_col}`\n\n")

            # Check for imbalance
            val_counts = df[target_col].value_counts()
            f.write("### Class Distribution\n")
            f.write("```\n" + val_counts.to_frame().to_string() + "\n```\n\n")

            # Plot
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            sns.countplot(x=target_col, data=df)
            plt.title(f'Distribution of {target_col}')

            plt.subplot(1, 2, 2)
            plt.pie(val_counts, labels=val_counts.index, autopct='%1.1f%%')
            plt.title(f'Percentage of {target_col}')

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "target_distribution.png"))
            plt.close()

            f.write("![Target Distribution](target_distribution.png)\n\n")

            # Imbalance warning
            min_class_pct = val_counts.min() / len(df)
            if min_class_pct < 0.1:
                f.write("> ⚠️ **WARNING**: Extreme class imbalance detected. "
                       f"Minority class is only {min_class_pct*100:.2f}%. "
                       "Consider using SMOTE, class weights, or focal loss for modeling.\n\n")

        # 4. Correlation Analysis (Numerical)
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 1:
            f.write("## 4. Correlation Analysis\n\n")

            # Calculate correlation matrix
            corr = df[num_cols].corr()

            # Plot heatmap
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr, annot=False, cmap='coolwarm', center=0)
            plt.title('Correlation Heatmap')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
            plt.close()

            f.write("![Correlation Heatmap](correlation_heatmap.png)\n\n")

            # Top correlations
            f.write("### Top Correlations\n")
            corr_unstack = corr.abs().unstack()
            corr_unstack = corr_unstack[corr_unstack < 1].sort_values(ascending=False)
            top_corr = corr_unstack.head(10).to_frame(name='Correlation')
            f.write("```\n" + top_corr.to_string() + "\n```\n\n")

        # 5. Distribution Plots (Top 6 numerical columns by variance)
        if len(num_cols) > 0:
            f.write("## 5. Numerical Distributions (Top Variables)\n\n")

            # Select top columns by variance to avoid plotting too many
            variances = df[num_cols].var().sort_values(ascending=False)
            top_vars = variances.head(6).index

            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()

            for i, col in enumerate(top_vars):
                if i < len(axes):
                    sns.histplot(data=df, x=col, kde=True, ax=axes[i])
                    axes[i].set_title(f'Distribution of {col}')

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "numerical_distributions.png"))
            plt.close()

            f.write("![Numerical Distributions](numerical_distributions.png)\n\n")

    print(f"Report generated successfully: {report_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated EDA Script")
    parser.add_argument("file_path", help="Path to the CSV file")
    parser.add_argument("--output", "-o", help="Output directory for report")
    parser.add_argument("--target", "-t", help="Target column name for supervised analysis")

    args = parser.parse_args()

    analyze_data(args.file_path, args.output, args.target)
