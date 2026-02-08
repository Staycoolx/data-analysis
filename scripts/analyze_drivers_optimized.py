#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, r2_score, mean_absolute_error

# Set style
plt.style.use('ggplot')
sns.set_palette("husl")
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def analyze_drivers(file_path, target_col, output_dir=None, task_type='auto'):
    """
    Analyze key drivers of a target variable using Random Forest Permutation Importance.
    """
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    if output_dir is None:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_dir = os.path.join(os.getcwd(), f"{base_name}_driver_analysis")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating driver analysis in: {output_dir}")

    report_file = os.path.join(output_dir, "Driver_Analysis_Report.md")

    # 1. Data Preprocessing
    # Drop useless columns (ID, constant columns)
    nunique = df.nunique()
    drop_cols = nunique[nunique <= 1].index.tolist()
    if 'EmployeeNumber' in df.columns: drop_cols.append('EmployeeNumber')
    if 'id' in df.columns.str.lower(): drop_cols.extend(df.columns[df.columns.str.lower() == 'id'].tolist())

    # Drop text columns that are not categorical (too many unique values)
    # But keep neighbourhood as it is important
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() > 100 and col not in ['neighbourhood', target_col]:
             drop_cols.append(col)

    df_clean = df.drop(columns=drop_cols, errors='ignore')
    print(f"Dropped constant/ID/High-cardinality columns: {drop_cols}")

    # Drop rows with missing target
    df_clean = df_clean.dropna(subset=[target_col])

    # Encode target
    y = df_clean[target_col]
    X = df_clean.drop(columns=[target_col])

    # Detect task type if auto
    if task_type == 'auto':
        if y.dtype == 'object' or y.nunique() < 20:
            task_type = 'classification'
        else:
            task_type = 'regression'

    print(f"Task Type: {task_type}")

    # Encode features
    le_dict = {}
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        # Handle new categories in future if needed, but for now fit_transform is fine
        X[col] = le.fit_transform(X[col].astype(str))
        le_dict[col] = le

    # Handle NaNs in features (simple fill)
    X = X.fillna(X.median(numeric_only=True)).fillna(0)

    # Encode target for classification
    if task_type == 'classification' and y.dtype == 'object':
        le_target = LabelEncoder()
        y = le_target.fit_transform(y)
        print(f"Target classes: {le_target.classes_}")

    # 2. Modeling (Random Forest)
    # Use fewer estimators and limit depth for faster execution on large datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if task_type == 'classification':
        model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, class_weight='balanced', n_jobs=1)
    else:
        model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=1)

    print("Training model...")
    model.fit(X_train, y_train)

    # 3. Evaluation
    y_pred = model.predict(X_test)

    with open(report_file, 'w') as f:
        f.write(f"# Driver Analysis Report: {target_col}\n\n")
        f.write(f"**Task Type**: {task_type.capitalize()}\n\n")

        f.write("## 1. Model Performance\n\n")
        if task_type == 'classification':
            report = classification_report(y_test, y_pred)
            f.write("```\n" + report + "\n```\n\n")
        else:
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            f.write(f"- **R² Score**: {r2:.4f}\n")
            f.write(f"- **MAE**: {mae:.4f}\n\n")

        # 4. Feature Importance (Permutation)
        print("Calculating permutation importance (this may take a moment)...")
        # Optimized: n_jobs=1, n_repeats=3
        result = permutation_importance(model, X_test, y_test, n_repeats=3, random_state=42, n_jobs=1)

        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': result.importances_mean,
            'Std': result.importances_std
        }).sort_values('Importance', ascending=False)

        f.write("## 2. Key Drivers (Permutation Importance)\n\n")
        f.write("Top factors that influence the target variable:\n\n")
        f.write("```\n" + importance_df.head(15).to_string(index=False) + "\n```\n\n")

        # Plot Importance
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(15), palette='viridis')
        plt.title(f'Top Drivers of {target_col}')
        plt.xlabel('Importance (Model Performance Drop)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "feature_importance.png"))
        plt.close()
        f.write("![Feature Importance](feature_importance.png)\n\n")

        # 5. Top Features Visualization
        f.write("## 3. Top Features Analysis\n\n")
        top_features = importance_df.head(4)['Feature'].tolist()

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()

        for i, feature in enumerate(top_features):
            plot_data = df_clean.copy()

            # Limit plot data size for performance
            if len(plot_data) > 2000:
                plot_data = plot_data.sample(2000, random_state=42)

            if task_type == 'regression':
                sns.scatterplot(x=feature, y=target_col, data=plot_data, ax=axes[i], alpha=0.6)
                axes[i].set_title(f'{feature} vs {target_col}')
            else:
                if pd.api.types.is_numeric_dtype(plot_data[feature]) and plot_data[feature].nunique() > 10:
                    sns.boxplot(x=target_col, y=feature, data=plot_data, ax=axes[i])
                    axes[i].set_title(f'{feature} Distribution by {target_col}')
                else:
                    ct = pd.crosstab(plot_data[feature], plot_data[target_col], normalize='index')
                    ct.plot(kind='bar', stacked=True, ax=axes[i], colormap='viridis')
                    axes[i].set_title(f'{feature} vs {target_col}')
                    axes[i].set_ylabel('Percentage')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "top_drivers_analysis.png"))
        plt.close()
        f.write("![Top Drivers Analysis](top_drivers_analysis.png)\n\n")

    print(f"Driver analysis complete: {report_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated Key Driver Analysis")
    parser.add_argument("file_path", help="Path to the CSV file")
    parser.add_argument("target_col", help="Target column to analyze")
    parser.add_argument("--output", "-o", help="Output directory for report")
    parser.add_argument("--type", choices=['auto', 'classification', 'regression'], default='auto', help="Task type")

    args = parser.parse_args()

    analyze_drivers(args.file_path, args.target_col, args.output, args.type)
