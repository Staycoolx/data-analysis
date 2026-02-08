#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, r2_score, mean_absolute_error, accuracy_score
import joblib

def predict_target(file_path, target_col, output_dir=None, task_type='auto', save_model=True):
    """
    Train a predictive model and generate predictions.
    Outputs:
    1. Evaluation metrics
    2. Predictions CSV (Original Data + Prediction + Probability)
    3. Trained Model (.joblib)
    """
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    if output_dir is None:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_dir = os.path.join(os.getcwd(), f"{base_name}_prediction")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating predictions in: {output_dir}")

    report_file = os.path.join(output_dir, "Prediction_Report.md")

    # 1. Data Preprocessing
    # Drop ID-like columns
    nunique = df.nunique()
    drop_cols = nunique[nunique <= 1].index.tolist()
    if 'EmployeeNumber' in df.columns: drop_cols.append('EmployeeNumber')
    if 'id' in df.columns.str.lower(): drop_cols.extend(df.columns[df.columns.str.lower() == 'id'].tolist())

    # Drop high-cardinality categorical columns (likely IDs or text)
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() > 100 and col != target_col:
             drop_cols.append(col)

    df_clean = df.drop(columns=drop_cols, errors='ignore')
    print(f"Dropped columns: {drop_cols}")

    # Drop rows with missing target
    df_clean = df_clean.dropna(subset=[target_col])

    # Separate Target
    y = df_clean[target_col]
    X = df_clean.drop(columns=[target_col])

    # Detect task type
    if task_type == 'auto':
        if y.dtype == 'object' or y.nunique() < 20:
            task_type = 'classification'
        else:
            task_type = 'regression'

    print(f"Task Type: {task_type.capitalize()}")

    # Feature Encoding
    le_dict = {}
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        le_dict[col] = le

    # Simple Imputation
    X = X.fillna(X.median(numeric_only=True)).fillna(0)

    # Target Encoding (for Classification)
    if task_type == 'classification' and y.dtype == 'object':
        le_target = LabelEncoder()
        y = le_target.fit_transform(y)
        target_classes = le_target.classes_
    else:
        target_classes = None

    # 2. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Modeling
    if task_type == 'classification':
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=1)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)

    print("Training model...")
    model.fit(X_train, y_train)

    # 4. Evaluation
    y_pred = model.predict(X_test)

    with open(report_file, 'w') as f:
        f.write(f"# Prediction Report: {target_col}\n\n")
        f.write(f"**Task Type**: {task_type.capitalize()}\n\n")

        f.write("## Model Performance (Test Set)\n\n")
        if task_type == 'classification':
            acc = accuracy_score(y_test, y_pred)
            f.write(f"- **Accuracy**: {acc:.4f}\n\n")
            f.write("```\n" + classification_report(y_test, y_pred) + "\n```\n")
        else:
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            f.write(f"- **R² Score**: {r2:.4f}\n")
            f.write(f"- **MAE**: {mae:.4f}\n")

    # 5. Output Predictions (Full Dataset)
    print("Generating predictions for full dataset...")
    full_pred = model.predict(X)

    # Create result dataframe
    result_df = df.copy() # Use original df to keep IDs

    if task_type == 'classification' and target_classes is not None:
        result_df['Predicted_Value'] = [target_classes[i] for i in full_pred]
        # Add probabilities if classification
        probs = model.predict_proba(X)
        result_df['Prediction_Confidence'] = np.max(probs, axis=1)
    else:
        result_df['Predicted_Value'] = full_pred

    csv_path = os.path.join(output_dir, "predictions.csv")
    result_df.to_csv(csv_path, index=False)
    print(f"Predictions saved to: {csv_path}")

    # 6. Save Model
    if save_model:
        model_path = os.path.join(output_dir, "model.joblib")
        joblib.dump(model, model_path)
        print(f"Model saved to: {model_path}")

    print(f"Prediction task complete: {report_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a predictive model and output predictions")
    parser.add_argument("file_path", help="Path to the CSV file")
    parser.add_argument("target_col", help="Target column to predict")
    parser.add_argument("--output", "-o", help="Output directory")
    parser.add_argument("--type", choices=['auto', 'classification', 'regression'], default='auto')

    args = parser.parse_args()

    predict_target(args.file_path, args.target_col, args.output, args.type)
