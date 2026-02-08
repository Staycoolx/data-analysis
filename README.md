# Data Analysis Skill 📊

🇨🇳 [中文指南 (Chinese Guide)](README_CN.md)

A comprehensive data analysis toolkit for Claude Code, designed to automate the **D-D-S-P** (Describe, Diagnose, Segment, Predict) workflow.

> **Philosophy**: Don't just run code; think like an analyst. This skill guides you from "What happened?" to "What will happen?" using robust, automated scripts.

## 🌟 Core Features

### 1. Describe - `auto_eda.py`
**Goal**: Understand data quality, distribution, and basic stats.
- Auto-detects missing values and outliers.
- Generates distribution plots and correlation heatmaps.
- **Usage**: `python3 scripts/auto_eda.py data.csv --target target_col`

### 2. Diagnose - `analyze_drivers_optimized.py`
**Goal**: Identify key factors driving the target variable.
- Uses Random Forest Permutation Importance to rank drivers.
- Auto-detects Classification vs. Regression tasks.
- **Usage**: `python3 scripts/analyze_drivers_optimized.py data.csv target_col --output report`

### 3. Segment - `analyze_groups.py`
**Goal**: Compare performance across different cohorts.
- Auto-bins numerical variables (e.g., Age groups).
- Aggregates metrics (mean, sum, count) by group.
- **Usage**: `python3 scripts/analyze_groups.py data.csv group_col target_col`

### 4. Predict - `predict_target.py`
**Goal**: Forecast future outcomes or classify new data.
- Trains robust baseline models (Random Forest).
- Outputs predictions CSV and reusable model file (`.joblib`).
- **Usage**: `python3 scripts/predict_target.py data.csv target_col --output prediction`

### 5. Time Series - `forecast_timeseries_std.py`
**Goal**: Analyze trends, seasonality, and holiday effects.
- Decomposes time series into Trend and Seasonal components.
- Analyzes hourly patterns and weekend/holiday effects.
- **Usage**: `python3 scripts/forecast_timeseries_std.py time_series_data.csv value_col --datetime_col date_col`

---

## 🚀 Quick Start

### Prerequisites
This toolkit relies only on standard Python data science libraries to ensure stability:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels holidays joblib
```

### Installation
Clone this repository to your local machine:
```bash
git clone https://github.com/Staycoolx/data-analysis.git
```

### Standard Workflow

1.  **Check Data Quality**:
    ```bash
    python3 scripts/auto_eda.py your_data.csv --target target_col
    ```
2.  **Find Key Drivers**:
    ```bash
    python3 scripts/analyze_drivers_optimized.py your_data.csv target_col
    ```
3.  **Compare Segments**:
    ```bash
    python3 scripts/analyze_groups.py your_data.csv group_col target_col
    ```
4.  **Predict Future**:
    ```bash
    python3 scripts/predict_target.py your_data.csv target_col
    ```

---

## 🛠 Project Structure

```
data-analysis/
├── SKILL.md                 # Main guide for Claude
├── README.md                # English documentation
├── README_CN.md             # Chinese documentation
├── scripts/                 # Automation scripts
│   ├── auto_eda.py
│   ├── analyze_drivers.py
│   ├── analyze_groups.py
│   ├── predict_target.py
│   └── ...
└── references/              # Contextual guides
    ├── causal-inference.md
    └── code-templates.md
```

## 📝 License
MIT License
