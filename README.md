# Data Analysis Skill 📊

🇨🇳 [中文指南 (Chinese Guide)](README_CN.md)

A full-stack data analysis and reporting toolkit for Claude Code. Combines the **D-D-S-P analysis pipeline** (Describe, Diagnose, Segment, Predict) with **multi-expert deep analysis** and **professional HTML/PPT report output**.

> **Philosophy**: Think first, then execute. Don't just run scripts — reason like an analyst. Complete the task, then proactively surface what the user might have missed.

---

## 🌟 What's New

This skill now integrates two workflows into one:

| Workflow | When to Use | Output |
|----------|-------------|--------|
| **D-D-S-P Pipeline** | Exploratory analysis, quick insights | Terminal / Markdown |
| **Multi-Expert Deep Analysis** | Formal reports, >500 rows or >10 fields | HTML Report / PPTX |

---

## 🔄 Decision Flow

```
Receive data
    ↓
Need to read Excel/PPTX? → Yes → scripts/read_excel.py / read_pptx.py
    ↓
Phase 1: D-D-S-P Analysis Pipeline (script-driven)
    ↓
Need a formal report output?
    ├─ No (quick explore) → Terminal / Markdown output
    └─ Yes (report/deep analysis) → Phase 2: Multi-Expert Analysis → Phase 3: HTML / PPT
```

---

## 🛠 Analysis Scripts

### Phase 1: D-D-S-P Pipeline

#### 1. Describe — `auto_eda.py`
Understand data quality, distribution, and basic statistics.
- Auto-detects missing values and outliers
- Generates distribution plots and correlation heatmaps
```bash
python3 scripts/auto_eda.py data.csv --output report_dir --target target_col
```

#### 2. Diagnose — `analyze_drivers_optimized.py`
Identify key factors driving the target variable.
- Random Forest Permutation Importance ranking
- Auto-detects Classification vs. Regression tasks
```bash
python3 scripts/analyze_drivers_optimized.py data.csv target_col --output driver_report
```

#### 3. Segment — `analyze_groups.py`
Compare performance across different cohorts.
- Auto-bins numerical variables (Age, Price, etc.)
- Aggregates metrics (mean, sum, count) by group
```bash
python3 scripts/analyze_groups.py data.csv group_col target_col --agg mean sum count
```

#### 4. Predict — `predict_target.py`
Forecast future outcomes or classify new data.
- Trains Random Forest baseline models
- Outputs predictions CSV and reusable `.joblib` model
```bash
python3 scripts/predict_target.py data.csv target_col --output prediction_result
```

#### 5. Time Series — `forecast_timeseries_std.py`
Analyze trends, seasonality, and holiday effects.
- Decomposes series into Trend + Seasonal components
- Analyzes hourly patterns and weekend/holiday effects
```bash
python3 scripts/forecast_timeseries_std.py data.csv value_col --datetime_col date_col
```

### Data Readers

| Script | Purpose |
|--------|---------|
| `read_excel.py` | Read Excel files → markdown / CSV / JSON |
| `read_pptx.py` | Read PPTX structure and content |

---

## 📊 Phase 2: Multi-Expert Deep Analysis

Triggered when data has >500 rows, >10 fields, or user requests a formal report.

**Four-stage workflow:**

1. **Data Understanding** — Dimensions, time range, field list, initial insights
2. **Expert Selection** — Choose 3–5 complementary expert personas (quantitative / strategic / risk / behavioral)
3. **Parallel Analysis** — Each expert runs as an independent subagent simultaneously
4. **Unified Synthesis** — A senior analyst perspective integrates all findings into the final report (no expert names appear)

---

## 🎨 Phase 3: Report Output

### HTML Reports (default)

11 styles to choose from — randomly selected when not specified:

**Classic:** Financial Times · McKinsey · The Economist · Goldman Sachs · Swiss/NZZ

**Design:** Stamen Design · Fathom · Sagmeister & Walsh · Takram · Irma Boom · Build

Style parameters → `references/report-style-gallery.md`

**Layout baseline (always enforced):**
```css
body { max-width: 1200px; margin: 0 auto; padding: 40px 48px; }
```

### PPT Output

```bash
node scripts/html2pptx.js slides.html output.pptx   # single file
node scripts/build_pptx.js                           # multi-page merge
```

PPT styles → `references/visual-design-system.md`

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels holidays joblib
node --version  # Node.js required for PPT conversion
```

### Installation
```bash
git clone https://github.com/Staycoolx/data-analysis.git
```

---

## 📁 Project Structure

```
data-analysis/
├── SKILL.md                      # Claude Code skill guide
├── README.md                     # English documentation
├── README_CN.md                  # Chinese documentation
├── scripts/
│   ├── auto_eda.py               # D: Describe
│   ├── analyze_drivers_optimized.py  # D: Diagnose
│   ├── analyze_groups.py         # S: Segment
│   ├── predict_target.py         # P: Predict
│   ├── forecast_timeseries_std.py    # Time series
│   ├── read_excel.py             # Excel reader
│   ├── read_pptx.py              # PPTX reader
│   ├── html2pptx.js              # HTML → PPTX converter
│   └── build_pptx.js             # Multi-page PPTX builder
└── references/
    ├── causal-inference.md       # PSM / DID / RDD methods
    ├── code-templates.md         # Reusable code patterns
    ├── report-style-gallery.md   # 11 HTML report styles
    ├── html-templates.md         # HTML visualization components
    ├── visual-design-system.md   # PPT design system
    ├── workflows.md              # Detailed workflow specs
    └── ad-analytics.md           # Ad / marketing analytics
```

## 📝 License
MIT License
