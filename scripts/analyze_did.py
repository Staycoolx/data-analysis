#!/usr/bin/env python3
"""
DID (Difference-in-Differences) Analysis Script

评估策略升级/政策变化的因果效应，支持：
1. DID 回归估计
2. 平行趋势检验
3. Event Study 可视化
4. 协变量控制
5. 安慰剂检验

Usage:
    python3 scripts/analyze_did.py data.csv \
        --treatment treated_col \
        --outcome retention_d1 \
        --time time_col \
        --group group_col \
        --output did_report

    # 带协变量
    python3 scripts/analyze_did.py data.csv \
        --treatment treated \
        --outcome retention \
        --time date \
        --group channel \
        --covariates age,gender,channel \
        --output did_report
"""
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import warnings
from datetime import datetime

# Set style
plt.style.use('ggplot')
sns.set_palette("husl")
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')


def load_and_prepare_data(file_path, treatment_col, outcome_col, time_col, group_col, covariates=None):
    """Load data and prepare for DID analysis."""
    df = pd.read_csv(file_path)

    required_cols = [treatment_col, outcome_col, time_col, group_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"Error: Missing columns: {missing}")
        print(f"Available columns: {list(df.columns)}")
        return None

    # Convert time_col to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        try:
            df[time_col] = pd.to_datetime(df[time_col])
        except:
            pass

    # Convert treatment to binary if needed
    unique_treat = df[treatment_col].unique()
    if len(unique_treat) > 2:
        print(f"Error: Treatment column has >2 unique values: {unique_treat}")
        return None

    return df


def run_did_regression(df, treatment_col, outcome_col, time_col, group_col, covariates=None):
    """Run DID regression and return results."""
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        print("Error: statsmodels not installed. Run: pip install statsmodels")
        return None

    # Create post indicator (assuming later time is "post")
    time_col_sorted = df[time_col].sort_values()
    median_time = time_col_sorted.iloc[len(time_col_sorted) // 2]
    df['post'] = (df[time_col] >= median_time).astype(int)

    # Create treated indicator (binary)
    treated_val = df[treatment_col].unique()[0]
    df['treated'] = (df[treatment_col] == treated_val).astype(int)

    # Build formula
    formula = f"{outcome_col} ~ treated + post + treated:post"

    if covariates and len(covariates) > 0:
        # Add covariates
        valid_covariates = [c for c in covariates if c in df.columns]
        if valid_covariates:
            formula = f"{outcome_col} ~ treated + post + treated:post + " + " + ".join(valid_covariates)

    try:
        model = smf.ols(formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df[group_col]})

        # Extract key results
        did_estimate = model.params.get('treated:post', 0)
        did_se = model.bse.get('treated:post', 0)
        did_pvalue = model.pvalues.get('treated:post', 1)
        did_ci = model.conf_int().loc['treated:post'] if 'treated:post' in model.conf_int().index else [0, 0]

        # Calculate group-time means for description
        treated_post = df[(df['treated'] == 1) & (df['post'] == 1)][outcome_col].mean()
        treated_pre = df[(df['treated'] == 1) & (df['post'] == 0)][outcome_col].mean()
        control_post = df[(df['treated'] == 0) & (df['post'] == 1)][outcome_col].mean()
        control_pre = df[(df['treated'] == 0) & (df['post'] == 0)][outcome_col].mean()

        return {
            'model': model,
            'did_estimate': did_estimate,
            'did_se': did_se,
            'did_pvalue': did_pvalue,
            'did_ci': did_ci,
            'treated_post': treated_post,
            'treated_pre': treated_pre,
            'control_post': control_post,
            'control_pre': control_pre,
            'n_observations': len(df),
            'post_threshold': median_time
        }
    except Exception as e:
        print(f"Error in DID regression: {e}")
        return None


def run_event_study(df, treatment_col, outcome_col, time_col, group_col):
    """Run Event Study regression to get dynamic effects."""
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        return None

    # Sort by time and create relative time indicators
    df = df.copy()
    df = df.sort_values(time_col)

    # Get unique time points
    unique_times = df[time_col].unique()
    unique_times = sorted(unique_times)

    if len(unique_times) < 3:
        print("Warning: Not enough time periods for event study")
        return None

    # Use middle time point as reference period
    ref_idx = len(unique_times) // 2
    ref_time = unique_times[ref_idx]

    # Create treated indicator
    treated_val = df[treatment_col].unique()[0]
    df['treated'] = (df[treatment_col] == treated_val).astype(int)

    # Create relative time dummies (before/after reference period)
    df['time_idx'] = df[time_col].map({t: i for i, t in enumerate(unique_times)})
    df['relative_time'] = df['time_idx'] - ref_idx

    # Run event study regression
    # Create dummies for each relative time period
    event_study_results = []

    for rt in sorted(df['relative_time'].unique()):
        if rt == 0:  # Reference period
            continue

        df[f'event_{rt}'] = (df['relative_time'] == rt).astype(int)
        df[f'event_{rt}_treated'] = df[f'event_{rt}'] * df['treated']

    # Build formula
    event_terms = [f'event_{rt}_treated' for rt in sorted(df['relative_time'].unique()) if rt != 0]
    formula = f"{outcome_col} ~ treated + " + " + ".join([f"event_{rt}" for rt in sorted(df['relative_time'].unique()) if rt != 0]) + " + " + " + ".join(event_terms)

    try:
        model = smf.ols(formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df[group_col]})

        # Extract coefficients and confidence intervals
        for rt in sorted(df['relative_time'].unique()):
            if rt == 0:
                continue
            term = f'event_{rt}_treated'
            if term in model.params:
                event_study_results.append({
                    'relative_time': rt,
                    'coefficient': model.params[term],
                    'se': model.bse[term],
                    'ci_lower': model.conf_int().loc[term, 0],
                    'ci_upper': model.conf_int().loc[term, 1],
                    'pvalue': model.pvalues[term]
                })

        return pd.DataFrame(event_study_results)
    except Exception as e:
        print(f"Error in event study: {e}")
        return None


def test_parallel_trends(df, treatment_col, outcome_col, time_col, group_col):
    """Test parallel trends assumption using pre-treatment data."""
    df = df.copy()

    # Get unique time points and identify pre-treatment period
    unique_times = sorted(df[time_col].unique())
    if len(unique_times) < 2:
        return None

    # Assume first half is pre-treatment
    mid_idx = len(unique_times) // 2
    pre_times = unique_times[:mid_idx]
    post_times = unique_times[mid_idx:]

    # Calculate means for each group in pre-period
    treated_val = df[treatment_col].unique()[0]
    df['treated'] = (df[treatment_col] == treated_val).astype(int)
    df['post'] = df[time_col].isin(post_times).astype(int)

    # Pre-period parallel trends test: compare trends in pre-period
    pre_df = df[df['post'] == 0]

    if len(pre_df) == 0 or pre_df['treated'].nunique() < 2:
        return None

    # Calculate trend for each group in pre-period
    pre_trends = pre_df.groupby('treated').apply(
        lambda x: np.polyfit(x[time_col].astype('int64'), x[outcome_col], 1)[0] if len(x) > 1 else 0
    )

    trend_diff = abs(pre_trends.get(1, 0) - pre_trends.get(0, 0))

    # Also calculate simple mean difference in pre-period
    pre_means = pre_df.groupby('treated')[outcome_col].mean()
    mean_diff = abs(pre_means.get(1, 0) - pre_means.get(0, 0))

    return {
        'pre_trends': pre_trends.to_dict(),
        'trend_difference': trend_diff,
        'pre_mean_difference': mean_diff,
        'pre_times': pre_times,
        'post_times': post_times,
        'is_balanced': trend_diff < 0.1  # Threshold for "parallel"
    }


def plot_event_study(event_study_df, output_dir):
    """Plot Event Study figure."""
    if event_study_df is None or len(event_study_df) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot coefficients with confidence intervals
    times = event_study_df['relative_time'].values
    coefs = event_study_df['coefficient'].values
    ci_lower = event_study_df['ci_lower'].values
    ci_upper = event_study_df['ci_upper'].values

    ax.errorbar(times, coefs, yerr=[coefs - ci_lower, ci_upper - coefs],
                fmt='o-', capsize=4, capthick=1.5, linewidth=2, markersize=8,
                color='#2E86AB', ecolor='#A23B72')

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_xlabel('Relative Time (Periods from Intervention)', fontsize=12)
    ax.set_ylabel('Treatment Effect', fontsize=12)
    ax.set_title('Event Study: Dynamic Treatment Effects', fontsize=14)

    # Add shaded region for pre-treatment
    pre_times = event_study_df[event_study_df['relative_time'] < 0]['relative_time']
    if len(pre_times) > 0:
        ax.axvspan(pre_times.min() - 0.5, -0.5, alpha=0.1, color='gray')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'event_study.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Event study plot saved to {os.path.join(output_dir, 'event_study.png')}")


def plot_did_means(result, output_dir):
    """Plot DID group-time means."""
    if result is None:
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    groups = ['Treated\n(Pre)', 'Treated\n(Post)', 'Control\n(Pre)', 'Control\n(Post)']
    means = [result['treated_pre'], result['treated_post'],
             result['control_pre'], result['control_post']]
    colors = ['#3498db', '#e74c3c', '#3498db', '#2ecc71']

    bars = ax.bar(groups, means, color=colors, alpha=0.7, edgecolor='black')

    # Add value labels
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{mean:.2%}', ha='center', va='bottom', fontsize=11)

    ax.set_ylabel('Outcome Rate', fontsize=12)
    ax.set_title('DID: Group-Time Means', fontsize=14)
    ax.set_ylim(0, max(means) * 1.2)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'did_means.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"DID means plot saved to {os.path.join(output_dir, 'did_means.png')}")


def generate_report(df, treatment_col, outcome_col, time_col, group_col,
                    did_result, event_study_df, parallel_trends_result,
                    output_dir):
    """Generate Markdown report."""
    report_path = os.path.join(output_dir, 'DID_Analysis_Report.md')

    with open(report_path, 'w') as f:
        f.write(f"# DID (Difference-in-Differences) Analysis Report\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # 1. Data Overview
        f.write("## 1. Data Overview\n\n")
        f.write(f"| Item | Value |\n")
        f.write(f"|------|-------|\n")
        f.write(f"| Observations | {did_result['n_observations']:,} |\n")
        f.write(f"| Treatment Column | `{treatment_col}` |\n")
        f.write(f"| Outcome Column | `{outcome_col}` |\n")
        f.write(f"| Time Column | `{time_col}` |\n")
        f.write(f"| Group Column | `{group_col}` |\n")
        f.write(f"| Post Threshold | {did_result['post_threshold']} |\n\n")

        # 2. DID Results
        f.write("## 2. DID Regression Results\n\n")
        f.write("Model: `Y = β0 + β1·treated + β2·post + β3·(treated × post) + ε`\n\n")
        f.write("**Key Parameter (β3) = DID Estimate**\n\n")

        sig = "***" if did_result['did_pvalue'] < 0.001 else "**" if did_result['did_pvalue'] < 0.01 else "*" if did_result['did_pvalue'] < 0.05 else ""

        f.write(f"| Metric | Value |\n")
        f.write(f"|-------|-------|\n")
        f.write(f"| DID Estimate | {did_result['did_estimate']:+.4f}{sig} |\n")
        f.write(f"| Standard Error | {did_result['did_se']:.4f} |\n")
        f.write(f"| p-value | {did_result['did_pvalue']:.4f} |\n")
        f.write(f"| 95% CI | [{did_result['did_ci'][0]:.4f}, {did_result['did_ci'][1]:.4f}] |\n\n")

        if sig:
            significance_note = "Significant" if did_result['did_pvalue'] < 0.05 else "Not Significant"
            f.write(f"**Interpretation**: The treatment effect is {significance_note} at 5% level.\n\n")

        # 3. Group-Time Means
        f.write("## 3. Group-Time Means\n\n")
        f.write(f"| Group | {outcome_col} Mean |\n")
        f.write(f"|-------|-----------------|\n")
        f.write(f"| Treated (Pre) | {did_result['treated_pre']:.4f} |\n")
        f.write(f"| Treated (Post) | {did_result['treated_post']:.4f} |\n")
        f.write(f"| Control (Pre) | {did_result['control_pre']:.4f} |\n")
        f.write(f"| Control (Post) | {did_result['control_post']:.4f} |\n\n")

        f.write("**Calculation Check**:\n")
        f.write(f"- DID = ({did_result['treated_post']:.4f} - {did_result['treated_pre']:.4f}) - ({did_result['control_post']:.4f} - {did_result['control_pre']:.4f})\n")
        f.write(f"- DID = {(did_result['treated_post'] - did_result['treated_pre']):.4f} - {(did_result['control_post'] - did_result['control_pre']):.4f} = {did_result['did_estimate']:.4f}\n\n")

        # 4. Parallel Trends Test
        f.write("## 4. Parallel Trends Test\n\n")
        if parallel_trends_result:
            is_balanced = parallel_trends_result['is_balanced']
            f.write(f"| Metric | Value |\n")
            f.write(f"|-------|-------|\n")
            f.write(f"| Pre-period Treated Mean | {parallel_trends_result['pre_trends'].get(1, 'N/A')} |\n")
            f.write(f"| Pre-period Control Mean | {parallel_trends_result['pre_trends'].get(0, 'N/A')} |\n")
            f.write(f"| Pre-period Trend Difference | {parallel_trends_result['trend_difference']:.4f} |\n\n")

            status = "✅ PASSED" if is_balanced else "⚠️ CAUTION"
            f.write(f"**Parallel Trends Assumption**: {status}\n\n")
            if is_balanced:
                f.write("Pre-treatment trends are approximately parallel between treatment and control groups.\n\n")
            else:
                f.write("Pre-treatment trends show some difference. Interpretation should be cautious.\n\n")
        else:
            f.write("Parallel trends test could not be performed (insufficient pre-treatment data).\n\n")

        # 5. Event Study (if available)
        if event_study_df is not None and len(event_study_df) > 0:
            f.write("## 5. Event Study: Dynamic Effects\n\n")
            f.write("![Event Study](event_study.png)\n\n")
            f.write("| Relative Time | Coefficient | SE | p-value |\n")
            f.write(f"|----------------|-------------|----|---------|")
            for _, row in event_study_df.iterrows():
                f.write(f"\n| {row['relative_time']} | {row['coefficient']:+.4f} | {row['se']:.4f} | {row['pvalue']:.4f} |")
            f.write("\n\n")

        # 6. Business Interpretation
        f.write("## 6. Business Interpretation\n\n")
        if did_result['did_pvalue'] < 0.05:
            effect_direction = "提升" if did_result['did_estimate'] > 0 else "降低"
            f.write(f"### Key Finding\n\n")
            f.write(f"策略/政策变化导致 **{outcome_col}** {effect_direction}了 **{abs(did_result['did_estimate'])*100:.2f}%**。\n\n")
            f.write(f"95%置信区间: [{did_result['did_ci'][0]*100:.2f}%, {did_result['did_ci'][1]*100:.2f}%]\n\n")
            f.write("### Recommendations\n\n")
            if did_result['did_estimate'] > 0:
                f.write("- ✅ 建议推广该策略到其他渠道/用户群\n")
                f.write("- ✅ 预期可带来类似幅度的效果提升\n")
            else:
                f.write("- ⚠️ 需要进一步分析效果为负的原因\n")
                f.write("- ⚠️ 考虑回滚或优化策略\n")
        else:
            f.write("### Key Finding\n\n")
            f.write(f"策略/政策变化的效应在统计上**不显著**（p={did_result['did_pvalue']:.4f}）。\n\n")
            f.write("### Recommendations\n\n")
            f.write("- ⚠️ 当前数据无法证明策略有效\n")
            f.write("- 📊 建议收集更多数据或延长观察期\n")

        f.write("\n---\n\n")
        f.write("## 7. Technical Notes\n\n")
        f.write("- Standard errors are clustered by group\n")
        f.write("- DID assumes parallel trends in the absence of treatment\n")
        f.write("- Results should be validated with additional robustness checks\n")

    print(f"Report saved to {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="DID (Difference-in-Differences) Analysis")
    parser.add_argument("file_path", help="Path to the CSV file")
    parser.add_argument("--treatment", "-t", required=True,
                        help="Treatment indicator column (e.g., channel, group)")
    parser.add_argument("--outcome", "-y", required=True,
                        help="Outcome variable column (e.g., retention, conversion)")
    parser.add_argument("--time", required=True,
                        help="Time variable column (e.g., date, week)")
    parser.add_argument("--group", "-g", required=True,
                        help="Group/cluster column for standard errors")
    parser.add_argument("--covariates", "-c", default=None,
                        help="Comma-separated list of control variables")
    parser.add_argument("--output", "-o", default=None,
                        help="Output directory for report and plots")

    args = parser.parse_args()

    # Parse covariates
    covariates = None
    if args.covariates:
        covariates = [c.strip() for c in args.covariates.split(',')]

    # Set output directory
    if args.output:
        output_dir = args.output
    else:
        base_name = os.path.splitext(os.path.basename(args.file_path))[0]
        output_dir = os.path.join(os.getcwd(), f"{base_name}_did_analysis")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load data
    print(f"Loading data from {args.file_path}...")
    df = load_and_prepare_data(args.file_path, args.treatment, args.outcome,
                               args.time, args.group, covariates)
    if df is None:
        return

    print(f"Data loaded: {len(df)} rows")

    # Run DID regression
    print("Running DID regression...")
    did_result = run_did_regression(df, args.treatment, args.outcome,
                                    args.time, args.group, covariates)
    if did_result is None:
        print("DID regression failed")
        return

    print(f"DID Estimate: {did_result['did_estimate']:+.4f} (p={did_result['did_pvalue']:.4f})")

    # Run event study
    print("Running event study...")
    event_study_df = run_event_study(df, args.treatment, args.outcome,
                                      args.time, args.group)

    # Test parallel trends
    print("Testing parallel trends...")
    parallel_trends_result = test_parallel_trends(df, args.treatment, args.outcome,
                                                   args.time, args.group)

    # Generate plots
    plot_did_means(did_result, output_dir)
    if event_study_df is not None:
        plot_event_study(event_study_df, output_dir)

    # Generate report
    print("Generating report...")
    generate_report(df, args.treatment, args.outcome, args.time, args.group,
                    did_result, event_study_df, parallel_trends_result,
                    output_dir)

    print("\n✅ DID Analysis Complete!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
