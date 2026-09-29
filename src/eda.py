"""
Exploratory Data Analysis (EDA) module for Task 1.
Generates comprehensive summary statistics, structural analyses, and publication-ready figures.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from src.config import FIGURES_DIR, TABLES_DIR, TARGET_COL, NUMERICAL_COLS, CATEGORICAL_COLS

# Configure aesthetics for clean, modern academic visualizations
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "semibold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.titleweight": "bold",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

PALETTE = ["#2b5c8f", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]

def perform_eda_analysis(df_raw: pd.DataFrame) -> dict:
    """
    Executes a comprehensive statistical audit of the dataset.
    Returns structured results for reporting and table generation.
    """
    results = {}
    
    # Dimensions & basic info
    results["shape"] = df_raw.shape
    results["num_rows"] = df_raw.shape[0]
    results["num_cols"] = df_raw.shape[1]
    
    # Missing / whitespace values inspection
    whitespace_mask = df_raw["TotalCharges"].astype(str).str.strip() == ""
    results["whitespace_total_charges_count"] = int(whitespace_mask.sum())
    results["whitespace_tenure_values"] = df_raw.loc[whitespace_mask, "tenure"].tolist()
    
    # Clean temporary copy for numerical analysis
    df_clean = df_raw.copy()
    df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"].str.strip(), errors="coerce")
    
    # Column types
    results["dtypes"] = df_raw.dtypes.to_dict()
    results["numerical_cols"] = NUMERICAL_COLS
    results["categorical_cols"] = CATEGORICAL_COLS
    
    # Descriptive statistics
    num_stats = df_clean[NUMERICAL_COLS].describe().T
    num_stats["median"] = df_clean[NUMERICAL_COLS].median()
    num_stats["skew"] = df_clean[NUMERICAL_COLS].skew()
    results["numerical_stats"] = num_stats
    num_stats.to_csv(TABLES_DIR / "task1_numerical_descriptive_stats.csv")
    
    # Categorical summary
    cat_summary = []
    for col in CATEGORICAL_COLS:
        top_val = df_clean[col].mode()[0]
        top_freq = (df_clean[col] == top_val).sum()
        cat_summary.append({
            "Feature": col,
            "Unique_Values": df_clean[col].nunique(),
            "Top_Category": top_val,
            "Top_Frequency": top_freq,
            "Top_Percentage": round(top_freq / len(df_clean) * 100, 2)
        })
    cat_df = pd.DataFrame(cat_summary)
    results["categorical_stats"] = cat_df
    cat_df.to_csv(TABLES_DIR / "task1_categorical_summary.csv", index=False)
    
    # Target distribution
    churn_counts = df_clean[TARGET_COL].value_counts()
    churn_pcts = df_clean[TARGET_COL].value_counts(normalize=True) * 100
    target_summary = pd.DataFrame({
        "Count": churn_counts,
        "Percentage": churn_pcts
    })
    results["target_distribution"] = target_summary
    target_summary.to_csv(TABLES_DIR / "task1_target_distribution.csv")
    
    # Outlier detection using IQR on numerical columns
    outlier_summary = {}
    for col in NUMERICAL_COLS:
        s = df_clean[col].dropna()
        q25, q75 = np.percentile(s, 25), np.percentile(s, 75)
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        outliers = s[(s < lower_bound) | (s > upper_bound)]
        outlier_summary[col] = {
            "Q25": q25,
            "Q75": q75,
            "IQR": iqr,
            "Lower_Bound": lower_bound,
            "Upper_Bound": upper_bound,
            "Outlier_Count": len(outliers),
            "Outlier_Pct": round(len(outliers) / len(s) * 100, 2)
        }
    results["outliers"] = pd.DataFrame(outlier_summary).T
    results["outliers"].to_csv(TABLES_DIR / "task1_outlier_summary.csv")
    
    # Numerical correlation with target
    df_clean["Target_Num"] = (df_clean[TARGET_COL] == "Yes").astype(int)
    corr_matrix = df_clean[NUMERICAL_COLS + ["Target_Num"]].corr()
    results["correlation_matrix"] = corr_matrix
    corr_matrix.to_csv(TABLES_DIR / "task1_correlation_matrix.csv")
    
    return results

def generate_eda_plots(df_raw: pd.DataFrame):
    """Generates and saves all required Task 1 visualizations."""
    df_clean = df_raw.copy()
    df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"].str.strip(), errors="coerce")
    
    # 1. Target Variable Distribution
    fig, ax = plt.subplots(figsize=(6, 5))
    counts = df_clean[TARGET_COL].value_counts()
    bars = ax.bar(["No Churn (0)", "Churn (1)"], counts.values, color=["#2b5c8f", "#d95f02"], width=0.55, edgecolor="black", linewidth=1.2)
    for bar in bars:
        yval = bar.get_height()
        pct = yval / len(df_clean) * 100
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 70, f"{yval:,}\n({pct:.1f}%)", ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax.set_title("Target Variable Distribution (Churn Class Imbalance)", pad=15)
    ax.set_ylabel("Customer Count")
    ax.set_ylim(0, counts.max() * 1.18)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "task1_churn_distribution.png")
    plt.close(fig)
    
    # 2. Numerical Distributions stratified by Churn
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    num_vars = [
        ("tenure", "Tenure (Months)", "Number of Months with Telco"),
        ("MonthlyCharges", "Monthly Charges ($)", "Monthly Recurring Charge"),
        ("TotalCharges", "Total Charges ($)", "Total Accumulated Expenditure")
    ]
    for i, (col, title, xlabel) in enumerate(num_vars):
        sns.histplot(
            data=df_clean,
            x=col,
            hue=TARGET_COL,
            element="step",
            stat="density",
            common_norm=False,
            palette={"No": "#2b5c8f", "Yes": "#d95f02"},
            ax=axes[i],
            kde=True,
            line_kws={"linewidth": 2}
        )
        axes[i].set_title(title)
        axes[i].set_xlabel(xlabel)
        axes[i].set_ylabel("Density")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "task1_numerical_distributions.png")
    plt.close(fig)
    
    # 3. Categorical Relationships with Churn
    key_cat_features = ["Contract", "InternetService", "PaymentMethod", "TechSupport"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for idx, col in enumerate(key_cat_features):
        churn_rate_by_cat = df_clean.groupby(col)[TARGET_COL].apply(lambda s: (s == "Yes").mean() * 100).reset_index()
        churn_rate_by_cat.columns = [col, "ChurnRate"]
        churn_rate_by_cat = churn_rate_by_cat.sort_values("ChurnRate", ascending=False)
        
        bars = axes[idx].bar(
            churn_rate_by_cat[col],
            churn_rate_by_cat["ChurnRate"],
            color="#d95f02",
            edgecolor="black",
            alpha=0.85
        )
        axes[idx].axhline(df_clean[TARGET_COL].eq("Yes").mean() * 100, color="navy", linestyle="--", linewidth=1.5, label=f"Average Churn ({df_clean[TARGET_COL].eq('Yes').mean()*100:.1f}%)")
        axes[idx].set_title(f"Churn Rate by {col}")
        axes[idx].set_ylabel("Churn Percentage (%)")
        axes[idx].set_ylim(0, max(churn_rate_by_cat["ChurnRate"]) * 1.25)
        axes[idx].tick_params(axis="x", rotation=25)
        axes[idx].legend(loc="upper right")
        for bar in bars:
            height = bar.get_height()
            axes[idx].text(bar.get_x() + bar.get_width()/2., height + 1, f"{height:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "task1_categorical_relationships.png")
    plt.close(fig)
    
    # 4. Outliers Inspection via Boxplots
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for i, col in enumerate(NUMERICAL_COLS):
        sns.boxplot(
            data=df_clean,
            y=col,
            x=TARGET_COL,
            hue=TARGET_COL,
            legend=False,
            palette={"No": "#2b5c8f", "Yes": "#d95f02"},
            ax=axes[i],
            width=0.45,
            fliersize=3
        )
        axes[i].set_title(f"{col} by Churn Status")
        axes[i].set_xlabel("Churn Status")
        axes[i].set_ylabel(col)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "task1_outliers_boxplots.png")
    plt.close(fig)
    
    # 5. Correlation Heatmap
    fig, ax = plt.subplots(figsize=(7, 6))
    df_clean["Churn_Binary"] = (df_clean[TARGET_COL] == "Yes").astype(int)
    corr = df_clean[NUMERICAL_COLS + ["Churn_Binary"]].corr()
    sns.heatmap(
        corr,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=0.8,
        cbar_kws={"label": "Pearson Correlation Coefficient"},
        ax=ax
    )
    ax.set_title("Correlation Heatmap (Numerical Features & Churn Target)", pad=12)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "task1_correlation_matrix.png")
    plt.close(fig)
    
    print("Task 1 EDA figures successfully saved to outputs/figures/")
