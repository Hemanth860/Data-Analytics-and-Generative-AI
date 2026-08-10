"""
Zepto Data & AI Platform - Module 2: Analytics Pipeline Part A (/analytics/01_eda.py)
Author: AI/ML Engineer (B.Tech Capstone Project)

This script performs Part A of Module 2:
1. Loads dataset with offline fallback (titanic.csv) and profiles it (info, describe, shape, missing percentages).
2. Implements percentage-based missing-value handling rules:
   - < 5% missing: Drop rows (embarked/embark_town).
   - 5% - 30% missing: Impute median (age).
   - > 30% missing: Drop column or encode 'missing' as category (deck).
3. Univariate Analysis: IQR outlier detection for age & fare; mean/median/mode skewness check.
4. Bivariate Analysis: Survival rate by sex, pclass, and sex+pclass combined; 6x6 Correlation heatmap.
5. Multivariate Data Story: 4 distinct charts saved to artifacts/.
6. Exploratory Standardization Sanity Check: z-score standardization of age & fare (mean ~0, std ~1).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
CSV_PATH = os.path.join(os.path.dirname(__file__), "titanic.csv")

def load_and_profile_data():
    """Load dataset with committed offline fallback and print profiling stats."""
    if os.path.exists(CSV_PATH):
        print(f"Loading dataset from committed offline fallback: '{CSV_PATH}'...")
        df = pd.read_csv(CSV_PATH)
    else:
        print("Attempting to load dataset via seaborn...")
        df = sns.load_dataset('titanic')
        df.to_csv(CSV_PATH, index=False)
        print(f"Saved committed offline fallback to '{CSV_PATH}'.")
        
    print("\n" + "="*70)
    print("DATASET PROFILING SUMMARY")
    print("="*70)
    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nColumn Data Types and Info:")
    df.info()
    
    print("\nSummary Statistics:")
    print(df.describe(include='all').T)
    
    print("\nMissing Value Percentages:")
    missing_counts = df.isnull().sum()
    missing_pcts = (missing_counts / len(df)) * 100
    missing_df = pd.DataFrame({"Missing Count": missing_counts, "Missing Percentage (%)": missing_pcts.round(2)})
    missing_df = missing_df[missing_df["Missing Count"] > 0].sort_values(by="Missing Percentage (%)", ascending=False)
    print(missing_df)
    
    return df, missing_df

def apply_missing_value_handling(df, missing_df):
    """
    Apply percentage-based threshold rules:
    - < 5% missing: Drop rows
    - 5% - 30% missing: Impute median
    - > 30% missing: Drop column or encode 'missing' category
    """
    print("\n" + "="*70)
    print("APPLYING PERCENTAGE-BASED MISSING VALUE HANDLING RULES")
    print("="*70)
    
    df_clean = df.copy()
    
    for col, row in missing_df.iterrows():
        pct = row["Missing Percentage (%)"]
        if pct < 5.0:
            print(f"Column '{col}' missing rate is {pct}% (<5%) -> Dropping {int(row['Missing Count'])} rows.")
            df_clean = df_clean.dropna(subset=[col])
        elif 5.0 <= pct <= 30.0:
            median_val = df_clean[col].median()
            print(f"Column '{col}' missing rate is {pct}% (5%-30%) -> Imputing missing values with median: {median_val:.2f}.")
            df_clean[col] = df_clean[col].fillna(median_val)
        else:
            print(f"Column '{col}' missing rate is {pct}% (>30%) -> Dropping column due to unreliable imputation.")
            df_clean = df_clean.drop(columns=[col])
            
    print(f"Cleaned DataFrame Shape: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")
    return df_clean

def univariate_analysis(df_clean):
    """Histogram, boxplot, IQR outlier count, and skewness analysis for age & fare."""
    print("\n" + "="*70)
    print("UNIVARIATE ANALYSIS & IQR OUTLIER DETECTION")
    print("="*70)
    
    plt.figure(figsize=(12, 5))
    
    # Age plot
    plt.subplot(1, 2, 1)
    sns.histplot(df_clean["age"], kde=True, color="skyblue")
    plt.title("Distribution of Age")
    
    # Fare plot
    plt.subplot(1, 2, 2)
    sns.histplot(df_clean["fare"], kde=True, color="salmon")
    plt.title("Distribution of Fare")
    
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, "univariate_distributions.png"), dpi=300)
    plt.close()
    
    # Outlier detection using IQR
    for col in ["age", "fare"]:
        q1 = df_clean[col].quantile(0.25)
        q3 = df_clean[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)]
        print(f"Column '{col}': Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f} | Outlier Bounds: [{lower_bound:.2f}, {upper_bound:.2f}] | Outliers Count: {len(outliers)}")
        
    # Skewness calculation for Fare
    mean_fare = df_clean["fare"].mean()
    median_fare = df_clean["fare"].median()
    mode_fare = df_clean["fare"].mode()[0]
    
    print(f"\nFare Central Tendency Metrics: Mean={mean_fare:.2f}, Median={median_fare:.2f}, Mode={mode_fare:.2f}")
    if mean_fare > median_fare:
        skew_type = "Right-Skewed (Positive Skew)"
    elif mean_fare < median_fare:
        skew_type = "Left-Skewed (Negative Skew)"
    else:
        skew_type = "Symmetric"
        
    print(f"Fare Distribution Conclusion: {skew_type} (Mean > Median > Mode ordering).")

def bivariate_analysis(df_clean):
    """Survival rates by sex, pclass, and sex+pclass; 6x6 correlation matrix & heatmap."""
    print("\n" + "="*70)
    print("BIVARIATE ANALYSIS & SURVIVAL BREAKDOWNS")
    print("="*70)
    
    # Survival by Sex
    sex_survival = df_clean.groupby("sex")["survived"].mean() * 100
    print("\nSurvival Rate by Sex:")
    for s, rate in sex_survival.items():
        print(f" - {s.capitalize()}: {rate:.2f}%")
        
    # Survival by Pclass
    pclass_survival = df_clean.groupby("pclass")["survived"].mean() * 100
    print("\nSurvival Rate by Pclass:")
    for p, rate in pclass_survival.items():
        print(f" - Class {p}: {rate:.2f}%")
        
    # Survival by Sex & Pclass
    combined_survival = df_clean.groupby(["sex", "pclass"])["survived"].mean() * 100
    print("\nSurvival Rate by Sex & Pclass Combined:")
    for (s, p), rate in combined_survival.items():
        print(f" - {s.capitalize()}, Class {p}: {rate:.2f}%")
        
    # 6x6 Correlation Matrix (survived, pclass, age, sibsp, parch, fare)
    num_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
    corr_matrix = df_clean[num_cols].corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("6x6 Numeric Correlation Heatmap")
    plt.tight_layout()
    heatmap_path = os.path.join(ARTIFACTS_DIR, "correlation_heatmap.png")
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    
    print("\n6x6 Correlation Matrix:")
    print(corr_matrix.round(3))
    
    # Identify top two off-diagonal correlations
    off_diag_pairs = []
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            col1, col2 = num_cols[i], num_cols[j]
            r = corr_matrix.loc[col1, col2]
            off_diag_pairs.append((col1, col2, r, abs(r)))
            
    off_diag_pairs.sort(key=lambda x: x[3], reverse=True)
    top1 = off_diag_pairs[0]
    top2 = off_diag_pairs[1]
    
    print("\nTop 2 Strongest Off-Diagonal Correlations:")
    print(f" 1. {top1[0]} & {top1[1]}: r = {top1[2]:.3f} (|r| = {top1[3]:.3f})")
    print(f" 2. {top2[0]} & {top2[1]}: r = {top2[2]:.3f} (|r| = {top2[3]:.3f})")

def generate_multivariate_story_charts(df_clean):
    """Generate 4 distinct charts for multivariate story and save to artifacts/."""
    print("\n" + "="*70)
    print("GENERATING MULTIVARIATE DATA STORY CHARTS")
    print("="*70)
    
    # Chart 1: Survival Rate by Sex & Class
    plt.figure(figsize=(7, 5))
    sns.barplot(data=df_clean, x="pclass", y="survived", hue="sex", palette="Set2")
    plt.title("Chart 1: Survival Rate by Passenger Class and Sex")
    plt.xlabel("Passenger Class (Pclass)")
    plt.ylabel("Survival Rate")
    plt.savefig(os.path.join(ARTIFACTS_DIR, "chart1_survival_sex_class.png"), dpi=300)
    plt.close()
    
    # Chart 2: Fare vs Age by Survival
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df_clean, x="age", y="fare", hue="survived", style="sex", alpha=0.7, palette="coolwarm")
    plt.title("Chart 2: Fare vs Age Distribution by Survival Status")
    plt.xlabel("Age (Years)")
    plt.ylabel("Fare (GBP/INR)")
    plt.savefig(os.path.join(ARTIFACTS_DIR, "chart2_fare_vs_age.png"), dpi=300)
    plt.close()
    
    # Chart 3: Age Boxplot by Class and Survival
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df_clean, x="pclass", y="age", hue="survived", palette="Blues")
    plt.title("Chart 3: Age Distribution Across Classes by Survival")
    plt.xlabel("Passenger Class")
    plt.ylabel("Age")
    plt.savefig(os.path.join(ARTIFACTS_DIR, "chart3_age_box_class_survival.png"), dpi=300)
    plt.close()
    
    # Chart 4: Family Size (SibSp + Parch) Impact on Survival
    df_clean["family_size"] = df_clean["sibsp"] + df_clean["parch"] + 1
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df_clean, x="family_size", y="survived", palette="viridis")
    plt.title("Chart 4: Survival Rate by Family Size (SibSp + Parch + 1)")
    plt.xlabel("Family Size")
    plt.ylabel("Survival Rate")
    plt.savefig(os.path.join(ARTIFACTS_DIR, "chart4_survival_family_size.png"), dpi=300)
    plt.close()
    
    print(f"Generated 4 charts successfully in '{ARTIFACTS_DIR}'.")

def exploratory_standardization_sanity_check(df_clean):
    """EDA sanity check: standardize age and fare using z = (x - mean) / std."""
    print("\n" + "="*70)
    print("EDA SANITY CHECK: BEFORE/AFTER STANDARDIZATION")
    print("="*70)
    
    df_std = df_clean.copy()
    
    for col in ["age", "fare"]:
        mean_before = df_clean[col].mean()
        std_before = df_clean[col].std()
        
        df_std[col + "_zscore"] = (df_clean[col] - mean_before) / std_before
        
        mean_after = df_std[col + "_zscore"].mean()
        std_after = df_std[col + "_zscore"].std()
        
        print(f"Column '{col}':")
        print(f"  Before Standardization -> Mean: {mean_before:.4f}, Std: {std_before:.4f}")
        print(f"  After Z-Score Scaling  -> Mean: {mean_after:.4f} (~0), Std: {std_after:.4f} (~1)")

def main():
    print("Executing Module 2 Part A: EDA & Profiling Pipeline...")
    df, missing_df = load_and_profile_data()
    df_clean = apply_missing_value_handling(df, missing_df)
    univariate_analysis(df_clean)
    bivariate_analysis(df_clean)
    generate_multivariate_story_charts(df_clean)
    exploratory_standardization_sanity_check(df_clean)
    print("\nPart A completed successfully!")

if __name__ == "__main__":
    main()
