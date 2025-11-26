import pandas as pd
import os

def basic_cleaning(input_path: str, output_path: str):
    """Clean raw housing data — text normalization, duplicates, consistent casing, and remove outliers."""
    
    df = pd.read_csv(input_path)
    print(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")

    # Normalize case and strip whitespace for object columns
    object_cols = df.select_dtypes(include="object").columns
    for col in object_cols:
        df[col] = df[col].str.strip().str.lower()

    # Remove duplicates
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    print(f"Removed {before - after} duplicate rows.")

    # Confirm numeric types
    numeric_cols = ["price", "area", "bedrooms", "bathrooms", "stories", "parking"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # --- Remove outliers using IQR method ---
    """
    def remove_outliers(df, cols):
        for col in cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower) & (df[col] <= upper)]
        return df

    before_outliers = df.shape[0]
    df = remove_outliers(df, numeric_cols)
    after_outliers = df.shape[0]
    print(f"Removed {before_outliers - after_outliers} outlier rows.")
    """

    # Save cleaned data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Cleaned data saved to {output_path}")

    return df

if __name__ == "__main__":
    basic_cleaning("data/raw/Housing.csv", "data/processed/Housing_clean.csv")