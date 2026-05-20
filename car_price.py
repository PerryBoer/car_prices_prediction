"""
Car price dataset inspection, cleaning, predicting workflow.

Goal:
- Understand raw data quality
- Visualize target distribution
- Clean obvious datatype issues
- Remove duplicates and extreme price outliers
- Re-visualize after cleaning
- Train test split
- Build baseline linear regression model
- Build random forest model
- Give fiat 500 example
"""

# basic packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


DATA_PATH = "car_price_prediction.csv"
TARGET_COLUMN = "Price"
PRICE_UPPER_QUANTILE = 0.90
DROP_COLUMNS = ["ID"]
RANDOM_STATE = 42


def load_data(path):
    """Load dataset from CSV."""
    return pd.read_csv(path)


def inspect_dataframe(df):
    """Print basic dataframe diagnostics."""
    print("\n--- HEAD ---")
    print(df.head())

    print("\n--- INFO ---")
    print(df.info())

    print("\n--- DESCRIBE NUMERIC ---")
    print(df.describe())

    print("\n--- DESCRIBE ALL ---")
    print(df.describe(include="all"))

    print("\n--- MISSING VALUES ---")
    print(df.isna().sum().sort_values(ascending=False))

    print("\n--- DUPLICATES ---")
    print(df.duplicated().sum())

    print("\n--- DTYPES ---")
    print(df.dtypes)


def plot_price_distribution(df, title_suffix=""):
    """Plot histogram and boxplot of the original price."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(df[TARGET_COLUMN], bins=50, ax=axes[0])
    axes[0].set_title(f"Price Histogram {title_suffix}")
    axes[0].set_xlabel("Price")

    sns.boxplot(y=df[TARGET_COLUMN], ax=axes[1])
    axes[1].set_title(f"Price Boxplot {title_suffix}")
    axes[1].set_ylabel("Price")

    plt.tight_layout()
    plt.show()


def plot_log_price_distribution(df, title_suffix=""):
    """Plot histogram of log-transformed price."""
    log_price = np.log1p(df[TARGET_COLUMN])

    plt.figure(figsize=(8, 5))
    sns.histplot(log_price, bins=50)
    plt.title(f"Log Price Histogram {title_suffix}")
    plt.xlabel("log1p(Price)")
    plt.tight_layout()
    plt.show()


def inspect_column_values(df, columns, n_values=20):
    """Print value counts or examples for selected columns."""
    for column in columns:
        if column not in df.columns:
            continue

        print(f"\n--- {column} ---")
        print(df[column].value_counts(dropna=False).head(n_values))


def plot_top_categories(df, column, top_n=15):
    """Plot top categories for a categorical column."""
    if column not in df.columns:
        return

    top_counts = df[column].value_counts().head(top_n)

    plt.figure(figsize=(10, 5))
    sns.barplot(
        x=top_counts.values,
        y=top_counts.index,
        orient="h",
    )
    plt.title(f"Top {top_n} values for {column}")
    plt.xlabel("Count")
    plt.ylabel(column)
    plt.tight_layout()
    plt.show()


def plot_numeric_column(df, column):
    """Plot histogram and boxplot for a numeric column."""
    if column not in df.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(df[column].dropna(), bins=50, ax=axes[0])
    axes[0].set_title(f"{column} Histogram")

    sns.boxplot(y=df[column], ax=axes[1])
    axes[1].set_title(f"{column} Boxplot")

    plt.tight_layout()
    plt.show()


def clean_numeric_from_string(df, column):
    """Extract numeric part from a string column and convert to float."""
    df = df.copy()

    if column not in df.columns:
        return df

    df[column] = (
        df[column]
        .astype(str)
        .str.extract(r"(\d+\.?\d*)")[0]
        .astype(float)
    )

    return df


def clean_levy(df):
    """Clean Levy column by replacing non-numeric placeholders."""
    df = df.copy()

    if "Levy" not in df.columns:
        return df

    df["Levy"] = (
        df["Levy"]
        .replace("-", np.nan)
        .astype(str)
        .str.extract(r"(\d+\.?\d*)")[0]
        .astype(float)
    )

    return df


def remove_duplicates(df):
    """Remove duplicate rows."""
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)

    print(f"Removed {before - after} duplicate rows.")

    return df


def remove_invalid_prices(df):
    """Remove rows with invalid target values."""
    before = len(df)

    df = df[
        df[TARGET_COLUMN].notna()
        & (df[TARGET_COLUMN] > 0)
    ]

    after = len(df)
    print(f"Removed {before - after} rows with invalid prices.")

    return df


def remove_price_outliers(df, upper_quantile=PRICE_UPPER_QUANTILE):
    """Remove extreme price outliers based on upper quantile."""
    before = len(df)
    upper_limit = df[TARGET_COLUMN].quantile(upper_quantile)

    df = df[df[TARGET_COLUMN] <= upper_limit]

    after = len(df)

    print(f"Price upper limit: {upper_limit:,.2f}")
    print(f"Removed {before - after} price outlier rows.")

    return df

def remove_mileage_outliers(df, max_mileage=1_000_000):
    """Remove unrealistic mileage values."""
    df = df.copy()

    before = len(df)

    df = df[
        df["Mileage"].isna()
        | (df["Mileage"] <= max_mileage)
    ]

    after = len(df)

    print(f"Maximum mileage allowed: {max_mileage:,.0f}")
    print(f"Removed {before - after} mileage outlier rows.")

    return df

def clean_dataframe(
    df,
    remove_outliers=True,
    upper_quantile=PRICE_UPPER_QUANTILE,
):
    """Main cleaning pipeline."""
    df = df.copy()

    df = remove_duplicates(df)
    df = remove_invalid_prices(df)

    df = clean_numeric_from_string(df, "Mileage")
    df = clean_numeric_from_string(df, "Engine volume")
    df = clean_levy(df)

    if remove_outliers:
        df = remove_price_outliers(
            df,
            upper_quantile=upper_quantile,
        )

        df = remove_mileage_outliers(
        df,
        max_mileage=600000,
        )

    return df

def fill_missing_numeric_with_median(df, columns):
    """Fill missing numeric values with the column median."""
    df = df.copy()

    for column in columns:
        if column not in df.columns:
            continue

        missing_before = df[column].isna().sum()

        if missing_before > 0:
            median_value = df[column].median()
            df[column] = df[column].fillna(median_value)

            print(
                f"Filled {missing_before} missing values in {column} "
                f"with median: {median_value:,.2f}"
            )

    return df


def prepare_features_target(df):
    """Prepare features and log-transformed target."""
    df = df.copy()

    X = df.drop(columns=[TARGET_COLUMN] + DROP_COLUMNS, errors="ignore")
    y = np.log1p(df[TARGET_COLUMN])

    return X, y


def build_baseline_model(X):
    """Build an interpretable linear regression baseline."""

    numeric_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns

    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                "passthrough",
                numeric_features,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )

    return model


def build_random_forest_model(X):
    """Build a simple Random Forest regression model."""

    numeric_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns

    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns

    preprocessor = ColumnTransformer(
        transformers=[("numeric", "passthrough", numeric_features), 
                      ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features,)]
                      )

    model = Pipeline(
        steps=[("preprocessor", preprocessor), 
               ("regressor", RandomForestRegressor(n_estimators=200,random_state=RANDOM_STATE,n_jobs=-1,min_samples_leaf=2))]
        )

    return model


def evaluate_model(model, X_test, y_test, type):
    """Evaluate predictions on the original price scale."""
    log_predictions = model.predict(X_test)

    predictions = np.expm1(log_predictions)
    actuals = np.expm1(y_test)

    mae = mean_absolute_error(actuals, predictions)
    rmse = root_mean_squared_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)

    print(f"\n================ {type} MODEL RESULTS ================")
    print(f"MAE:  {mae:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"R²:   {r2:.3f}")


def main():
    """Run full inspection and cleaning workflow."""
    raw_df = load_data(DATA_PATH)

    print("\n================ RAW DATA INSPECTION ================")
    inspect_dataframe(raw_df)

    print("\n================ RAW TARGET PLOTS ================")
    plot_price_distribution(raw_df, title_suffix="Before Cleaning")
    plot_log_price_distribution(raw_df, title_suffix="Before Cleaning")

    print("\n================ CATEGORICAL INSPECTION ================")
    categorical_columns = [
        "Manufacturer",
        "Model",
        "Category",
        "Fuel type",
        "Gear box type",
        "Drive wheels",
        "Doors",
        "Wheel",
        "Color",
        "Levy",
        "Mileage",
        "Engine volume",
        "Doors",
    ]

    inspect_column_values(raw_df, categorical_columns)

    plot_top_categories(raw_df, "Manufacturer", top_n=15)
    plot_top_categories(raw_df, "Category", top_n=15)
    plot_top_categories(raw_df, "Fuel type", top_n=15)

    print("\n================ CLEANING DATA ================")
    clean_df = clean_dataframe(
        raw_df,
        remove_outliers=True,
        upper_quantile=0.99,
    )

    print("\n================ MISSING VALUE IMPUTATION ================")
    clean_df = fill_missing_numeric_with_median(
        clean_df,
        columns=[
            "Levy",
        ],
    )

    print("\nMissing values after imputation:")
    print(clean_df.isna().sum().sort_values(ascending=False).head(10))

    print("\n================ CLEANED DATA INSPECTION ================")
    inspect_dataframe(clean_df)

    print("\n================ CLEANED TARGET PLOTS ================")
    plot_price_distribution(clean_df, title_suffix="After Cleaning")
    plot_log_price_distribution(clean_df, title_suffix="After Cleaning")

    print("\n================ CLEANED NUMERIC PLOTS ================")
    numeric_columns = [
        "Price",
        "Prod. year",
        "Mileage",
        "Engine volume",
        "Cylinders",
        "Airbags",
        "Levy",
    ]

    for column in numeric_columns:
        plot_numeric_column(clean_df, column)

    print("\n================ FINAL SHAPE ================")
    print(f"Raw shape: {raw_df.shape}")
    print(f"Clean shape: {clean_df.shape}")

    print("\n================ BASELINE MODEL ================")

    X, y = prepare_features_target(clean_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    baseline_model = build_baseline_model(X_train)
    baseline_model.fit(X_train, y_train)

    evaluate_model(
        baseline_model,
        X_test,
        y_test,
        type='Linear Regression'
    )

    print("\n================ RANDOM FOREST MODEL ================")

    random_forest_model = build_random_forest_model(X_train)
    random_forest_model.fit(X_train, y_train)

    evaluate_model(
        random_forest_model,
        X_test,
        y_test,
        type='Random Forest'
    )

    print("\n================ Fiat 500 Prediction ================")
    
    fiat_500_example = pd.DataFrame({
        "Levy": [clean_df["Levy"].median()],
        "Manufacturer": ["FIAT"],
        "Model": ["500"],
        "Prod. year": [2011],
        "Category": ["Hatchback"],
        "Leather interior": ["No"],
        "Fuel type": ["Petrol"],
        "Engine volume": [1.2],
        "Mileage": [120000],
        "Cylinders": [4.0],
        "Gear box type": ["Manual"],
        "Drive wheels": ["Front"],
        "Doors": ["04-May"],
        "Wheel": ["Left wheel"],
        "Color": ["White"],
        "Airbags": [4],
    })

    # log_prediction = random_forest_model.predict(fiat_500_example)
    lr_prediction = baseline_model.predict(fiat_500_example)
    predicted_price = np.expm1(lr_prediction)

    print(f"Linear regression predicted Fiat 500 price: {predicted_price[0]:,.2f}")

    log_prediction = random_forest_model.predict(fiat_500_example)
    predicted_price = np.expm1(log_prediction)

    print(f"Random forest predicted Fiat 500 price: {predicted_price[0]:,.2f}")


if __name__ == "__main__":
    main()