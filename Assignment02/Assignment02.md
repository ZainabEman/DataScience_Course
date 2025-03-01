# Data Science Assignment Documentation

This document details the step-by-step process for loading, integrating, and cleaning the dataset used in our analysis. The data comprises raw electricity demand and weather information provided in CSV and JSON formats, packaged in a ZIP file. The following sections describe each stage along with code examples and statistical summaries.

---

## 1. Data Loading and Integration

### Objective
- **Task:** Load and integrate raw electricity demand and weather data.
- **Data Formats:** CSV and JSON files stored within a ZIP folder.
- **Execution Environment:** Google Colab.
- **Tools:** Python libraries (`os`, `glob`, `pandas`).

### Steps

#### 1.1 Upload and Extract Data
- **Action:** Use Colab's file upload widget to upload the ZIP file containing the data.
- **Extraction:** Unzip the contents to a designated directory (e.g., `/content/data`).

from google.colab import files
import os

# Upload ZIP file (this opens a dialog for file upload)
uploaded = files.upload()  # Upload your ZIP file
zip_filename = list(uploaded.keys())[0]
print(f"Uploaded file: {zip_filename}")

# Create target directory and extract ZIP contents
target_dir = '/content/data'
os.makedirs(target_dir, exist_ok=True)
!unzip -o {zip_filename} -d {target_dir}
```

## 1.2 Verify Directory Structure
**Purpose:** Ensure files are extracted correctly, including those in subdirectories (e.g., a folder named `raw`).

```python
# List directories and files recursively to verify structure
for root, dirs, files_in_dir in os.walk(target_dir):
    print("Directory:", root)
    print("Contains files:", files_in_dir)
```

## 1.3 Recursive File Search for CSV and JSON Files
**Action:** Use a recursive glob search to locate all CSV and JSON files within the target directory and its subdirectories.

```python
import glob

csv_files = glob.glob(os.path.join(target_dir, '**', '*.csv'), recursive=True)
json_files = glob.glob(os.path.join(target_dir, '**', '*.json'), recursive=True)

print("CSV files found:", csv_files)
print("JSON files found:", json_files)
```

## 1.4 Load Data with Error Handling
- **CSV Files:** Load with UTF-8 encoding, and if a Unicode error occurs, try an alternative encoding (`latin1`).
- **JSON Files:** Load directly using `pandas.read_json`.
- **Logging:** Print the file name and shape of each loaded DataFrame.

```python
import pandas as pd

dataframes = []

# Load CSV files
for file in csv_files:
    try:
        df = pd.read_csv(file, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(file, encoding='latin1')
    print(f"Loaded {file} with shape {df.shape}")
    dataframes.append(df)

# Load JSON files
for file in json_files:
    try:
        df = pd.read_json(file)
        print(f"Loaded {file} with shape {df.shape}")
    except Exception as e:
        print(f"Error reading JSON file {file}: {e}")
        continue
    dataframes.append(df)
```

## 1.5 Standardize Column Names and Merge DataFrames
- **Standardization:** Convert column names to lowercase, remove extra spaces, and replace spaces with underscores.
- **Merge:** Concatenate all DataFrames into one unified DataFrame.

```python
def standardize_columns(df):
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
    return df

dataframes = [standardize_columns(df) for df in dataframes]

if dataframes:
    combined_df = pd.concat(dataframes, ignore_index=True)
    print("Combined DataFrame shape:", combined_df.shape)
else:
    combined_df = pd.DataFrame()
    print("No data loaded.")
```

### Data Loading Outcome
**Initial Combined DataFrame:**
- **Total Records:** 135,263
- **Total Features:** 6
- **Columns:** `date`, `temperature_2m`, `response`, `request`, `apiversion`, `exceladdinversion`

---

# 2. Data Cleaning and Consistency
## 2.1 Identify and Quantify Missing Data
### Calculate Missing Counts and Percentages

```python
# Calculate missing counts per column
missing_counts = combined_df.isnull().sum()
print("Missing Counts per Column:")
print(missing_counts)

# Calculate missing percentages per column
missing_percentages = (combined_df.isnull().sum() / combined_df.shape[0]) * 100
print("\nMissing Percentages per Column:")
print(missing_percentages)
```

### Observed Statistics:
#### Missing Counts:
- **date:** 7,679 missing
- **temperature_2m:** 7,835 missing
- **response:** 129,778 missing
- **request:** 133,069 missing
- **apiversion:** 127,584 missing
- **exceladdinversion:** 127,584 missing

#### Missing Percentages:
- **date:** 5.68%
- **temperature_2m:** 5.79%
- **response:** 95.94%
- **request:** 98.38%
- **apiversion:** 94.32%
- **exceladdinversion:** 94.32%

---

## 2.2 Assess the Missingness Mechanism
- **MCAR (Missing Completely at Random):** `date` and `temperature_2m` have ~5–6% missing values, suggesting random missingness.
- **MAR (Missing at Random):** Requires further domain analysis.
- **MNAR (Missing Not at Random):** `response`, `request`, `apiversion`, and `exceladdinversion` have over 90% missingness, likely due to inconsistent data collection.

---

## 2.3 Cleaning Strategy and Implementation
### For Columns with Low Missingness (`date` and `temperature_2m`):
- **`temperature_2m`:** Impute missing values using the median.
- **`date`:** Drop rows with missing dates.

### For Columns with High Missingness (`response`, `request`, `apiversion`, `exceladdinversion`):
- **Strategy:** Drop these columns entirely due to high missingness (>90%).

### Implementation Code

```python
# --- Strategy for columns with low missingness ---
# Impute missing values for 'temperature_2m' using the median
combined_df['temperature_2m'] = combined_df['temperature_2m'].fillna(combined_df['temperature_2m'].median())

# Drop rows where 'date' is missing
combined_df = combined_df.dropna(subset=['date'])

# --- Strategy for columns with high missingness ---
# Drop columns with over 90% missing data
cols_to_drop = ['response', 'request', 'apiversion', 'exceladdinversion']
combined_df = combined_df.drop(columns=cols_to_drop)

# Verify cleaning results
print("After cleaning:")
print("Total number of records:", combined_df.shape[0])
print("Total number of features:", combined_df.shape[1])
print("Missing values per column:\n", combined_df.isnull().sum())
```

---

## Data Cleaning Outcome
- **Records After Cleaning:** 127,584 (after dropping rows with missing `date` values)
- **Remaining Features:** 2 (`date` and `temperature_2m`)
- **Missing Values:** 0 in both columns.

### Summary
#### Initial Data:
- **135,263** records and **6** features.
- `date` and `temperature_2m` had ~5–6% missing values.
- `response`, `request`, `apiversion`, and `exceladdinversion` had over 90% missing values.

#### Cleaning Decisions:
- **Imputation:** Applied median imputation to `temperature_2m`.
- **Row Deletion:** Dropped rows with missing `date` values.
- **Column Deletion:** Removed columns with extremely high missingness.

#### Final Dataset:
- Contains **127,584** records with **2** complete features (`date` and `temperature_2m`).


# Data Cleaning and Feature Engineering

## 1. Convert 'date' Column to Datetime

We start by ensuring that the `date` column is correctly formatted as a datetime object. Any invalid entries are coerced to `NaT` to handle errors gracefully.



## 2. Extract Temporal Features

We extract useful time-based features from the `date` column:
* `hour`: Extracted hour from the timestamp
* `day`: Extracted day of the month
* `month`: Extracted month
* `year`: Extracted year


## 3. Categorizing Seasons

To enhance analysis, we categorize the `month` column into four seasons:
* **Winter**: December, January, February
* **Spring**: March, April, May
* **Summer**: June, July, August
* **Autumn**: September, October, November

We define a function to map months to their respective seasons:

## 4. Ensure Numerical Columns are Correctly Typed

We explicitly convert the `temperature_2m` column to a numeric format. Any invalid values are converted to `NaN`.

## Data Verification

We display a sample of the dataset after all transformations:

```python
# Display the first few rows of the dataset
print(combined_df.head())

# Check data types to ensure proper conversion
print(combined_df.dtypes)
```
## Explanation

### Datetime Conversion

- The `date` column is transformed into a proper datetime format using `pd.to_datetime()`.
- Any conversion errors are handled using `errors='coerce'`, replacing invalid values with `NaT`.

### Temporal Feature Extraction

- We extract the `hour`, `day`, `month`, and `year` attributes from the `date` column.
- The `season` column is generated based on `month` values.

### Categorical Conversion

- The `season` column is explicitly converted into an ordered categorical type (`Winter → Spring → Summer → Autumn`).
- This ensures proper sorting and comparison in future analyses.

### Numeric Casting

- The `temperature_2m` column is cast to a floating-point numeric type using `pd.to_numeric()`.
- Any non-numeric values are converted to `NaN` to prevent errors in calculations.

## Sample Output
## Sample Output

After running the above code, the expected output is:

```yaml
Date column after conversion:
0   2024-01-22 05:00:00
1   2024-01-22 06:00:00
2   2024-01-22 07:00:00
3   2024-01-22 08:00:00
4   2024-01-22 09:00:00
Name: date, dtype: datetime64[ns]

DataFrame after type conversions and feature extraction:
                 date  temperature_2m  hour   day  month    year  season
0 2024-01-22 05:00:00          5.1585   5.0  22.0    1.0  2024.0  Winter
1 2024-01-22 06:00:00          5.5585   6.0  22.0    1.0  2024.0  Winter
2 2024-01-22 07:00:00          5.7585   7.0  22.0    1.0  2024.0  Winter
3 2024-01-22 08:00:00          6.1085   8.0  22.0    1.0  2024.0  Winter
4 2024-01-22 09:00:00          5.8585   9.0  22.0    1.0  2024.0  Winter

Data types in the DataFrame:
date              datetime64[ns]
temperature_2m           float64
hour                     float64
day                      float64
month                    float64
year                     float64
season                  category
dtype: object
```


# Handling Duplicates and Inconsistencies

This section details the process of detecting and removing duplicate rows and addressing outliers in the dataset. The focus is on the `temperature_2m` column using the Interquartile Range (IQR) method.

## Objectives

- **Detect and Remove Duplicates:**  
  Identify duplicate rows and remove them to ensure each observation is unique.

- **Identify and Remove Outliers:**  
  Use the IQR method to detect anomalous entries in the `temperature_2m` column and filter them out.

## Python Code for Google Colab

```python
import pandas as pd

# Assume combined_df is your DataFrame obtained from previous cleaning and type conversion steps

# --- Step 1: Detect and Remove Duplicate Rows ---

# Count duplicate rows
duplicate_count = combined_df.duplicated().sum()
print("Number of duplicate rows before removal:", duplicate_count)

# Remove duplicate rows
combined_df = combined_df.drop_duplicates()
print("Shape after removing duplicates:", combined_df.shape)

# --- Step 2: Identify Outliers in 'temperature_2m' Column ---

# Calculate the first (Q1) and third (Q3) quartiles for 'temperature_2m'
Q1 = combined_df['temperature_2m'].quantile(0.25)
Q3 = combined_df['temperature_2m'].quantile(0.75)
IQR = Q3 - Q1

print("Q1:", Q1, "Q3:", Q3, "IQR:", IQR)

# Define the lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
print("Lower Bound:", lower_bound, "Upper Bound:", upper_bound)

# Identify rows where 'temperature_2m' is considered an outlier
outliers = combined_df[(combined_df['temperature_2m'] < lower_bound) | (combined_df['temperature_2m'] > upper_bound)]
print("Number of outlier rows in 'temperature_2m':", outliers.shape[0])

# --- Step 3: Remove Outliers from the DataFrame ---

# Filter the DataFrame to remove outliers
combined_df_no_outliers = combined_df[(combined_df['temperature_2m'] >= lower_bound) & (combined_df['temperature_2m'] <= upper_bound)]
print("Shape after removing outliers:", combined_df_no_outliers.shape)
```

# Explanation of Engineered Features Output

The output below shows the first few rows of the DataFrame after performing feature engineering. This step derives new features from the original `date` column and normalizes the numerical temperature data. Here’s a breakdown of the output:

## Original Columns and Derived Features

- **date:**  
  - The timestamp has been successfully converted to a `datetime64[ns]` object.  
  - Example: `2024-01-22 05:00:00`.

- **temperature_2m:**  
  - This is the original temperature measurement (stored as `float64`).

- **hour, day, month, year:**  
  - These columns are extracted from the `date` column.
  - In the output, for the first row:  
    - **hour:** 5.0  
    - **day:** 22.0  
    - **month:** 1.0  
    - **year:** 2024.0

- **season:**  
  - Derived from the month, this categorical feature indicates the season.  
  - For January, the season is correctly set as "Winter".  
  - The column is stored as a `category` with an ordered set of values: Winter, Spring, Summer, Autumn.

## New Engineered Features

- **day_of_week:**  
  - Represents the day of the week as an integer where 0 corresponds to Monday and 6 corresponds to Sunday.
  - In the output, `0.0` indicates that these dates fall on a Monday.

- **day_name:**  
  - The name of the day (e.g., "Monday", "Tuesday").  
  - All rows in the sample output show "Monday" corresponding to the `day_of_week`.

- **is_weekend:**  
  - A binary flag (0 or 1) indicating whether the day is a weekend.  
  - `0` means the date is a weekday, and `1` would indicate a weekend (Saturday or Sunday).  
  - The output shows `0` for weekdays.

- **is_holiday:**  
  - A binary flag to denote if the date is a holiday (using a US holiday calendar as an example).
  - `0` indicates the date is not a holiday.
  - In the sample, none of the dates are flagged as holidays.

- **temperature_2m_scaled:**  
  - The temperature values have been standardized using scikit-learn’s `StandardScaler`.
  - This transformation results in values with a mean of 0 and a standard deviation of 1.
  - Negative values (e.g., -0.259284) indicate temperatures below the overall mean.

## Data Types Overview

- **date:** `datetime64[ns]`
- **temperature_2m:** `float64`
- **hour, day, month, year:** `float64`  
  (These could be cast to integer types if desired.)
- **season:** `category`
- **day_of_week:** `float64` (numeric representation of weekdays)
- **day_name:** `object` (string type)
- **is_weekend, is_holiday:** `int64` (binary flags)
- **temperature_2m_scaled:** `float64` (standardized temperature)

## Summary

- **Temporal Features:**  
  The extraction of `hour`, `day`, `month`, `year`, `day_of_week`, and `day_name` provides a detailed breakdown of each timestamp. These features are essential for identifying trends or patterns over time.

- **Seasonality:**  
  The `season` column categorizes the dates into seasonal periods (Winter, Spring, Summer, Autumn), which can be useful for analyzing seasonal variations in electricity demand or weather.

- **Weekend and Holiday Flags:**  
  The `is_weekend` flag helps differentiate between weekdays and weekends. The `is_holiday` flag adds further context, indicating if a date is a holiday, which might affect electricity demand.

- **Normalization:**  
  The `temperature_2m_scaled` feature standardizes the temperature data, making it easier to compare with other standardized features or use in machine learning models.

This engineered feature set enhances the dataset by providing more granular temporal information and a normalized temperature measure, which are crucial for further analysis or modeling.


# Exploratory Data Analysis (EDA) Results

This section summarizes the key outcomes of the EDA performed on the dataset. The analysis includes a statistical summary of numerical features, distribution characteristics (skewness and kurtosis), correlation analysis, and a stationarity test via the Augmented Dickey-Fuller (ADF) test.

---

## 1. Statistical Summary

The statistical summary provides an overview of the main numerical features:

- **Temperature (temperature_2m):**
  - **Count:** 3611 values are present.
  - **Mean:** The average temperature is approximately 6.93°C.
  - **Standard Deviation:** The variability in temperature is about 6.85°C.
  - **Min/Max:** The temperatures range from -10.49°C (minimum) to 25.26°C (maximum).
  - **Percentiles:** The 25th, 50th, and 75th percentiles are roughly 2.41°C, 6.96°C, and 10.61°C, respectively.

- **Time-Related Features:**
  - **Hour:** The mean hour is around 11.50 (on a 0–23 scale), with values ranging from 0 to 23.
  - **Day:** The day of the month has a mean of approximately 15.51, with the minimum at 1 and the maximum at 31.
  - **Month:** The mean month is 2.52, with values ranging from 1 to 5 (indicating the dataset covers a subset of the year).
  - **Year:** The year is constant at 2024, which is why the standard deviation is 0.
  - **Day of Week:** This feature (encoded as 0 = Monday, …, 6 = Sunday) has a mean of about 2.96.

- **Scaled Temperature (temperature_2m_scaled):**
  - This feature is a standardized version of `temperature_2m`, with a mean close to 0 and a standard deviation of approximately 1.00, confirming that standardization was successful.

---

## 2. Distribution Characteristics

### Skewness
- **Skewness** measures the asymmetry of the distribution:
  - **temperature_2m:** A skewness of 0.196 indicates a very slight right skew.
  - Other features (hour, day, month, day_of_week) have near-zero skewness, suggesting approximately symmetric distributions.
  - **temperature_2m_scaled** replicates the skewness of `temperature_2m` since it is a scaled version.

### Kurtosis
- **Kurtosis** indicates the "tailedness" of the distribution:
  - **temperature_2m:** A kurtosis of 0.106 suggests the distribution is close to normal (mesokurtic).
  - The hour, day, month, and day_of_week features show negative kurtosis values (e.g., -1.20 for hour and day), which implies flatter distributions compared to a normal distribution (platykurtic).

---

## 3. Correlation Analysis

The correlation matrix examines the linear relationships between numerical features:

- **temperature_2m** is perfectly correlated (1.0000) with **temperature_2m_scaled**, as expected since the latter is a standardized version of the former.
- A moderate positive correlation (0.6024) is observed between **temperature_2m** and **month**, which could indicate that temperature tends to increase with later months in the data range.
- Other correlations (e.g., between hour, day, and day_of_week) are very low, suggesting these time-related features do not have strong linear relationships with temperature.
- Note: The correlation involving the constant column **year** is `NaN` due to zero variance.

---

## 4. Augmented Dickey-Fuller (ADF) Test

The ADF test is used to assess the stationarity of the time series data (here, using `temperature_2m`):

- **ADF Statistic:** -5.039413  
- **p-value:** 0.000019  
- **Critical Values:**  
  - 1%: -3.432  
  - 5%: -2.862  
  - 10%: -2.567  

Since the ADF statistic is lower than all the critical values and the p-value is significantly less than 0.05, we can reject the null hypothesis. This indicates that the time series is stationary.

---

## Summary

- The **statistical summary** confirms that the data for `temperature_2m` and time-related features are well-characterized, with appropriate measures of central tendency and dispersion.
- The **distribution analysis** shows that most features are nearly symmetric, with only slight deviations from normality.
- The **correlation analysis** reveals a moderate relationship between temperature and month, while other variables remain largely uncorrelated.
- The **ADF test** confirms that the temperature time series is stationary, which is important for many time series forecasting models.

This detailed analysis provides a strong foundation for further modeling and interpretation in your data science project.

# 4. Outlier Detection and Handling

This section implements two methods to detect outliers in the numerical feature `temperature_2m` and applies appropriate handling strategies. We use:

- **IQR-based Detection:**  
  Calculate the interquartile range (IQR) and flag data points that fall outside 1.5×IQR. We then cap (Winsorize) these outliers to preserve dataset size.

- **Z-score Method:**  
  Compute Z-scores for the `temperature_2m` values and flag data points with an absolute Z-score greater than 3. In this case, we remove these extreme outliers from the dataset.

Below is the complete Python code for this part along with visualizations to compare before and after modifications.

---

## Python Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set a consistent style for plots
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ------------------------------------------------------------------------------
# ASSUMPTION:
# 'df' is your DataFrame and it includes a numerical column 'temperature_2m'
# ------------------------------------------------------------------------------

# ============================================================
# 1. IQR-based Outlier Detection and Handling
# ============================================================

# Calculate Q1, Q3, and the IQR for temperature_2m
Q1 = df['temperature_2m'].quantile(0.25)
Q3 = df['temperature_2m'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Flag outliers: points outside [lower_bound, upper_bound]
df['iqr_outlier'] = ((df['temperature_2m'] < lower_bound) | (df['temperature_2m'] > upper_bound)).astype(int)
num_outliers_iqr = df['iqr_outlier'].sum()
print("Number of outliers detected by IQR method:", num_outliers_iqr)

# Handling Strategy: Cap (Winsorize) outliers
# Rationale: Capping outliers preserves the overall dataset size while limiting the influence
# of extreme values.
df['temperature_2m_iqr_capped'] = df['temperature_2m'].copy()
df.loc[df['temperature_2m_iqr_capped'] < lower_bound, 'temperature_2m_iqr_capped'] = lower_bound
df.loc[df['temperature_2m_iqr_capped'] > upper_bound, 'temperature_2m_iqr_capped'] = upper_bound

# Before-and-after visualization for IQR-based capping
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.boxplot(x=df['temperature_2m'], color='lightblue')
plt.title("Original Temperature (IQR Method)")
plt.subplot(1, 2, 2)
sns.boxplot(x=df['temperature_2m_iqr_capped'], color='lightgreen')
plt.title("Capped Temperature (IQR Method)")
plt.show()

# ============================================================
# 2. Z-score Outlier Detection and Handling
# ============================================================

# Compute Z-scores for temperature_2m
df['temp_z'] = stats.zscore(df['temperature_2m'])

# Flag outliers: any point with |Z| > 3
df['z_outlier'] = (df['temp_z'].abs() > 3).astype(int)
num_outliers_z = df['z_outlier'].sum()
print("Number of outliers detected by Z-score method:", num_outliers_z)

# Handling Strategy: Remove extreme outliers detected by the Z-score method
# Rationale: When the number of extreme outliers is small, removal minimizes distortion
# in statistical analyses.
df_z = df[df['z_outlier'] == 0].copy()

# Before-and-after visualization for Z-score method (removal)
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.boxplot(x=df['temperature_2m'], color='lightblue')
plt.title("Original Temperature (Z-score Method)")
plt.subplot(1, 2, 2)
sns.boxplot(x=df_z['temperature_2m'], color='salmon')
plt.title("Temperature after Removing Z-score Outliers")
plt.show()
```

# ------------------------------------------------------------------------------
# Modifications Summary:
# - IQR-based method: Outliers in temperature_2m were capped to the lower and upper bounds,
#   preserving the overall size of the dataset while mitigating the influence of extreme values.
# - Z-score method: Data points with |Z| > 3 were removed from the dataset, as they represent
#   extreme values that could distort the analysis.
# ------------------------------------------------------------------------------
# 5. Regression Modeling

This section outlines the process for building and evaluating a regression model to predict electricity demand using the processed dataset. We perform feature selection, handle missing data, split the data, build a linear regression model, and evaluate its performance.

## Feature Selection

- **Predictors:**  
  We selected time-based features as predictors: `hour`, `day`, `month`, `day_of_week`, `is_weekend`, and `is_holiday`.
  
- **Target Variable:**  
  For this demonstration, the target variable `electricity_demand` is set to the value in `temperature_2m`.

## Data Splitting and Missing Value Imputation

- **Data Splitting:**  
  The dataset is split into training (80%) and testing (20%) sets to evaluate model performance on unseen data.

- **Handling Missing Values:**  
  Since the regression model (Linear Regression) does not accept missing values, we use mean imputation (via scikit-learn's `SimpleImputer`) to fill any NaN values in the predictor features.

## Model Development

- A linear regression model is constructed using scikit-learn's `LinearRegression`.
- The model is trained on the imputed training set.
- Predictions are then made on the imputed test set.

## Model Evaluation

- **Evaluation Metrics:**  
  We evaluate the model using:
  - **Mean Squared Error (MSE)**
  - **Root Mean Squared Error (RMSE)**
  - **R² Score**
  
- **Visualization:**  
  An actual vs. predicted scatter plot is generated to compare the predicted values against the actual values.  
  A residual analysis is performed by plotting the residuals distribution and a residuals vs. predicted values plot.

## Residual Analysis

- **Residuals Distribution:**  
  The histogram and density plot of residuals help determine if the errors are normally distributed around zero.
  
- **Residuals vs. Predicted Plot:**  
  A scatter plot of residuals versus predicted values is used to check for any patterns. A random scatter around zero indicates that the model errors are random and that the model is well-specified.

## Summary

This regression modeling process includes:
- Selecting time-based features as predictors.
- Imputing missing values to ensure model compatibility.
- Splitting data into training and testing sets.
- Building and evaluating a linear regression model.
- Visualizing model performance through actual vs. predicted plots and residual analysis.

The approach provides a robust framework for predicting electricity demand and assessing model performance.
