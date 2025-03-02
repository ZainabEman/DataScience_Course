import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

# Set Streamlit page configuration
st.set_page_config(page_title="Electricity Demand Prediction Project", layout="wide")

# ------------------------------------------------------------------------------
# Helper Function to Make DataFrame Displayable
# ------------------------------------------------------------------------------
def make_displayable(df):
    df_display = df.copy()
    # Convert datetime columns to string to avoid Arrow conversion issues
    for col in df_display.columns:
        if pd.api.types.is_datetime64_any_dtype(df_display[col]):
            df_display[col] = df_display[col].astype(str)
    return df_display

# ------------------------------------------------------------------------------
# Load Data (simulate a processed DataFrame for demonstration)
# ------------------------------------------------------------------------------
@st.cache_data
def load_data():
    # Create a date range and simulate data (using 'h' for hourly frequency)
    dates = pd.date_range(start="2024-01-01", periods=100, freq='h')
    data = {
        "date": dates,
        "temperature_2m": np.random.uniform(low=-10, high=25, size=len(dates)),
        "hour": dates.hour,
        "day": dates.day,
        "month": dates.month,
        "year": dates.year,
        "day_of_week": dates.dayofweek,
        "is_weekend": [1 if x >= 5 else 0 for x in dates.dayofweek],
        "is_holiday": np.zeros(len(dates)),  # For demo, no holidays
    }
    df = pd.DataFrame(data)
    # Create a scaled temperature column (simulate standardized values)
    df["temperature_2m_scaled"] = (df["temperature_2m"] - df["temperature_2m"].mean()) / df["temperature_2m"].std()
    # For regression target, assume electricity_demand equals temperature_2m (for demo)
    df["electricity_demand"] = df["temperature_2m"]
    return df

df = load_data()

# ------------------------------------------------------------------------------
# Sidebar Navigation
# ------------------------------------------------------------------------------
st.sidebar.title("Navigation")
section = st.sidebar.radio("Select Section:", 
                           ["Overview",
                            "Data Cleaning & Consistency",
                            "Type Conversions & Feature Engineering",
                            "Exploratory Data Analysis",
                            "Outlier Detection & Handling",
                            "Regression Modeling"])

# ------------------------------------------------------------------------------
# Updated Data Loading and Helper Functions
# ------------------------------------------------------------------------------
@st.cache_data
def load_data():
    # Create a date range with 200 hourly periods to cover multiple days
    dates = pd.date_range(start="2024-01-01", periods=200, freq='h')
    data = {
        "date": dates,
        "temperature_2m": np.random.uniform(low=-10, high=25, size=len(dates)),
        "hour": dates.hour,
        "day": dates.day,
        "month": dates.month,
        "year": dates.year,
        "day_of_week": dates.dayofweek,
        "is_weekend": [1 if x >= 5 else 0 for x in dates.dayofweek],
        "is_holiday": np.zeros(len(dates)),  # For demo, no holidays
    }
    df = pd.DataFrame(data)
    # Create a scaled temperature column (simulate standardized values)
    df["temperature_2m_scaled"] = (df["temperature_2m"] - df["temperature_2m"].mean()) / df["temperature_2m"].std()
    # For regression target, assume electricity_demand equals temperature_2m (for demo)
    df["electricity_demand"] = df["temperature_2m"]
    return df

def make_displayable(df):
    df_display = df.copy()
    # Convert datetime columns to formatted strings to display full date and time
    for col in df_display.columns:
        if pd.api.types.is_datetime64_any_dtype(df_display[col]):
            df_display[col] = df_display[col].dt.strftime("%Y-%m-%d %H:%M:%S")
    return df_display

df = load_data()

# ------------------------------------------------------------------------------
# Section 1: Data Loading & Integration
# ------------------------------------------------------------------------------
if section == "Overview":
    st.title("Overview")
    st.markdown("""
    ## Data Loading and Integration

    **Objective:**
    - **Task:** Load and integrate raw electricity demand and weather data.
    - **Data Formats:** CSV and JSON files stored within a ZIP folder.
    - **Execution Environment:** Google Colab.
    - **Tools:** Python libraries (`os`, `glob`, `pandas`).

    **Steps:**

    1. **Upload and Extract Data:**
       - Use Colab's file upload widget to upload the ZIP file containing the data.
       - Unzip the contents to a designated directory (e.g., `/content/data`).

    2. **Verify Directory Structure:**
       - Ensure that the files are extracted correctly, including those in subdirectories (for example, a folder named `raw`).

    3. **Recursive File Search:**
       - Use a recursive glob search to locate all CSV and JSON files within the target directory and its subdirectories.

    4. **Load Data with Error Handling:**
       - Load CSV files using UTF-8 encoding (with fallback to `latin1` if needed).
       - Load JSON files using `pandas.read_json`.
       - Print file names and shapes for verification.

    5. **Standardize Column Names and Merge DataFrames:**
       - Convert column names to lowercase, trim extra spaces, and replace spaces with underscores.
       - Concatenate all DataFrames into one unified DataFrame.

    **Data Loading Outcome:**
    - **Total Records:** 135,263
    - **Total Features:** 6
    - **Columns:** `date`, `temperature_2m`, `response`, `request`, `apiversion`, `exceladdinversion`

    **Next Steps:**
    The processed data will be further cleaned, transformed, and enriched with additional features in the subsequent sections:
    - **Data Cleaning & Consistency**
    - **Data Type Conversions**
    - **Feature Engineering**
    - **Exploratory Data Analysis (EDA)**
    - **Outlier Detection & Handling**
    - **Regression Modeling**
    """)
    st.write("### DataFrame Overview:")
    st.write("Original DataFrame Shape:", df.shape)
    # Filter out constant columns for a more informative preview
    informative_columns = [col for col in df.columns if df[col].nunique() > 1]
    st.write("Informative Columns (with variability):", informative_columns)
    st.dataframe(make_displayable(df[informative_columns].head()))
# ------------------------------------------------------------------------------
# Section 2: Data Cleaning & Consistency
# ------------------------------------------------------------------------------
elif section == "Data Cleaning & Consistency":
    st.title("Data Cleaning & Consistency")
    st.write("This section shows how missing values were identified and handled.")
    
    # Hardcoded missing value statistics from the markdown file
    data_cleaning_stats = pd.DataFrame({
        "Column": ["date", "temperature_2m", "response", "request", "apiversion", "exceladdinversion"],
        "Missing Count": [7680, 7835, 129778, 133069, 127584, 127584],
        "Missing Percentage": ["5.68%", "5.79%", "95.94%", "98.38%", "94.32%", "94.32%"]
    })
    
    st.write("### Missing Value Summary:")
    st.dataframe(data_cleaning_stats)
    
    # Create a bar chart showing the missing counts for each column
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="Column", y="Missing Count", data=data_cleaning_stats, palette="viridis", ax=ax)
    ax.set_title("Missing Value Counts per Feature")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Missing Count")
    # Add data labels on the bars
    for index, row in data_cleaning_stats.iterrows():
        ax.text(index, row["Missing Count"], f'{row["Missing Count"]}', color="black", ha="center", va="bottom")
    st.pyplot(fig)
    
# ------------------------------------------------------------------------------
# Section 3: Data Type Conversions & Feature Engineering
# ------------------------------------------------------------------------------

elif section == "Type Conversions & Feature Engineering":
    st.title("Data Cleaning, Data Type Conversions & Feature Engineering")
    st.markdown("""
    ## 1. Convert 'date' Column to Datetime

    We start by ensuring that the `date` column is correctly formatted as a datetime object.
    Any invalid entries are coerced to `NaT` to handle errors gracefully.

    ## 2. Extract Temporal Features

    We extract useful time-based features from the `date` column:
    - **hour:** Extracted hour from the timestamp.
    - **day:** Extracted day of the month.
    - **month:** Extracted month.
    - **year:** Extracted year.

    ## 3. Categorizing Seasons

    To enhance analysis, we categorize the `month` column into four seasons:
    - **Winter:** December, January, February.
    - **Spring:** March, April, May.
    - **Summer:** June, July, August.
    - **Autumn:** September, October, November.

    A function is defined to map months to their respective seasons. The resulting `season`
    column is then explicitly converted into an ordered categorical type (`Winter → Spring → Summer → Autumn`),
    ensuring proper sorting and comparison in future analyses.

    ## 4. Ensure Numerical Columns are Correctly Typed

    We explicitly convert the `temperature_2m` column to a numeric format using `pd.to_numeric()`.
    Any non-numeric values are converted to `NaN` to prevent errors in calculations.

    ## Data Verification

    We display a sample of the dataset after all transformations along with the data types.
    """)

    st.write("### Sample Output After Conversions and Feature Extraction:")
    # Filter to display only informative columns
    informative_columns = [col for col in df.columns if df[col].nunique() > 1]
    st.dataframe(make_displayable(df[informative_columns].head()))

    st.write("### Data Types:")
    st.write(df[informative_columns].dtypes.astype(str))
    
    st.markdown("""
    ## Feature Engineering

    New features were engineered from the timestamp to enhance our analysis:
    - **Day of Week:** Numeric value (0 = Monday, …, 6 = Sunday)
    - **Is Weekend:** Binary flag indicating weekends.
    - **Is Holiday:** Binary flag (for demo, no holidays).
    - **Normalized Temperature:** Standardized `temperature_2m` values.
    """)
    st.image("Engineered Feature.png", caption="Engineered Feature", use_container_width=True)


# ------------------------------------------------------------------------------
# Section 5: Exploratory Data Analysis (EDA)
# ------------------------------------------------------------------------------
elif section == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis (EDA)")
    
    # Statistical Summary
    st.subheader("Statistical Summary")
    st.code("""=== Statistical Summary ===
       temperature_2m         hour          day        month    year  
count     3611.000000  2923.000000  2923.000000  2923.000000  2923.0   
mean         6.933715    11.496408    15.508040     2.520698  2024.0   
std          6.847542     6.928616     8.802778     1.142109     0.0   
min        -10.491500     0.000000     1.000000     1.000000  2024.0   
25%          2.408500     5.000000     8.000000     1.000000  2024.0   
50%          6.958500    11.000000    15.000000     3.000000  2024.0   
75%         10.608500    18.000000    23.000000     4.000000  2024.0   
max         25.258501    23.000000    31.000000     5.000000  2024.0   

       day_of_week  temperature_2m_scaled  
count  2923.000000           3.611000e+03  
mean      2.959631           9.445043e-17  
std       1.999764           1.000138e+00  
min       0.000000          -2.545093e+00  
25%       1.000000          -6.609440e-01  
50%       3.000000           3.620091e-03  
75%       5.000000           5.367320e-01  
max       6.000000           2.676482e+00  

Skewness of numerical features:
temperature_2m           0.196583
hour                     0.001477
day                      0.010766
month                    0.032511
year                     0.000000
day_of_week              0.031919
temperature_2m_scaled    0.196583
dtype: float64

Kurtosis of numerical features:
temperature_2m           0.106266
hour                    -1.206587
day                     -1.196645
month                   -1.277384
year                     0.000000
day_of_week             -1.249512
temperature_2m_scaled    0.106266
dtype: float64
""")

    # Boxplot
    st.subheader("Boxplot of Temperature")
    st.image("BoxPlot of temprature.png", caption="Boxplot of Temperature", use_container_width=True)

    # Density Plot
    st.subheader("Density Plot of Temperature")
    st.image("Density plot of temprature.png", caption="Density Plot of Temperature", use_container_width=True)

    # Electricity Demand Over Time
    st.subheader("Electricity Demand (Temperature) Over Time")
    st.image("electricity demand overtime.png", caption="Electricity Demand (Temperature) Over Time", use_container_width=True)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    st.image("Heatmap.png", caption="Correlation Heatmap", use_container_width=True)

    # Histogram and Density Plot
    st.subheader("Histogram and Density Plot of Temperature")
    st.image("histogram and density plot of temprature.png", caption="Histogram & Density Plot of Temperature", use_container_width=True)

    # Correlation Matrix
    st.subheader("Correlation Matrix")
    st.code("""=== Correlation Matrix ===
                       temperature_2m      hour       day     month  year
temperature_2m               1.000000  0.120903 -0.020858  0.602388   NaN
hour                         0.120903  1.000000 -0.000753 -0.008889   NaN
day                         -0.020858 -0.000753  1.000000 -0.047324   NaN
month                        0.602388 -0.008889 -0.047324  1.000000   NaN
year                              NaN       NaN       NaN       NaN   NaN
day_of_week                  0.046259 -0.002950  0.025253  0.006659   NaN
temperature_2m_scaled        1.000000  0.120903 -0.020858  0.602388   NaN

                       day_of_week  temperature_2m_scaled
temperature_2m            0.046259               1.000000
hour                     -0.002950               0.120903
day                       0.025253              -0.020858
month                     0.006659               0.602388
year                           NaN                    NaN
day_of_week               1.000000               0.046259
temperature_2m_scaled     0.046259               1.000000
""")

    # Augmented Dickey-Fuller Test
    st.subheader("Augmented Dickey-Fuller Test")
    st.code("""=== Augmented Dickey-Fuller Test ===
ADF Statistic: -5.039413
p-value: 0.000019
Critical Values:
   1%: -3.432
   5%: -2.862
   10%: -2.567
""")


# ------------------------------------------------------------------------------
# Section 6: Outlier Detection & Handling
# ------------------------------------------------------------------------------
elif section == "Outlier Detection & Handling":
    st.title("Outlier Detection & Handling")
    st.write("This section demonstrates two outlier detection methods: IQR-based and Z-score based methods.")
    
    st.write("### IQR-based Outlier Detection:")
    
    st.image("Z-score Method & Z-score Outliers.png", caption="Z-score Outlier Detection", use_container_width=True)

# ------------------------------------------------------------------------------
# Section 7: Regression Modeling
# ------------------------------------------------------------------------------
elif section == "Regression Modeling":
    st.title("Regression Modeling")
    st.write("This section builds and evaluates a regression model to predict electricity demand.")

    st.write("### Feature Selection:")
    st.write("Predictors: `hour`, `day`, `month`, `day_of_week`, `is_weekend`, `is_holiday`")
    st.write("Target: `electricity_demand` (equals `temperature_2m` for this demo)")
    
    
    st.write("### Actual vs. Predicted Electricity Demand:")
    st.image("Actual Electricity Demand scatter plot.png", caption="Actual vs. Predicted Scatter Plot", use_container_width=True)
    
    st.write("### Residual Analysis:")
    st.image("Actual vs. Predicted scatter plot..png", caption="Residual Analysis (Histogram & Residuals vs. Predicted)", use_container_width=True)
    
    st.write("### Prediction Graphs Description:")
    st.markdown("""
    - **Actual vs. Predicted Scatter Plot:**  
      This plot shows the relationship between the actual electricity demand and the model's predictions. A red dashed line indicates the ideal scenario where the predictions perfectly match the actual values. Any deviation from this line highlights prediction errors.
      
    - **Residual Analysis Plot:**  
      The residual analysis includes a histogram of the residuals (the differences between the actual and predicted values) and a scatter plot of residuals versus predicted values. Ideally, residuals should be randomly distributed around zero, indicating that the model errors are random and that the model is well-fitted.
    """)


# ------------------------------------------------------------------------------
# Footer / Project Information - Interactive UI
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Footer / Project Information (Interactive)
# ------------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("""
### Project Links

[<img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" width="25" style="vertical-align: middle; margin-right: 5px;"> GitHub Repository](https://github.com/yourusername/your-repo)

[<img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" style="vertical-align: middle; margin-right: 5px;"> LinkedIn](https://www.linkedin.com/in/yourprofile)

[<img src="https://cdn-icons-png.flaticon.com/512/2111/2111505.png" width="25" style="vertical-align: middle; margin-right: 5px;"> Medium Blog](https://medium.com/@yourusername)

[<img src="https://cdn-icons-png.flaticon.com/512/5968/5968672.png" width="25" style="vertical-align: middle; margin-right: 5px;"> Documentation](https://yourdocumentationlink.com)
""", unsafe_allow_html=True)

