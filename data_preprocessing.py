import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer
import numpy as np
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from matplotlib.ticker import FuncFormatter

def initial_analysi(data_calender,
                    data_inventory,
                    data_sales,
                    print_analysis = False):

    if print_analysis:
        print("This is the head of the calendar data:\n",data_calender.head())
        print("This is the columns of the calendar data:\n",data_calender.columns)
        print("This is the info of the calendar data:\n",data_calender.info())
        print("This is the describe of the calendar data:",data_calender.describe())
        
        print("This is the head of the inventory data:\n",data_inventory.head())
        print("This is the columns of the inventory data:\n",data_inventory.columns)
        print("This is the info of the inventory data:\n",data_inventory.info())
        print("This is the describe of the inventory data:",data_inventory.describe())

        print("This is the head of the sales data:\n",data_sales.head())
        print("This is the columns of the sales data:\n",data_sales.columns)
        print("This is the info of the sales data:\n",data_sales.info())
        print("This is the describe of the sales data:",data_sales.describe())

    # merge all the data together
    data_temp = pd.merge(data_calender, data_inventory, how = "left", on = "warehouse")
    data_merged = pd.DataFrame(pd.merge(data_sales, data_temp, how = "left", on = ["date", "warehouse","unique_id"]))

    if print_analysis:
        print("This is the head of the merged data:\n",data_merged.head())
        print("This is the columns of the merged data:\n",data_merged.columns)
        print("This is the info of the merged data:\n",data_merged.info())
        print("This is the describe of the merged data:\n",data_merged.describe())
        # Types of the data
        print("This is the types of the merged data:\n",data_merged.dtypes)
    
    # unique_id                   int64     --> int32
    # date                       object     --> datetime
    # warehouse                  object     --> category
    # total_orders              float64     --> float32
    # sales                     float64     --> float32
    # sell_price_main           float64     --> float32
    # availability              float64     --> float32
    # type_0_discount           float64     --> float32
    # type_1_discount           float64     --> float32
    # type_2_discount           float64     --> float32
    # type_3_discount           float64     --> float32      
    # type_4_discount           float64     --> float32
    # type_5_discount           float64     --> float32
    # type_6_discount           float64     --> float32
    # holiday_name               object     --> category
    # holiday                     int64     --> bool
    # shops_closed                int64     --> bool
    # winter_school_holidays      int64     --> bool
    # school_holidays             int64     --> bool
    # product_unique_id           int64     --> int32
    # name                       object     --> category
    # L1_category_name_en        object     --> category
    # L2_category_name_en        object     --> category
    # L3_category_name_en        object     --> category
    # L4_category_name_en        object     --> category
    # date                       object     --> datetime

    data = data_merged.astype({"unique_id": "int32",
                        "warehouse": "category",
                        "total_orders": "float32",
                        "sales": "float32",
                        "sell_price_main": "float32",
                        "availability": "float32",
                        "type_0_discount": "float32",
                        "type_1_discount": "float32",
                        "type_2_discount": "float32",
                        "type_3_discount": "float32",
                        "type_4_discount": "float32",
                        "type_5_discount": "float32",
                        "type_6_discount": "float32",
                        "holiday_name": "category",
                        "holiday": "bool",
                        "shops_closed": "bool",
                        "winter_school_holidays": "bool",
                        "school_holidays": "bool",
                        "product_unique_id": "int32",
                        "L1_category_name_en": "category",
                        "L2_category_name_en": "category",
                        "L3_category_name_en": "category",
                        "L4_category_name_en": "category"})

    data["date"] = pd.to_datetime(data_merged["date"])

    # Print columns whose type has been changed after astype
    print("This is the columns whose type has been changed after analysis:\n")
    for col in data_merged.columns:
        if col in data.columns:
            orig_type = data_merged[col].dtype
            new_type = data[col].dtype
            if orig_type != new_type:
                print(f"Column '{col}': {orig_type} -> {new_type}")

    return data

def impute_KNN(data):
    imputer = KNNImputer(n_neighbors=10)
    data_filled = imputer.fit_transform(data)
    return data_filled

def outlier_detection(data):
    # Isolation Forest
    # Detect outliers on sales within each warehouse group using IsolationForest
    data['outlier_score'] = np.nan
    data['outlier'] = np.nan
    
    # Apply outlier detection only to non-holiday data
    non_holiday_data = data[data['holiday'] == False]
    # IsolationForest outlier detection (tqdm_joblib-loading bar removed, as it doesn't work here)
    for warehouse, group in non_holiday_data.groupby('warehouse'):
        # Reshape for scikit-learn (IsolationForest expects 2D arrays)
        sales_values = group['sales'].values.reshape(-1, 1)
        model_IsolationForest = IsolationForest(n_estimators=20, contamination=0.0075, random_state=42)
        model_IsolationForest.fit(sales_values)
        outlier_scores = model_IsolationForest.decision_function(sales_values)
        outliers = model_IsolationForest.predict(sales_values)
        # Map results back to the original dataframe
        data.loc[group.index, 'outlier_score'] = outlier_scores
        data.loc[group.index, 'outlier'] = outliers
    
    # Mark all holiday data as inliers (not outliers)
    holiday_indices = data[data['holiday'] == True].index
    data.loc[holiday_indices, 'outlier'] = 1  # 1 means inlier
    data.loc[holiday_indices, 'outlier_score'] = 0  # Set a default score for holidays

    # Convert outlier column to boolean: -1 means outlier, 1 means inlier
    data['outlier'] = data['outlier'] == -1
    data_outliers = pd.DataFrame(data[data['outlier'] == True])
    data = pd.DataFrame(data[data['outlier'] == False])
    return data, data_outliers

def cleaing_data(data):
    '''



    '''
    # Drop duplicates
    data = data.drop_duplicates()
    print("This is the duplicates of the merged data:\n", data.duplicated().sum())

    missing_values = data.isnull().sum() / len(data) * 100
    print("This is the missing values of the merged data:\n", missing_values)
    
    # Find columns with less than 5% missing values to impute
    cols_to_impute = data.columns[missing_values >= 5].tolist().remove('holiday_name') 
    print("This is the columns to impute:\n", cols_to_impute)

    # Impute missing values using "impute_function()" function
    data_temp = data.copy()
    if cols_to_impute is not None:
        for col in cols_to_impute:
            data_temp[col] = impute_KNN(data_temp[col])
    else:
        print("No columns to impute")
    
    # There are a few sales that are missing. Fill wtih 0:
    data_temp['sales'] = data_temp['sales'].fillna(0)
    data_temp['total_orders'] = data_temp['total_orders'].fillna(0)


    data_cleaned = data_temp #outlier_detection(data_temp)

    return data_cleaned

def EDA(data, outlier_detection = False):

    data['date'].max() # 2024-06-02 00:00:00
    data['date'].min() # 2020-08-01 00:00:00
    # Difference is 1402 days.
    data['date'].unique() # length is 1402 days, correct!

  
    numeric_data = data.select_dtypes(include=[np.number])
    plt.figure(figsize=(8,6))
    sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap before outlier detection")
    plt.show()

    if outlier_detection:
        
        data, data_outliers = outlier_detection(data)

        # print("These are the outliers:\n", data_outliers)          
        numeric_data = data.select_dtypes(include=[np.number])
        plt.figure(figsize=(8,6))
        sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title("Correlation Heatmap after outlier detection")
        plt.show()


 
    # aggregate sales by week and / or month for each warehouse
    #data['week'] = data['date'].dt.week
    data['year_month'] = data['date'].dt.strftime('%Y-%m')
    #data_weekly_sales = data.groupby(['warehouse', 'week']).agg({'sales': 'sum'}).reset_index()
    data_monthly_sales = data.groupby(['warehouse', 'year_month']).\
                              agg({'sales': 'sum'}).reset_index().\
                              rename(columns={'sales': 'monthly_sales'})
    data = pd.merge(data, data_monthly_sales, how = "left", on = ["warehouse", "year_month"])
    
    # plotting total sales over time for each warehouse.
    # Plotting both sales and monthly_sales curves for all warehouses in different colors without a loop
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=data[data['year_month'] != data.groupby('warehouse')['year_month'].transform('max')], x='date', y='monthly_sales', hue='warehouse', legend='brief', linestyle='-')
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.title("Monthly Sales Over Time by Warehouse")
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x):,}'.replace(',', ' ')))
    plt.tight_layout()
    plt.show()

    


    return data

def main():
    # -----------------------------------------
    # ----------------- Data Loading -----------------
    # -----------------------------------------
    # load dataset
    data_calender = pd.read_csv('data/calendar.csv')
    data_inventory = pd.read_csv('data/inventory.csv')
    data_sales = pd.read_csv('data/sales_train.csv')


    # -----------------------------------------
    # ----------------- Initial Analysis -----------------
    # -----------------------------------------
    # info / summary
    # data types
    data = initial_analysi(data_calender, 
                           data_inventory, 
                           data_sales, 
                           print_analysis = False) 
    

    # -----------------------------------------
    # ----------------- Exploratory Data Analysis -----------------
    # -----------------------------------------    
    # missing values and duplicates
    # outliers detection and handling
    data = cleaing_data(data)

    # visualizations
    # data correlation
    # data distribution
    data = EDA(data)
    
    # -----------------------------------------
    # ----------------- Feature Engineering -----------------
    # -----------------------------------------
    # create new features
    # transform features
    # encode categorical variables
    
    
    # -----------------------------------------
    # ----------------- Feature Selection -----------------
    # -----------------------------------------
    # select relevant features
    # remove multicollinearity
    # feature importance analysis
    
    # -----------------------------------------
    # ----------------- Scaling vs Normalization -----------
    # -----------------------------------------
    # scale numerical features
    # normalize if needed
    
    # -----------------------------------------
    # ----------------- Data Splitting -----------------
    # -----------------------------------------
    # train/test split
    # validation split (if needed)
    # time series split (if applicable)
    
    return data

if __name__ == "__main__":
    main()