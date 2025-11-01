import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest

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

    # Duplicates
    print("This is the duplicates of the merged data:\n",data.duplicated().sum())

    return data

def impute_function(data):
    print("This is the data to impute:\n", data)
    return data_filled


def drop_outliers(data):
    print("This is the data to drop outliers:\n", data)
    return data_dropped

def cleaing_data(data):
    '''



    '''
    # Drop duplicates
    data = data.drop_duplicates()
    print("This is the duplicates of the merged data:\n", data.duplicated().sum())

    missing_values = data.isnull().sum() / len(data) * 100
    print("This is the missing values of the merged data:\n", missing_values)
    
    # Find columns with less than 5% missing values to impute
    cols_to_impute = data.columns[missing_values < 5].tolist()
    print("This is the columns to impute:\n", cols_to_impute)

    # Impute missing values using "impute_function()" function
    data_temp = data.copy()
    for col in cols_to_impute:
        data_temp[col] = impute_function(data_temp[col])
    
    data_cleaned = drop_outliers(data_temp)

    return data_cleaned

def EDA(data):


    # histogram of the data
    data.hist(bins=100) # Nothing to see here

    # correlation matrix
    corr = data.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.show()


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
    EDA(data)
    
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
    
    return True

if __name__ == "__main__":
    main()