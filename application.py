import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import missingno as msno
from datetime import date
from joblib import PrintTime
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.neighbors import LocalOutlierFactor
from statsmodels.stats.proportion import proportions_ztest

from feature_engineering import cat_cols, cat_but_car, rare_encoder

######################################
## Titanic veris seti uçtan uca feature engineering & data preproccesing
########################################

df = pd.read_csv("datasets/Titanic-Dataset.csv")

df.columns = [col.upper() for col in df.columns]

######################################
## 1.Features Engineering (Değişken Mühendisliği)
########################################

#Cabin Bool
df["NEW_CABIN_BOOL"] = df["CABIN"].notnull().astype("int")

#Name Count
df["NEW_NAME_COUNT"] = df["NAME"].str.len()

#Name Word Count
df["NEW_NAME_WORD_COUNT"] = df["NAME"].apply(lambda x: len(str(x).split(" ")))

#DR
df["NEW_NAME_DR"] = df["NAME"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))

#Name Title
df["NEW_TITLE"] = df.NAME.str.extract(' ([A-Za-z]+)\.', expand=False)

#family size
df["NEW_FAMILY_SIZE"] = df["SIBSP"] + df["PARCH"] + 1

#age_pclass
df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]

#is alone
df.loc[((df['SIBSP'] + df['PARCH']) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df['SIBSP'] + df['PARCH']) == 0), "NEW_IS_ALONE"] = "YES"

#age level
df.loc[(df["SEX"] == "male") & (df["AGE"] <= 21), 'NEW_SEX_CAT'] = "youngmale"

df.loc[(df["SEX"] == "male") & (df["AGE"] > 21) & (df["AGE"] <= 50), 'NEW_SEX_CAT'] = "maturemale"

df.loc[(df["SEX"] == "male") & (df["AGE"] > 50), 'NEW_SEX_CAT'] = "seniormale"

df.loc[(df["SEX"] == "female") & (df["AGE"] <= 21), 'NEW_SEX_CAT'] = "youngfemale"

df.loc[(df["SEX"] == "female") & (df["AGE"] > 21) & (df["AGE"] <= 50), 'NEW_SEX_CAT'] = "maturefemale"

df.loc[(df["SEX"] == "female") & (df["AGE"] > 50), 'NEW_SEX_CAT'] = "seniorfemale"

def grab_col_names(dataframe, cat_th = 10, car_th = 20):

    #cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in dataframe.columns if col not in cat_but_car]

    #num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[0]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if "PASSENGERID" not in col]

######################################
## 2.Outliers (Aykırı Değerler)
########################################
def outlier_thresholds(dataframe , col_name, q1=0.25, q3=0.75):
    quantile1 = dataframe[col_name].quantile(q1)
    quantile3 = dataframe[col_name].quantile(q3)
    interquantile_range  = quantile3 - quantile1
    up_limit = quantile3 + 1.5 * interquantile_range
    low_limit = quantile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low , up = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up) | (dataframe[col_name] < low)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable):
    low, up = outlier_thresholds(dataframe, variable)
    dataframe.loc[dataframe[variable] > up, variable] = up
    dataframe.loc[dataframe[variable] < low, variable] = low


for col in num_cols:
    print(col, check_outlier(df, col))

for col in num_cols:
    replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))

######################################
## 3.Missing Values (Eksik Değerler)
########################################
def missing_values(dataframe, na_name=False):
    na_cols = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_cols].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_cols].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio,2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end='\n\n')
    if na_name:
        return na_cols

missing_values(df)
df.drop("CABIN", axis=1, inplace=True)
remove_cols = ["TICKET", "NAME"]
df.drop(remove_cols, axis=1, inplace=True)

df["AGE"] = df["AGE"].fillna(df.groupby("NEW_TITLE")["AGE"].transform("median"))

df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]

df.loc[(df["SEX"] == "male") & (df["AGE"] <= 21), 'NEW_SEX_CAT'] = "youngmale"

df.loc[(df["SEX"] == "male") & (df["AGE"] > 21) & (df["AGE"] <= 50), 'NEW_SEX_CAT'] = "maturemale"

df.loc[(df["SEX"] == "male") & (df["AGE"] > 50), 'NEW_SEX_CAT'] = "seniormale"

df.loc[(df["SEX"] == "female") & (df["AGE"] <= 21), 'NEW_SEX_CAT'] = "youngfemale"

df.loc[(df["SEX"] == "female") & (df["AGE"] > 21) & (df["AGE"] <= 50), 'NEW_SEX_CAT'] = "maturefemale"

df.loc[(df["SEX"] == "female") & (df["AGE"] > 50), 'NEW_SEX_CAT'] = "seniorfemale"

df = df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == 'object' and len(x.unique()) <= 10) else x, axis=0)

######################################
## 4.Label Encoding
########################################
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtype not in [int , float]
               and df[col].nunique() == 2]
for col in binary_cols:
    label_encoder(df, col)


######################################
## 5.Rare Encoding
########################################
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT" : dataframe[col].value_counts(),
                            "RATIO" : dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN" : dataframe.groupby(col)[target].mean()}), end="\n\n")
def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtype == 'object'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis = None)]
    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df

rare_analyser(df, "SURVIVED", cat_cols)

df = rare_encoder(df,0.01)
df["NEW_TITLE"].value_counts()

######################################
## 6.One Hot Encoding
########################################
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns = categorical_cols, drop_first=drop_first)
    return dataframe
ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
df = one_hot_encoder(df, ohe_cols)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if "PASSENGERID" not in col]
rare_analyser(df, "SURVIVED", cat_cols)

useless_cols = [col for col in df.columns if df[col].nunique() == 2 and
                (df[col].value_counts() / len(df) < 0.01).any(axis=None)]
#df.drop(useless_cols, axis=1, inplace=True)

######################################
## 7.Standard Scaler
########################################
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df[num_cols].head()

######################################
## 7.Model
########################################
y = df["SURVIVED"]
X = df.drop(["PASSENGERID", "SURVIVED"], axis=1)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(x_train, y_train)
y_pred = rf_model.predict(x_test)
accuracy_score(y_pred, y_test)