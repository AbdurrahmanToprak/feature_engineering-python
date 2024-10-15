from pydoc import describe

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

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

def load_big_data():
    data = pd.read_csv('datasets/data.csv')
    return data
df = load_big_data()
print(df.head())

print(df.info())

####################  kutu grafiği  #######################################

sns.boxplot(x=df["Model Year"])
plt.show()

sns.boxplot(x=df["Legislative District"])
plt.show()

####################  aykırı değer  #######################################

q1 = df["Model Year"].quantile(0.25)

q3 = df["Model Year"].quantile(0.75)

iqr = q3 - q1

up = q3 + 1.5 * iqr
low = q1 - 1.5 * iqr
print(low, up) ############### eşik değerler

####################  aykırı değerleri getir  #######################################
print(df[(df["Model Year"] > up) | (df["Model Year"] < low)])

####################  aykırı değerlerin indexlerini getir  #######################################
print(df[(df["Model Year"] > up) | (df["Model Year"] < low)].index)

####################  aykırı değer var mı yok mu?  #######################################
print(df[(df["Model Year"] > up) | (df["Model Year"] < low)].any(axis=None))

######################### fonksiyonlaştırma eşik değer hesaplama##########################
def outlier_thresholds(dataframe , col_name, q1=0.25, q3=0.75):
    quantile1 = dataframe[col_name].quantile(q1)
    quantile3 = dataframe[col_name].quantile(q3)
    interquantile_range  = quantile3 - quantile1
    up_limit = quantile3 + 1.5 * interquantile_range
    low_limit = quantile1 - 1.5 * interquantile_range
    return low_limit, up_limit
print(outlier_thresholds(df, "Model Year"))
print(outlier_thresholds(df, "Legislative District"))

######################### fonksiyonlaştırma aykırı değer var mı yok mu? ##########################
def check_outlier(dataframe, col_name):
    low , up = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up) | (dataframe[col_name] < low)].any(axis=None):
        return True
    else:
        return False

print(check_outlier(df, "Legislative District"))
print(check_outlier(df, "Model Year"))

######################### sütun değişkenlerini tutma fonksiyonu ##########################
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

for col in num_cols:
    print(col, check_outlier(df, col))

def grab_outliers(dataframe, col_name , index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] > up) | (dataframe[col_name] < low))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] > up) | (dataframe[col_name] < low))].head())
    else:
        print(dataframe[((dataframe[col_name] > up) | (dataframe[col_name] < up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] > up) | (dataframe[col_name] < low))].index
        return outlier_index

grab_outliers(df, "Model Year")
grab_outliers(df, "Model Year", index=True)
########## silme ###############
def remove_outliers(dataframe, col_name):
    low, up = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] > up) | (dataframe[col_name] < low))]
    return df_without_outliers
for col in num_cols:
    new_df = remove_outliers(df, col)
print(df.shape[0] - new_df.shape[0])
print(new_df.head())

############# baskılama ############
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


################### çok değişkenli aykırı değer #####################33
dff = sns.load_dataset("diamonds")
dff = dff.select_dtypes(include=['int64', 'float64'])
dff = dff.dropna()
print(dff.head())
print(dff.columns)
print(dff.info())
for col in dff.columns:
    print(col, check_outlier(dff, col))

clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(dff)

dff_scores = clf.negative_outlier_factor_
dff_scores[0:5]
np.sort(dff_scores)

scores = pd.DataFrame(np.sort(dff_scores))
scores.plot(stacked=True, xlim = [0,50], style='.-')
plt.show()

th = np.sort(dff_scores)[3]
dff[dff_scores < th]
dff.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T

dff[dff_scores < th].drop(axis = 0, labels=dff[dff_scores < th].index)

dff.isnull().values.any()
dff.isnull().sum()
df.isnull().values.any()
df.isnull().sum()
#toplam eksik değer sayısı
df.isnull().sum().sum()
#en az bir eksik değere sahip olan gözlem değerleri
print(df[df.isnull().any(axis=1)])
#hepsi tam olan değerler
print(df[df.notnull().all(axis=1)])

(df.isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)


def missing_values(dataframe, na_name=False):
    na_cols = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_cols].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_cols].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio,2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end='\n\n')
    if na_name:
        return na_cols
print(missing_values(df))

df2 = df.apply(lambda x: x.fillna(x.mean()) if x.dtypes != 'object' else x, axis=0).head()
df2.isnull().sum()
df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == 'object' and len(x.unique()) <= 10) else x, axis=0).isnull().sum()

df.groupby("Make")["Model Year"].mean()

df["Model Year"].fillna(df.groupby("Make")["Model Year"].transform("mean")).isnull().sum()

df = load_big_data()

cat_cols, num_cols, cat_but_car = grab_col_names(df)

dff = pd.get_dummies(df[num_cols] , drop_first=True)
dff.head()

#değişkenlerin standartlaştırılması
scaler = MinMaxScaler()
dff= pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
dff.head()

#knn uygulanması
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
dff.head()

#standartlaştırmayı geri alma
dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)

# Tek bir sütun olarak atama yapıyoruz
df["Postal Code_imputed_knn"] = dff["Postal Code"]

# Null değerler için kontrol
df2 = df.loc[df["Postal Code"].isnull(), ["Postal Code", "Postal Code_imputed_knn"]]
print(df2)
df2 = df.loc[df["Postal Code"].isnull()]
print(df2)

###### eksik verilerin yapısını incelemek
msno.bar(df)
plt.show()

msno.heatmap(df)
plt.show()

#eksik değerlerin bağımlı değişken ile ilişkisinin incelenmesi

missing_values(df, True)
na_cols  = missing_values(df, True)

def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains('_NA_')].columns
    for col in na_flags:
        print(pd.DataFrame({'TARGET_MEAN': temp_df.groupby(col)[target].mean(),
                            'Count' : temp_df.groupby(col)[target].count()}), end='\n\n')

missing_vs_target(df, "Model Year", na_cols)

##### Label Encoding
df = sns.load_dataset("titanic")
print(df["sex"].head())

le = LabelEncoder()
print(le.fit_transform(df["sex"])[0:5])
le.inverse_transform([0,1])

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes not in [int , float]
               and df[col].nunique() == 2]
print(binary_cols)
for col in binary_cols:
    label_encoder(df, col)

#one hot encoding
pd.get_dummies(df, columns=["embarked"], drop_first=True).head()
pd.get_dummies(df, columns=["embarked"], dummy_na=True).head()
pd.get_dummies(df, columns=["embarked","sex"], drop_first=True).head()

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns = categorical_cols, drop_first=drop_first)
    return dataframe
cat_cols, num_cols, cat_but_car = grab_col_names(df)

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique()> 2]
one_hot_encoder(df, ohe_cols).head()

### kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi
cat_cols, num_cols, cat_but_car = grab_col_names(df)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col : dataframe[col_name].value_counts(),
                        "Ratio" : 100* dataframe[col_name].value_counts() / len(dataframe)}))
    print("#####################################")
    if plot:
        sns.countplot(x = dataframe[col_name], data=dataframe)
        plt.show()
for col in cat_cols:
    cat_summary(df, col)

## rare kategoriler ile bağımlı değişken arasındaki ilişkinin analizi
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT" : dataframe[col].value_counts(),
                            "RATIO" : dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN" : dataframe.groupby(col)[target].mean()}), end="\n\n")
rare_analyser(df, "age", cat_cols)

#rare encoder fonksiyonu

def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtype == 'object'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis = None)]
    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df
new_df = rare_encoder(df , 0.01)

rare_analyser(new_df, "age", cat_cols)


######## Feature Scaling (Özellik Ölçeklendirme)

## StandardScaler: Klasik srandartlaştırma. Ortalamayı çıkar, standart sapmaya böl z = (x - u) / s

df = sns.load_dataset("titanic")
ss = StandardScaler()
df["age_standard_scaler"] = ss.fit_transform(df[["age"]])
df.head()

## RobustScaler : Medyanı çıkar, iqr a böl

rs = RobustScaler()
df["age_robust_scaler"] = rs.fit_transform(df[["age"]])
df.describe().T

## MinMaxScaler : Verilen 2 değer arasında değişken dönüşümü

mms = MinMaxScaler()
df["age_minmax_scaler"] = mms.fit_transform(df[["age"]])
df.describe().T

## Görselleştirme

age_cols = [col for col in df.columns if "age" in col]

def num_summary(dataframe, numarical_cols, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numarical_cols].describe().T)

    if plot:
        dataframe[numarical_cols].hist(bins=20)
        plt.xlabel(numarical_cols)
        plt.title(numarical_cols)
        plt.show()

for col in age_cols:
    num_summary(df, col, plot=True)

## Numeric to Categorical: Sayısal değişkenleri kategorik değişkenlere çevirme
## Binning

df["age_qcut"] = pd.qcut(df["age"], 5)

## Features Extraction (Özellik Çıkarımı)
################################

## Binary Features : FLag, Bool, True-False

df = sns.load_dataset("titanic")
df["NEW_DECK_BOOL"] = df["deck"].notnull().astype("int")

df.groupby("NEW_DECK_BOOL").agg({"survived": "mean"})

from statsmodels.stats.proportion import proportions_ztest

test_stat , pvalue = proportions_ztest(count=[df.loc[df["NEW_DECK_BOOL"] == 1, "survived"].sum(),
                                              df.loc[df["NEW_DECK_BOOL"] == 0, "survived"].sum()],

                                        nobs=[df.loc[df["NEW_DECK_BOOL"] == 1, "survived"].shape[0],
                                              df.loc[df["NEW_DECK_BOOL"] == 0, "survived"].shape[0]])
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))

df.loc[((df['sibsp'] + df['parch']) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df['sibsp'] + df['parch']) == 0), "NEW_IS_ALONE"] = "YES"

df.groupby("NEW_IS_ALONE").agg({"survived" : "mean"})

test_stat , pvalue = proportions_ztest(count=[df.loc[df["NEW_IS_ALONE"] == "YES", "survived"].sum(),
                                              df.loc[df["NEW_IS_ALONE"] == "NO", "survived"].sum()],

                                        nobs=[df.loc[df["NEW_IS_ALONE"] == "YES", "survived"].shape[0],
                                              df.loc[df["NEW_IS_ALONE"] == "NO", "survived"].shape[0]])
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))

## Textler üzerinden özellik üretmek

df = pd.read_csv("datasets/Titanic-Dataset.csv")
df.head()

## Letter Count

df["NEW_NAME_COUNT"] = df["Name"].str.len()

## Word Count

df["NEW_NAME_WORD_COUNT"] = df["Name"].apply(lambda x: len(str(x).split(" ")))

## Özel yapıları yakalamak

df["NEW_NAME_DR"] = df["Name"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
df.groupby("NEW_NAME_DR").agg({"Survived" : "mean"})

## Regex ile değişken üretmek

df["NEW_TITLE"] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

df[["NEW_TITLE" , "Survived", "Age"]].groupby(["NEW_TITLE"]).agg({"Survived" : "mean", "Age" : ["count", "mean"]})

## Date değişkenleri üretmek

df_date = pd.read_csv("datasets/course_reviews.csv")
df_date['Timestamp'] = pd.to_datetime(df_date['Timestamp'], format="%Y-%m-%d %H:%M:%S")

#year
df_date['year']  = df_date['Timestamp'].dt.year
#month
df_date['month']  = df_date['Timestamp'].dt.month
# month diff
df_date['month_diff'] = (date.today().year - df_date['Timestamp'].dt.year) * 12 + date.today().month - df_date['Timestamp'].dt.month
#day_name
df_date['day_name'] = df_date['Timestamp'].dt.day_name()

####### Features Interactions (Özellik Etkileşimleri)
#########################
df = pd.read_csv("datasets/Titanic-Dataset.csv")

df["NEW_AGE_PCLASS"] = df["Age"] * df["Pclass"]

df["NEW_FAMILY_SIZE"] = df["SibSp"] + df["Parch"] + 1

df.loc[(df["Sex"] == "male") & (df["Age"] <= 21), 'NEW_SEX_CAT'] = "youngmale"

df.loc[(df["Sex"] == "male") & (df["Age"] > 21) & (df["Age"] <= 50), 'NEW_SEX_CAT'] = "maturemale"

df.loc[(df["Sex"] == "male") & (df["Age"] > 50), 'NEW_SEX_CAT'] = "seniormale"

df.loc[(df["Sex"] == "female") & (df["Age"] <= 21), 'NEW_SEX_CAT'] = "youngfemale"

df.loc[(df["Sex"] == "female") & (df["Age"] > 21) & (df["Age"] <= 50), 'NEW_SEX_CAT'] = "maturefemale"

df.loc[(df["Sex"] == "female") & (df["Age"] > 50), 'NEW_SEX_CAT'] = "seniorfemale"

df.groupby("NEW_SEX_CAT")["Survived"].mean()

