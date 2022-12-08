import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df = pd.read_csv("Week 7 (Yapay Öğrenme)/Bitirme Projesi I/hitters.csv")
df.describe().T

# Özellik Çıkarımı:

# Oyuncunun kariyeri boyunca yaptığı isabetli vuruşların tüm vuruşlara oranı
df["HittingAbility"] = df["CHits"] / df["CAtBat"]
# Oyuncunun son sezondaki isabetli vuruşların son sezondaki toplam vuruş sayısına oranı
df["LSHittingAbility"] = df["Hits"] / df["AtBat"]
# Oyuncunun son zamandaki takıma kazandırdığı sayı metriği
df["LSscore"] = df["HmRun"] * df["Runs"]
# Oyuncunun son sezon isabetli vuruş sayısının kariyeri boyunca yaptığı isabetli atış sayısına oranı
df["LSvC"] = df["Hits"] / df["CHits"]

df.head()

# Aykırı Değer Analizi:

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if "Salary" not in col]
for col in num_cols:
    print("Değişken: {}, Aykırı Değer: {}".format(col, check_outlier(df, col)))

replace_with_thresholds(df, "LSHittingAbility")

# Veri setinde aykırı değer gözlemlenmemiştir. Yeni değişkenlerden "LSHittingAbility" değişkeninde aykırı değer görülüyor.

# Eksik Değer Analizi:

df.isnull().sum()
missing_values_table(df)

# Salary değişkeninde eksik değerler gözlemlenmiştir.

df["Salary"].fillna(df.groupby("Years")["Salary"].transform("mean"), inplace=True)
df.isnull().sum()
df.groupby("Years")["Salary"].mean()
miss = df[df["Salary"].isnull() == True].index

for i in miss:
    if df.loc[i, "Years"] < 5:
        df.loc[i, "Salary"] = df.loc[df["Years"] < 5, "Salary"].mean()
    elif (df.loc[i, "Years"] >= 5) and (df.loc[i, "Years"] < 10):
        df.loc[i, "Salary"] = df.loc[(df["Years"] >= 5) & (df["Years"] < 15), "Salary"].mean()
    else:
        df.loc[i, "Salary"] = df.loc[(df["Years"] >= 15), "Salary"].mean()
df.loc[miss]

# Salary değişkenindeki eksik değerler, sporcunun major liginde oynama süresi kırılımında atama yaparak giderilmiştir.

# Korelasyon Analizi

df.corr()
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

# One-Hot Encoder
for col in cat_cols:
    print(col, df[col].nunique())
df = one_hot_encoder(df, cat_cols)
df.head()

# Kategorik değişkenlerin tümü 2 sınıftan oluşmaktadır. One hot encoder ile encoding işlemi gerçekleştirilip ilk sınıflar drop edilmiştir.

# Nümerik Değişkenlerin Standartlaştırılması

rs = RobustScaler()
df[num_cols] = rs.fit_transform(df[num_cols])

# Model

X = df[[col for col in df.columns if "Salary" not in col]]
y = df["Salary"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

reg_model = LinearRegression().fit(X_train, y_train)

# sabit (b - bias)
reg_model.intercept_

# coefficients (w - weights)
reg_model.coef_

# Model Başarı Değerlendirme

# Train RMSE

y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))

# Train RKARE

reg_model.score(X_train, y_train)

# Test RMSE

y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

# Test RKARe

reg_model.score(X_test, y_test)

# 5 Katlı CV RMSE

np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X,
                                 y,
                                 cv=5,
                                 scoring="neg_mean_squared_error")))
