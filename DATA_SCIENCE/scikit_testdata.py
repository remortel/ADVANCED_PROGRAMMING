## Example of downloading data from a public url
## and reading it into a Panda DataFrame

import os
import urllib.request
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/refs/heads/master/"
DATA_URL = DOWNLOAD_ROOT + "datasets/lifesat/"
DATA_PATH = os.path.join("datasets", "lifesat", "")
def fetch_data(data_url=DATA_URL, data_path=DATA_PATH):
    os.makedirs(data_path, exist_ok=True)
    for filename in ("oecd_bli_2015.csv", "gdp_per_capita.csv"):
        print("Downloading", filename)
        urllib.request.urlretrieve(data_url + filename, data_path + filename)

# Uncomment this line if you want to fetch the data
# no need to do this twice
#fetch_data()


# Load the data
oecd_bli = pd.read_csv(DATA_PATH+ "oecd_bli_2015.csv", encoding='utf-8-sig', thousands=',')
gdp_per_capita = pd.read_csv(DATA_PATH + "gdp_per_capita.csv",thousands=',',delimiter='\t',
                             encoding='latin1', na_values="n/a")

print("=====================================================================\n"
      "= TAKE NOTE, THE BETTER LIFE INDEX DATA IS IN LONG FORMAT\n"
      "= THIS MEANS THAT EACH INDICATOR AND ITS VALUE IS LISTED IN BLOCKS\n"
      "= AND THAT EACH COUNTRY WILL APPEAR MAMNY TIMES IN THIS LIST\n"
      "= LET US INVESTIGATE TOGETHER WHAT KIND OF DATA WE ARE DEALING WITH\n"
      "=====================================================================")
print(" LET US START WITH SOME INFO ON THE BLI FILE: ")
oecd_bli.info()
print(" NUMBER OF COUNTRY ENTRIES IN THE DATA:   \n", oecd_bli["Country"].value_counts() )
print(" INDICATORS ENCOUTERED IN THE DATA: \n", oecd_bli["Indicator"].value_counts())
print(" LET US START WITH SOME INFO ON THE GDP FILE: ")
gdp_per_capita.info()
print(" NUMBER OF COUNTRY ENTRIES IN THE DATA:   \n", gdp_per_capita["Country"].value_counts())
pd.set_option('display.max_columns', None)
print(" FIRST 10 ENTRIES IN THE DATA:   \n", gdp_per_capita.head(10) )
print(" It seems that column 2015 has the relevant gdp per capita in USD")

# prepare the data
def prepare_country_stats(oecd_bli, gdp_per_capita):
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=="TOT"]
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
    print(oecd_bli.columns.tolist())  # Now you'll see all 24 indicators
    print(oecd_bli.head(10))
    # Change name of column "2015" of the GDP data frame to "GDP per capita"
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    # set the index of the GDP file to "Country" instead of a number
    gdp_per_capita.set_index("Country", inplace=True)
    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
                                  left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    # show the head of the new dataframe
    #pd.set_option('display.max_columns', None)
    print(full_country_stats.shape)
    print(full_country_stats.iloc[:10, : ])
    # check for NaN values
    print(" CHECKING for NaN: ")
    print(full_country_stats[full_country_stats.isna().any(axis=1)])
    print(" REMOVING SOME OUTLIERS: ")
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    # remove_indices = []
    keep_indices = list(set(range(36)) - set(remove_indices))
    return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]

country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
#print(country_stats.iloc[:10])
X = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]]

# Visualize the data
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
country_stats.plot(kind='scatter', x="GDP per capita", y='Life satisfaction')


# Select a linear model
model = linear_model.LinearRegression()

# Train the model
model.fit(X, y)
plt.plot(X, model.predict(X), color="blue", linewidth=3)
plt.show()
# Make a prediction for Cyprus
X_new = [[22587]]  # Cyprus' GDP per capita
print(model.predict(X_new)) # outputs [[ 5.96242338]]
