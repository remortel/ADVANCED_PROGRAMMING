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


#fetch_data()


# Load the data
oecd_bli = pd.read_csv(DATA_PATH+ "oecd_bli_2015.csv", thousands=',')
gdp_per_capita = pd.read_csv(DATA_PATH + "gdp_per_capita.csv",thousands=',',delimiter='\t',
                             encoding='latin1', na_values="n/a")
# show the data heads
# uncomment this if you wat to  show all the columns
#pd.set_option('display.max_columns', None)
#print(oecd_bli.head())
#print(gdp_per_capita.head())
# now get some info on the data
#print(oecd_bli.info())
#print(gdp_per_capita.info())

# some columns contain useful information
# let's print a random sample of 20
print(oecd_bli.sample(n=20).loc[:,["Country","INEQUALITY","Indicator", "Value"]])
# prepare the data
def prepare_country_stats(oecd_bli, gdp_per_capita):
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=="TOT"]
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
    # let's print a random sample of 20 rows of the transformed data frame
    #print(oecd_bli.sample(n=20))
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)
    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
                                  left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    # show the head of the new dataframe
    #pd.set_option('display.max_columns', None)
    #print(full_country_stats.iloc[:10, : ])
    # check for NaN values
    #print(full_country_stats[full_country_stats.isna().any(axis=1)])
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    #remove_indices = []
    keep_indices = list(set(range(36)) - set(remove_indices))
    return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]

country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
print(country_stats.iloc[:10])
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
