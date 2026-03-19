import os
import tarfile
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from pandas import DataFrame
from pandas.io.parsers import TextFileReader
from sklearn import linear_model
import matplotlib as mpl

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/refs/heads/master/"
DATA_URL = DOWNLOAD_ROOT + "datasets/housing/"
DATA_PATH = os.path.join("datasets", "housing", "")
def fetch_data(data_url=DATA_URL, data_path=DATA_PATH):
    os.makedirs(data_path, exist_ok=True)
    filename = "housing.tgz"

    print("Downloading", filename)
    urllib.request.urlretrieve(data_url + filename, data_path + filename)
    housing_tgz = tarfile.open(data_path + filename)
    housing_tgz.extractall(data_path)
    housing_tgz.close()
# Uncomment this line if you want to fetch the data
# no need to do this twice
#fetch_data()

def load_housing_data(data_path=DATA_PATH):
    csv_path = os.path.join(data_path, "housing.csv")
    return pd.read_csv(csv_path)



housing = load_housing_data(data_path=DATA_PATH)
print(housing.head())
housing.info()
print( " THE OCEAN PROXIMITY FIELD HAS CATHEGORICAL ATTRIBUTES: \n", \
       housing["ocean_proximity"].value_counts())
pd.set_option('display.max_columns', None)
print(housing.describe())
housing.hist(bins=50)
#plt.show()
plt.close('all')
#split your data into a training set and a test set
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
print("TRAINING SET SIZE: ", len(train_set))
print("TEST SET SIZE: ", len(test_set))

# Let's check whether the distributions of the median_incomes are similar for both sets
# Plot histograms, capturing the counts

def plot_train_test_ratio(train_set, test_set):
    plt.close('all')
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    bins = 50

    # density=True normalizes so the histogram integrates to 1
    train_counts, bin_edges, _ = ax1.hist(train_set["median_income"], bins=bins, density=True)
    test_counts, _, _           = ax2.hist(test_set["median_income"],  bins=bins, density=True)

    ax1.set_title("Train - Median Income")
    ax2.set_title("Test - Median Income")

    # Now the ratio is meaningful — both histograms are on the same scale
    ratio = train_counts / (test_counts + 1e-10)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    ax3.scatter(bin_centers, ratio, s=10)
    ax3.axhline(y=1, color="red", linestyle="--", label="ratio = 1")
    ax3.set_title("Ratio Train / Test (normalized)")
    ax3.set_xlabel("Median Income")
    ax3.set_ylabel("Ratio")
    ax3.legend()
    ax3.set_ylim(0.8, 1.2)
    plt.tight_layout()
    plt.show()

#plot_train_test_ratio(train_set, test_set)
#If you want to improve the representativeness of teh test set
#it would be best that the test set has the same distribution of incomes as the training set
# we create 5 categories of incomes via the panda.cut function and add it as an attribute top the data set

housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1,2,3,4,5])
#plt.close('all')
#housing["income_cat"].hist()
#plt.show()

# Let's split the housing data set again in a stratified sampled split which will improve the
# agreement between the Median Income distributions
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    print("TRAIN:", train_index, "TEST:", test_index)
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

#plot_train_test_ratio(strat_train_set, strat_test_set)
# remove the income category as an attribute, since we don't want to train on this
housing.drop("income_cat", axis=1, inplace=True)
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

# let's look at some other interesting attributes: do the median prices depend on the location
plt.close('all')
strat_train_set.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=strat_train_set["population"]/100, label="population", figsize=(12, 8),
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
plt.legend()
plt.show()

# we want to predict housing prices, so lets look at correlations, but first drop the non-numerical attribute
corr_matrix = strat_train_set.drop(columns="ocean_proximity").corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))
# lets plot some correlations
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(strat_train_set[attributes], figsize=(12, 8))
plt.show()
# When plotting  a scatter plot of median income vs median house value.we see horizontal structures
strat_train_set.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1, figsize=(12, 8))
plt.show()
# some attributes per district are not that useful by themselves, but we can redefine some of them
#print(type(corr_matrix))

strat_train_set["rooms_per_household"] = strat_train_set["total_rooms"] / strat_train_set["households"]
strat_train_set["bedrooms_per_room"] = strat_train_set["total_bedrooms"] / strat_train_set["total_rooms"]
strat_train_set["population_per_household"]  = strat_train_set["population"] / strat_train_set["households"]
corr_matrix = strat_train_set.drop(columns="ocean_proximity").corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))
