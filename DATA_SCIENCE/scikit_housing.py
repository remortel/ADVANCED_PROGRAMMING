import os
import tarfile
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVR


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
#If you want to improve the representativeness of the test set
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

# Let's see whether there are exact duplicates of the median_house_value
# Group by the column and show indexes for each value
groups = strat_train_set.groupby('median_house_value').apply(lambda x: x.index.tolist())
print(groups)

# Filter to only show values that appear more than once
duplicated_groups = groups[groups.apply(len) > 100]
print(duplicated_groups)
#duplicates = strat_train_set['median_house_value'].duplicated(keep=False)
#print("The duplicates for median_house_value are:\n", duplicates)
plt.show()
# some attributes per district are not that useful by themselves, but we can redefine some of them
#print(type(corr_matrix))

strat_train_set["rooms_per_household"] = strat_train_set["total_rooms"] / strat_train_set["households"]
strat_train_set["bedrooms_per_room"] = strat_train_set["total_bedrooms"] / strat_train_set["total_rooms"]
strat_train_set["population_per_household"]  = strat_train_set["population"] / strat_train_set["households"]
corr_matrix = strat_train_set.drop(columns="ocean_proximity").corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))

# The addition of combined attributes can also be handled by writing a class that
# inherits from BaseEstimator and TransformerMixin
# you then neet to implement three methods:
# 1. fit() that returns self
# 2. transform()
# 3. fit_transform which can be obtained for free from inheriting from TransformerMixin
# If you inherit also from BaseEstimator, you don't need *args and **kwargs in your constructor
# and you will get two extra methods for free: get_params() and set_params
# OK, first we set the column indices we want to transform

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4 , 5, 6
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                        bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]




# We now prepare the data for training, so we separate the input features from
# the output prediction, or label, which is the median house value
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()
print("TRAINING SET SIZE: ", len(housing))
print("TRAINING LABELS: ", len(housing_labels))
housing.info()
# Now that we have a stratified sampled training set we need to clean it
# note that all features have 16512 valid numerical values
# except for the total_bedrooms and the derived quantity bedrooms_per_room
# which have 16354 valid numerical values
# There are several options to remove or replace the non valid numerical values
# 1. drop the rows in the table that have NaN for total bedrooms
# 2. drop the feature of total bedrooms and bedrooms_per_room completely for the whole set
# 3. calculate the median of total bedrooms and replace the NaN entries with the median
# Let's use the SimpleInputer class to automatically replace NaN values with the median of the
# other values of teh same feature

inputer = SimpleImputer(strategy="median")

# since the median can only be computed for numerical attributes, we need to create a copy of
# the datasets without the textual attribute ocean_proximity

housing_num = housing.drop("ocean_proximity", axis=1)
inputer.fit(housing_num)
# We now expect 11 numerically valid values
print("TESTING THE DATA FOR NUMERICAL VALUE MEANS:", housing_num.median().values)
X = inputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)

# Now we can deal with the non_numerical attribute ocean_proximity
housing_cat = housing[["ocean_proximity"]]
print("THE possible ocean_proximity values are:", housing_cat["ocean_proximity"].unique())
# it seems that we have 5 unique values for ocean_proximity
# let's encode these with a one-hot encoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
print("The encoded values for ocean_proximity are", housing_cat_1hot.toarray())
# we can always decaode back to the original categories
print("the original categories of ocean_proximity are", cat_encoder.inverse_transform(housing_cat_1hot))

# In order to transform the whole dataset with all the necessary inputers and encoders
# we can set up a transformation pipeline
# Let's first define the numerical transformation pipeline
#
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
    ])

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
    ])
housing_prepared = full_pipeline.fit_transform(housing)

# Now let's apply a Linear Regression model on the training data
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
# Let's see how well we predict the median_house_value
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions:", lin_reg.predict(some_data_prepared))
print("Labels:", list(some_labels))
# the predictions are within range of the labels, but the difference is quite big
# let's calculate the mean squared error over the whole training set

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print("THE MEAN ERROR ON THE PREDICTIONS FROM THE LINEAR MODEL IS:", lin_rmse)

# Let's try to use a more complex model: a Decision Tree
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
print("THE MEAN ERROR ON THE PREDICTIONS FROM THE DECISION TREE MODEL IS:", tree_rmse)

# A mean RMSE of zero seems a bit unrealistic, so let's try to
# train the DecisionTreeRegressor on several smaller batches of data
# the cv=10 parameter indicates that the training data set will
# be split in 10 subsets or folds
# We will compare the average score from the DEcision Tree with the
# average score from the Linear Regressor

tree_scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                        scoring="neg_mean_squared_error", cv=10)
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                        scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-tree_scores)
lin_rmse_scores = np.sqrt(-lin_scores)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard Deviation:", scores.std())

print("THE SCORES FOR 10-FOLD CROSS VALIDATION OF THE DECISION TREE ARE:\n")
display_scores(tree_rmse_scores)
print("THE SCORES FOR 10-FOLD CROSS VALIDATION OF THE LINEAR REGRESSOR ARE:\n")
display_scores(lin_rmse_scores)

# Let's try a final model: a linear SVM
SVR_reg=LinearSVR(C=150.0)
SVR_reg.fit(housing_prepared, housing_labels)
housing_predictions = SVR_reg.predict(housing_prepared)
SVR_mse = mean_squared_error(housing_labels, housing_predictions)
SVR_rmse = np.sqrt(SVR_mse)
print("THE MEAN ERROR ON THE PREDICTIONS FROM THE SVR MODEL IS:", SVR_rmse)
SVR_scores = cross_val_score(SVR_reg, housing_prepared, housing_labels,
                        scoring="neg_mean_squared_error", cv=10)
SVR_rmse_scores = np.sqrt(-SVR_scores)
print("THE SCORES FOR 10-FOLD CROSS VALIDATION OF THE SVM Regressor ARE:\n")
display_scores(SVR_rmse_scores)