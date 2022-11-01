import pandas as pd
import matplotlib as plt
from matplotlib import pyplot
import numpy as np

filename = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud" \
           "/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/auto.csv"

headers = ["symboling", "normalized-losses", "make", "fuel-type", "aspiration", "num-of-doors", "body-style",
           "drive-wheels", "engine-location", "wheel-base", "length", "width", "height", "curb-weight", "engine-type",
           "num-of-cylinders", "engine-size", "fuel-system", "bore", "stroke", "compression-ratio", "horsepower",
           "peak-rpm", "city-mpg", "highway-mpg", "price"]

# Use Pandas method read_csv() to load the data from the web address. Set the parameter
# "names" equal to the Python list "headers".
df = pd.read_csv(filename, names=headers)

# Using the method head() to display the first five rows of the dataframe.
# To see what the data set looks like, we'll use the head() method.
df.head()

# replace "?" to NaN
df.replace("?", np.nan, inplace=True)
# print(df.head(5))

# Evaluating for Missing Data
# The missing values are converted by default. Using the following functions to identify these missing values.
# There are two methods to detect missing data:
# isnull()
# notnull()
# The output is a boolean value indicating whether the value that is passed into the argument is in fact missing data.
missing_data = df.isnull()
missing_data.head(5)
# print("Missing data\n", missing_data)

# Count missing values in each column
# Using a for loop in Python, helps quickly figure out the number of missing values in each column.
# As mentioned above, "True" represents a missing value and "False" means the value is present in the dataset.
# In the body of the for loop the method ".value_counts()" counts the number of "True" values.
for column in missing_data.columns.values.tolist():
    print(column)
    print(missing_data[column].value_counts())
    print("")

# Calculate the mean value for the "normalized-losses" column
average_normalized_losses = df["normalized-losses"].astype("float").mean(axis=0)
print("Average of normalized-losses:", average_normalized_losses)

# Replacing "NaN" with mean value in "normalized-losses" column
df["normalized-losses"].replace(np.nan, average_normalized_losses, inplace=True)

# Calculate the mean value of the "bore" column
average_bore = df["bore"].astype('float').mean(axis=0)
print("Average of bore:", average_bore)

# Replacing "NaN" with the mean value in "bore" column
df["bore"].replace(np.nan, average_bore, inplace=True)

# Calculate the mean value in "stroke" column
average_stroke = df["stroke"].astype("float").mean(axis=0)
print("Average of stroke:", average_stroke)

# Replacing "NaN with the mena value in "stroke" column
df["stroke"].replace(np.nan, average_stroke, inplace=True)

# Calculate the mean value in "horsepower" column
average_horsepower = df["horsepower"].astype("float").mean(axis=0)
print("Average of horsepower:", average_horsepower)

# Replacing "NaN" with mean value in "horsepower" column
df["horsepower"].replace(np.nan, average_horsepower, inplace=True)

# Calculate the mean value in "peak-rpm" column
average_peak_rpm = df["peak-rpm"].astype("float").mean(axis=0)
print("Average of peak-rpm", average_peak_rpm)

# Replacing "NaN" with mean value in "peak-rpm" column
df['peak-rpm'].replace(np.nan, average_peak_rpm, inplace=True)

# To see which values are presented in particular column, using the "value_counts()" method
df["num-of-doors"].value_counts()

# To calculate the most common type automatically, using ".idxmax()" method
df["num-of-doors"].value_counts().idxmax()

# Replace the missing 'num-of-doors' values by the most frequent
df["num-of-doors"].replace(np.nan, "four", inplace=True)

# Dropping all the rows that do not have "price" data
df.dropna(subset=["price"], axis=0, inplace=True)

# After dropping of rows, index has to be reset
df.reset_index(drop=True, inplace=True)

# Listing data types for each column
# print(df.dtypes)

# Converting data types to proper format
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["bore", "stroke", "price", "peak-rpm"]] = df[["bore", "stroke", "price", "peak-rpm"]].astype("float")

# Checking if format change is correct
print(df.dtypes)

# Data Standardization
print(df.head())

# Applying data transformation to transform mpg into L/100km
df['city-L/100km'] = 235/df['city-mpg']

# Applying data transformation to transform highway-mpg into highway-L/100km
df["highway-mpg"] = 235/df["highway-mpg"]
df.rename(columns={"highway-mpg": "highway-L/100km"}, inplace=True)
print(df.head())

# Data Normalization
# Replacing (original value) by (original value) / (maximum value)
df["length"] = df["length"] / df["length"].max()
df["width"] = df["width"] / df["width"].max()
df["height"] = df["height"] / df["height"]. max()

# Data binning
# Converting data to correct format
df["horsepower"] = df["horsepower"].astype(int, copy=True)

# Plotting the histogram of horsepower to see what distribution of horsepower looks like
plt.pyplot.hist(df["horsepower"])

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")

# Building a bin array with a minimum value to a maximum value.
# The value will determine when one bin ends and another begins.
bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)

# Setting group names
group_names = ["Low", "Medium", "High"]

# Applying the function "cut" to determine what each value of `df['horsepower']` belongs to.
df["horsepower-binned"] = pd.cut(df["horsepower"], bins, labels=group_names, include_lowest=True)
print(df[["horsepower", "horsepower-binned"]].head(20))

# Using ".value_counts()" method to see the number of vehicles in each bin
vehicles_bin_count = df["horsepower-binned"].value_counts()
print(vehicles_bin_count)

# Plotting the distribution of each bin
pyplot.bar(group_names, vehicles_bin_count)

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")




