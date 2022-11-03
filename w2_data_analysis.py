import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN' \
       '-SkillsNetwork/labs/Data%20files/automobileEDA.csv '

df = pd.read_csv(path)
print(df.head())

# When visualizing individual variables, it is important to first understand what type of variable you are dealing with.
# This will help us find the right visualization method for that variable.
print(df.dtypes)

# Calculate the correlation between variables of type "int64" or "float64" using the method "corr":
print(df.corr())

# Finding correlation between the following columns: bore, stroke, compression-ratio, and horsepower:
print(df[["bore", "stroke", "compression-ratio", "horsepower"]].corr())

# Positive Linear Relationship
sns.regplot(x="engine-size", y="price", data=df)
plt.ylim(0, )
print(df[["engine-size", "price"]].corr())
# The correlation between 'engine-size' and 'price' is approximately 0.87.
# Highway mpg is a potential predictor variable of price. Let's find the scatter plot of "highway-mpg" and "price"
sns.regplot(x="highway-mpg", y="price", data=df)

# As highway-mpg goes up, the price goes down:
# This indicates an inverse/negative relationship between these two variables.
# Highway mpg could potentially be a predictor of price.
print(df[['highway-mpg', 'price']].corr())
# The correlation between 'highway-mpg' and 'price' is approximately -0.704.


# Weak Linear Relationship
# Checking if "peak-rpm" is a predictor of "price":
sns.regplot(x="peak-rpm", y="price", data=df)

# Peak rpm does not seem like a good predictor of the price at all since the regression line is close to horizontal
# Therefore, it's not a reliable variable
df[["peak-rpm", "price"]].corr()
# The correlation between 'peak-rpm' and 'price' and see it's approximately -0.101616.

# Categorical Variables - These are variables that describe a 'characteristic' of a data unit, and are selected from a
# small group of categories. The categorical variables can have the type "object" or "int64". A good way to visualize
# categorical variables is by using boxplots.
sns.boxplot(x="body-style", y="price", data=df)

# Distributions of price between the different body-style categories have a significant overlap, so body-style would
# not be a good predictor of price
# Checking if "engine-location" is predictor of "price":
sns.boxplot(x="engine-location", y="price", data=df)

# Distribution of price between these two engine-location categories, front and rear, are distinct enough to take
# engine-location as a potential good predictor of price.
# Checking if "drive-wheels" is predictor of "price":
sns.boxplot(x="drive-wheels", y='price', data=df)

# Distribution of price between the different drive-wheels categories differs. As such, drive-wheels could
# potentially be a predictor of price.
drive_wheels = df['drive-wheels']

# Descriptive Statistical Analysis.The describe function automatically computes basic statistics for all continuous
# variables. Any NaN values are automatically skipped in these statistics.
print(df.describe())

# Describe method skips variables of type object. But describe method can still be applied on these variables:
print(df.describe(include=['object']))

# Value counts is a good way of understanding how many units of each variable we have.
# Converting series to df and renaming columns:
drive_wheels_counts = drive_wheels.value_counts().to_frame()
drive_wheels_counts.rename(columns={'drive-wheels': 'value_counts'}, inplace=True)
drive_wheels_counts.index.name = 'drive-wheels'
engine_loc_counts = df['engine-location'].value_counts().to_frame()
engine_loc_counts.rename(columns={'engine-location': 'value_counts'}, inplace=True)
engine_loc_counts.index.name = 'engine-location'

# Basics of Grouping
print(drive_wheels.unique())
df_group_one = df[["drive-wheels", "body-styles", "price"]]
df_group_one = df_group_one.groupby([drive_wheels], as_index=False).mean()

# From data, it seems rear-wheel drive vehicles are, on average, the most expensive, while 4-wheel and front-wheel
# are approximately the same in price.
df_group_pivot = df_group_one.pivot(index=drive_wheels, columns="body-style")

# Filling missing data with value 0
df_group_pivot = df_group_pivot.fillna(0)

# Variables: Drive Wheels and Body Style vs. Price
plt.pcolor(df_group_pivot, cmap='RdBu')
plt.colorbar()
plt.show()
fig, ax = plt.subplots()
im = ax.pcolor(df_group_pivot, cmap='RdBu')

# Label names
row_labels = df_group_pivot.columns.levels[1]
col_labels = df_group_pivot.index

# Move ticks and labels to the center
ax.set_xticks(np.arange(df_group_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(df_group_pivot.shape[0]) + 0.5, minor=False)

# Insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

# Rotate label if too long
plt.xticks(rotation=90)
fig.colorbar(im)
plt.show()

# Correlation and Causation
# Correlation: a measure of the extent of interdependence between variables.
# Causation: the relationship between cause and effect between two variables.
# Wheel-Base vs. Price
pearson_coef, p_value = stats.pearsonr(df["wheel-base"], df["price"])
print("The Pearson Correlation Coefficient is:", pearson_coef, "P-value:", p_value)

# ANOVA: Analysis of Variance
grouped_test = df_group_one[['drive-wheels', 'price']].groupby(['drive-wheels'])
grouped_test.head(2)
test = grouped_test.get_group('4wd')['price']
f_val, p_val = stats.f_oneway(grouped_test.get_group('fwd')['price'], grouped_test.get_group('rwd')['price'],
                              grouped_test.get_group('4wd')['price'])
print("ANOVA results: F=", f_val, ", P =", p_val)