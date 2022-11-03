import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
plt.ylim(0,)
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


