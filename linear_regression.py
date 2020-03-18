# %%
import pandas as pd 
#from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
#import numpy
# %%
df = pd.read_csv('./Resouces/Salary_Data.csv')
df.head()

# %%
plt.scatter(df['YearsExperience'], df['Salary'])
plt.xlabel('Years of Experience')
plt.ylabel('(Target) Salary in USD')
plt.show()

# %%
# Reshape independent variable to meeet Scikit-learn reqirement
# reshape to a single feature (only one column, unknown num_samples)

X = df.YearsExperience.values.reshape(-1,1)
X.shape

# %%
# assign the target variable
y = df.Salary
# %%
# make an instance a LinearRegression Model (it's a class)
model = LinearRegression()

# training/ fitting model
regressor = model.fit(X, y)

regressor  # it's still an object (instance) of LinearRegression

# %%
# generate prediction
y_pred = regressor.predict(X)
# same as
# y_pred = model.predict(X)
y_pred.shape

# %%
# plot scatter and line 
plt.scatter(X,y)   # actual points
plt.plot(X, y_pred, 'r')
plt.show()

# %%
# print specific parameters
print(model.coef_)
print(regressor.intercept_)

# %%
