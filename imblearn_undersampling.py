# %% [markdown]
# Implement the cluster centroids and random undersampling techniques with the credit card default data. Then estimate a logistic regression model and report the classification evaluation metrics from both sampling methods. 

# INFO: ln_balance_limit is the log of the maximum balance they can have on the card; 1 is female, 0 male for sex; the education is denoted: 1 = graduate school; 2 = university; 3 = high school; 4 = others; 1 is married and 0 single for marriage; default_next_month is whether the person defaults in the following month (1 yes, 0 no).

# %%
import pandas as pd 
from collections import Counter

from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import ClusterCentroids

from sklearn.linear_model import LogisticRegression

from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

# %%
df = pd.read_csv('./Resouces/cc_default.csv')
df.head()

# %%
# set up features and target 
x_cols = [i for i in df.columns if i not in ('ID', 'default_next_month')]
X = df[x_cols]
y = df['default_next_month']

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# %% [markdown]
# #Random UnderSampling

# %%
rus = RandomUnderSampler(random_state = 1)

X_resampled_train, y_resampled_train = rus.fit_resample(X_train, y_train)

# valid the counter
Counter(y_resampled_train)
# %%
# Fit a Logistic regression model using random undersampled data

model = LogisticRegression(solver='lbfgs', random_state=1)
model.fit(X_resampled_train, y_resampled_train)
y_pred = model.predict(X_test)

# %%
# evaluate

confusion_matrix(y_test, y_pred)
# %%
balanced_accuracy_score(y_test, y_pred)
# %%

print(classification_report_imbalanced(y_test, y_pred))
# %% [markdown]
# #ClusterCentroid Undersampling

# %%
cc = ClusterCentroids(random_state=1)
X_resampled, y_resampled = cc.fit_resample(X_train, y_train)
Counter(y_resampled)
#%%
# Logistic regression using cluster centroid undersampled data
model = LogisticRegression(solver='lbfgs', random_state=78)
model.fit(X_resampled, y_resampled)
y_pred = model.predict(X_test)
# %%
# evaluate
confusion_matrix(y_test, y_pred)
# %%
balanced_accuracy_score(y_test, y_pred)
# %%
print(classification_report_imbalanced(y_test, y_pred))