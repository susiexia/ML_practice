# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from collections import Counter


# %%
# Generate imbalanced dataset
X, y = make_blobs(n_samples=[600, 60], random_state=1, cluster_std=5)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()


# %%
# Normal train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
Counter(y_train)

# %% [markdown]
# ## Random Oversampling

# %%
# implement random oversampling
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=1)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

Counter(y_resampled)


# %%
# Logistic regression using random oversampled data
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver='lbfgs', random_state=1)
model.fit(X_resampled, y_resampled)


# %%
# Display the confusion matrix
from sklearn.metrics import confusion_matrix

y_pred = model.predict(X_test)
confusion_matrix(y_test, y_pred)


# %%
from sklearn.metrics import balanced_accuracy_score

balanced_accuracy_score(y_test, y_pred)


# %%
# Print the imbalanced classification report
from imblearn.metrics import classification_report_imbalanced
print(classification_report_imbalanced(y_test, y_pred))

# %% [markdown]
# ## SMOTE

# %%
from imblearn.over_sampling import SMOTE
X_resampled, y_resampled = SMOTE(random_state=1, ratio=1.0).fit_resample(
    X_train, y_train
)
Counter(y_resampled)


# %%
model = LogisticRegression(solver='lbfgs', random_state=1)
model.fit(X_resampled, y_resampled)


# %%
# Calculated the balanced accuracy score
y_pred = model.predict(X_test)
balanced_accuracy_score(y_test, y_pred)


# %%
# Display the confusion matrix
confusion_matrix(y_test, y_pred)


# %%
# Print the imbalanced classification report
print(classification_report_imbalanced(y_test, y_pred))


# %%



