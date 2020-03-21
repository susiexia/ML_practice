# %%
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# %%
df = pd.read_csv('./Resouces/loans_data_encoded.csv')
df.head()

# %%

y = df['bad']     # alternatively add '.values' to change it to an array

df_2 = df.copy()
X = df_2.drop('bad', axis = 1)
# %%
# seperate and split frist
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1)

# %%
# standardized X_train and X_test, the results would be ndarray
scaler = StandardScaler()

X_scaler = scaler.fit(X_train)
# transform on both X sets
X_scalered_train = X_scaler.transform(X_train)
X_scalered_test = X_scaler.transform(X_test)

# check if standardized successfully
import numpy as np
X_scalered_train[:5]
print(np.mean(X_scalered_train[:,0]))
print(np.mean(X_scalered_train[:,0]))
# %%
DT_model = tree.DecisionTreeClassifier()

DT_model.fit(X_scalered_train, y_train)

y_pred = DT_model.predict(X_scalered_test)

# %%
# visualize
tree.plot_tree(DT_model)

# the root mode is X[1], column 2: term
# %%
# evaluate
acc_score = accuracy_score(y_test,y_pred)

cm_df = pd.DataFrame(confusion_matrix(y_test, y_pred), index = ['actually good(0)', 'actually bad(1)'], 
                                       columns=["Predicted good", "Predicted bad"])

report = classification_report(y_test, y_pred)

# %%
# Displaying results
print("Confusion Matrix")
print(cm_df)
print(f"Accuracy Score : {acc_score}")
print("Classification Report")
print(report)

# %%
from sklearn.metrics import precision_recall_fscore_support
summary = precision_recall_fscore_support(y_test,y_pred)
summary
# %% [markdown]

# In summary, this model may not be the best one for preventing fraudulent loan applications because the model's accuracy, 0.5, is low, and the precision and recall are not good enough to state that the model will be good at classifying fraudulent loan applications. Modeling is an iterative process: you may need more data, more cleaning, another model parameter, or a different model. It’s also important to have a goal that’s been agreed upon, so that you know when the model is good enough.