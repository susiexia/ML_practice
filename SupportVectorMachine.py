# %%
'''There are a number of classification algorithms that can be used to determine
 loan elgibility. Some algorithms run better than others. 
 Build a loan approver using the SVM algorithm and compare the accuracy and performance 
 of the SVM model with the Logistic Regression model.'''

# %%
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# %%
df = pd.read_csv('./Resouces/loans.csv')
df.head()    # had already been normalized

# %%
# STEP 0: data prepartion
y = df['status']
X = df.drop('status', axis=1)
X.shape          # a tuple: (100,5)

# %%
# STEP 1: train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, 
                                                    random_state = 1)

# STEP 2: Instantiate a SVC model
model = SVC(kernel='linear')

# STEP 3: train model
model.fit(X_train, y_train)

# STEP 4: make prediction
y_pred = model.predict(X_test)

results = pd.DataFrame({'Predicted y': y_pred, 'Actual y': y_test}).reset_index(drop=True)
results
# STEP 5-1 model score
model_score = model.score(X_test,y_test)
print(model_score)

# %%
# Step 5-1: evaluate, validate
acc_score = accuracy_score(y_test, y_pred)
acc_score            #same as model_score

# STEP 5-2: CM
cm_df = pd.DataFrame(confusion_matrix(y_test, y_pred), index=['Actual +', 'Actual -'],
                    columns=['Predicted +', 'Predicted -'])
cm_df

# STEP 5-3: classification report
print(classification_report(y_test,y_pred))

# %% [markdown]

# Summary: The model did not perform adaquately 
#to pose an acceptable risk for a lender due to lower accuracy score, and lower f1 score.

# %% [markdown]
# A comparable logistic Regression model

# %%
from sklearn.linear_model import LogisticRegression

# step 2
Logit_model = LogisticRegression(solver='lbfgs', random_state=2)

# step 3
Logit_model.fit(X_train, y_train)

# step 4
Logit_model_y_pred = Logit_model.predict(X_test)

Logit_model_result = pd.DataFrame({'Logit Predicted Y': Logit_model_y_pred,
                                    'Logit Actual Y': y_test}).reset_index(drop=True)
Logit_model_result
# %%
# Evaluate, validate, assess
Logit_acc = accuracy_score(y_test,Logit_model_y_pred)
print(Logit_acc)

# %%
Logit_CM_df = pd.DataFrame(confusion_matrix(y_test, Logit_model_y_pred), index=['Actual +', 'Actual -'],
                    columns=['Predicted +', 'Predicted -'])
Logit_CM_df

# %%
Logit_report = classification_report(y_test,Logit_model_y_pred)
print(Logit_report)

# %%
