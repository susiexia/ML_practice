# %%
''' LogiticRegression practice'''

# %%
import pandas as pd 
from sklearn.linear_model import LogisticRegression

# %%
df = pd.read_csv('./Resouces/diabetes.csv')
df.head()

# %%
# data preparation
X= df.drop('Outcome', axis =1)
y= df.Outcome
X.shape
# %%
# train and test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1, stratify = y)
X_train.shape
# %%
# make an instance of model
classifier_model = LogisticRegression(random_state=1, solver='lbfgs', max_iter=200) 

# fit train dataset
classifier_model.fit(X_train, y_train)

# get prediction drom the test dataset
y_pred = classifier_model.predict(X_test)

# model score
model_score = classifier_model.score(X_test, y_test) 
model_score
# %%
# accuracy_score   same as model_score
from sklearn.metrics import accuracy_score
acc_score = accuracy_score(y_test, y_pred)
acc_score 
# The model was correct 77.6% of the time 
# %%
# create a new DF for only y_test and y_pred
results = pd.DataFrame({"Prediction":y_pred, "Actual_y": y_test}).reset_index(drop=True)
results

# %%
# use confusion metrix to validate model
from sklearn.metrics import confusion_matrix, classification_report
matrix = confusion_matrix(y_test,y_pred)
report = classification_report(y_test, y_pred)
cm_df = pd.DataFrame(matrix, index=['Actual +', 'Actual -'], 
                    columns =['Predicted +','Predicted -'])
cm_df
# %%
print(report)

# %%
