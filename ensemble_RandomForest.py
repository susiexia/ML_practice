# %%
import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# %%
df = pd.read_csv('./Resouces/loans_data_encoded.csv')

# %%
# define the features set
df_2 = df.copy()
X = df_2.drop('bad', axis = 1)
X.head()

# define the target set, convert into 1D array
y = df['bad'].ravel()
# y= df['bad'].values()
y[:5]

# %%
# split into the training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 78)

# %%
# preprocessing X features
scaler = StandardScaler()

X_scaler = scaler.fit(X_train)
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)
# %%
# fit and prediction of RF model
RFmodel = RandomForestClassifier(n_estimators=128, random_state= 78) 
RFmodel = RFmodel.fit(X_train_scaled, y_train)

y_pred = RFmodel.predict(X_test_scaled)


RFmodel_result = pd.DataFrame({'RF Predicted Y': y_pred,
                                    'RF Actual Y': y_test}).reset_index(drop=True)
RFmodel_result

# %%
# evaluating the model
# Calculating the confusion matrix.
cm = confusion_matrix(y_test, y_pred)

# Create a DataFrame from the confusion matrix.
cm_df = pd.DataFrame(
    cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])

cm_df

# %%
model_score = RFmodel.score(X_test_scaled, y_test)
model_score

# %%
# Calculating the accuracy score.
acc_score = accuracy_score(y_test, y_pred)

# %%
# Displaying results
print("Confusion Matrix")
display(cm_df)
print(f"Accuracy Score : {acc_score}, Model test Accuracy score : {model_score}")
print("Classification Report")
print(classification_report(y_test, y_pred))

# %%
