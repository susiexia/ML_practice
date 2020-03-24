# %%
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# %%
df = pd.read_csv('./Resouces/loans_data_encoded.csv')

y = df['bad'].ravel()
X = df.drop('bad', axis = 1)

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1)

scaler = StandardScaler()

X_scaler = scaler.fit(X_train)
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

# %%
# use for loop to choose best learning rate
learning_rates = [0.05, 0.1, 0.25, 0.5, 0.75, 1]
for lr in learning_rates:
    GB_model = GradientBoostingClassifier(learning_rate=lr,
                                            n_estimators=20, max_depth=3, max_features=5, random_state=0)
    GB_classifier = GB_model.fit(X_train_scaled, y_train)

    print("Learning rate: ", lr)
    print("Accuracy score (training): {0:.3f}".format(
       GB_classifier.score(
           X_train_scaled,
           y_train)))
    print("Accuracy score (validation): {0:.3f}".format(
       GB_classifier.score(
           X_test_scaled,
           y_test)))

# %% [markdown]

# Of the learning rates used, 0.5 yields the best accuracy score for the testing set and a high accuracy score for the training set. This is the value we’ll implement in the final model.

# %%
GB_model = GradientBoostingClassifier(learning_rate=0.5,
                                    n_estimators=20,max_features=5,
                                       max_depth=3,
                                       random_state=0)
GB_model.fit(X_train_scaled, y_train)
y_pred = GB_model.predict(X_test_scaled)

acc_score = accuracy_score(y_test, y_pred)

results = pd.DataFrame({'Actually Y': y_test, 'Predicted Y':y_pred})\
    .reset_index(drop = True)
results

# %%
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(
   cm, index=["Actual 0", "Actual 1"],
   columns=["Predicted 0", "Predicted 1"]
)
display(cm_df)

# %%
print(classification_report(y_test, y_pred))

# %%
