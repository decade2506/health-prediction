import pandas as pd
import modules.logistic as l
import modules.scaler as s
import modules.testSplit as ts
# only use these imports for saving files and testing
from joblib import dump
from sklearn.metrics import accuracy_score, classification_report

data_frame = pd.read_csv("./heart_failure_clinical_records_dataset.csv")
X = data_frame.drop("DEATH_EVENT", axis=1)
Y = data_frame["DEATH_EVENT"]

X_train, Y_train, X_test, Y_test = ts.train_test(X.values.tolist(), Y.values.tolist(), test_size=0.3)

model = l.MyLogisticRegression(lr=0.01, steps=1000)
model.logisticRegression(X_train, Y_train)