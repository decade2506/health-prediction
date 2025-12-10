import pandas as pd
import modules.logistic
import modules.scaler
import modules.testSplit
# only use these imports for saving files and testing
from joblib import dump
from sklearn.metrics import accuracy_score, classification_report

data_frame = pd.read_csv("./heart_failure_clinical_records_dataset.csv")
X = data_frame.drop("DEATH_EVENT", axis=1)
Y = data_frame["DEATH_EVENT"]