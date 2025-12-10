import pandas as pd
import logistic
import scaler
import testSplit

data_frame = pd.read_csv("./heart_failure_clinical_records_dataset.csv")
X = data_frame.drop("DEATH_EVENT", axis=1)
Y = data_frame["DEATH_EVENT"]