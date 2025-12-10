import pandas as pd
import modules.logistic
import modules.scaler
import modules.testSplit

data_frame = pd.read_csv("./heart_failure_clinical_records_dataset.csv")
X = data_frame.drop("DEATH_EVENT", axis=1)
Y = data_frame["DEATH_EVENT"]