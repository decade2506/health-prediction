# Import Thư viện và module cần thiết
import pandas as pd
import modules.logistic as l
import modules.scaler as s
import modules.testSplit as ts
# only use these imports for saving files and testing
from joblib import dump
from sklearn.metrics import accuracy_score, classification_report

# Đọc dữ liệu từ csv
data_frame = pd.read_csv("./heart_failure_clinical_records_dataset.csv")
X = data_frame.drop("DEATH_EVENT", axis=1)
Y = data_frame["DEATH_EVENT"]

# Chia dữ liệu theo 30/70 với 70 để ktra và 30 đối chiếu
X_train, Y_train, X_test, Y_test = ts.train_test(X.values.tolist(), Y.values.tolist(), test_size=0.3)

# Chuẩn hóa dữ liệu
scaler = s.MyScaler()
X_train_scaled = scaler.fit_transform(pd.DataFrame(X_train))
X_test_scaled = scaler.transform(pd.DataFrame(X_test))

#Khai báo và train model
model = l.MyLogisticRegression(lr=0.1, steps=1000)
model.logisticRegression(X_train_scaled, Y_train)

#Từ model train dự đoán DEATH_EVENT
Y_pred = model.predict(X_test_scaled)

# Đối chiếu
accuracy = accuracy_score(Y_test, Y_pred)
print(f'Độ chính xác: {accuracy}')
report = classification_report(Y_test, Y_pred)
print(f'Báo cáo phân loại:\n {report}')

# Lưu model và scaler đã train
dump(model, "./logistic_model_2.pkl")
dump(scaler, "./scaler_2.pkl")

print("Lưu thành công :D")
