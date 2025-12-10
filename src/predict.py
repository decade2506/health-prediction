import pandas as pd
import joblib #only use this for loading saved model and scaler
import numpy as np

# 1. Load model và scaler đã lưu
# Cheat file
# model = load("./logistic_model.pkl")
# scaler = load("./scaler.pkl")

# File made from scratch
model = joblib.load("./logistic_model_2.pkl")
scaler = joblib.load("./scaler_2.pkl")

print("=== Dự đoán nguy cơ tử vong do Heart Failure cho 1 bệnh nhân mới ===")

# 2. Nhập thông số bệnh nhân từ người dùng
age = float(input("Tuổi: "))
anaemia = int(input("Thiếu máu (anaemia) (0=không, 1=có): "))
creatinine_phosphokinase = float(input("CPK (creatinine_phosphokinase): ")) #~30–150 for adult females, ~50–200 for adult males
diabetes = int(input("Tiểu đường (diabetes) (0=không, 1=có): "))
ejection_fraction = float(input("Ejection fraction (%): ")) # normal is 55% or higher
high_blood_pressure = int(input("Cao huyết áp (high_blood_pressure) (0=không,1=có): "))
platelets = float(input("Số lượng tiểu cầu (platelets): ")) #normal range is 150,000 to 450,000 platelets per microliter of blood
serum_creatinine = float(input("Serum creatinine: ")) #normal range is approximately 0.6 to 1.2 milligrams per deciliter (mg/dL) for adult
serum_sodium = float(input("Serum sodium: ")) #normal range is 135-145 milliequivalents per liter (mEq/L)
sex = int(input("Giới tính (sex) (0=Nữ, 1=Nam): "))
smoking = int(input("Hút thuốc (smoking) (0=không, 1=có): "))
time = float(input("Thời gian theo dõi (time): "))

# 3. Tạo DataFrame từ dữ liệu nhập - map all inputs to a single row dataframe
new_patient = pd.DataFrame([[
    age, anaemia, creatinine_phosphokinase, diabetes,
    ejection_fraction, high_blood_pressure, platelets,
    serum_creatinine, serum_sodium, sex, smoking, time
]], columns=[
    "age","anaemia","creatinine_phosphokinase","diabetes","ejection_fraction",
    "high_blood_pressure","platelets","serum_creatinine","serum_sodium",
    "sex","smoking","time"
])

# 4. Chuẩn hóa dữ liệu
new_patient_scaled = scaler.transform(new_patient)

# 5. Dự đoán và xác suất

# Prediction if built in joblib model
# prediction = model.predict(new_patient_scaled)[0]
# probability = model.predict_proba(new_patient_scaled)[0][1]

# Prediction if custom model from scratch
raw_pred = model.predict(new_patient_scaled)
if hasattr(raw_pred, '__len__') and not isinstance(raw_pred, str):
    prediction = int(raw_pred[0])
else:
    prediction = int(raw_pred)

# Get probability for the positive class
raw_prob = model.predict_proba(new_patient_scaled)
prob_arr = np.array(raw_prob)
if prob_arr.ndim == 2 and prob_arr.shape[1] >= 2:
    probability = float(prob_arr[0, 1])
elif prob_arr.ndim == 1:
    probability = float(prob_arr.ravel()[0])
else:
    probability = float(prob_arr)

# 6. Hiển thị kết quả
print("\n=== Kết quả dự đoán ===")
print(f"Dự đoán DEATH_EVENT: {"Tử vong" if prediction == 1 else "Sống"}")
print(f"Xác suất tử vong: {probability * 100:.2f}%")
