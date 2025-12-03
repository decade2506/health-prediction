import pandas as pd
import joblib

# 1. Load model và scaler đã lưu
model = joblib.load("./logistic_model.pkl")
scaler = joblib.load("./scaler.pkl")

print("=== Dự đoán nguy cơ tử vong do Heart Failure cho 1 bệnh nhân mới ===")

# 2. Nhập thông số bệnh nhân từ người dùng
age = float(input("Tuổi: "))
anaemia = int(input("Thiếu máu (anaemia) (0=không, 1=có): "))
creatinine_phosphokinase = float(input("CPK (creatinine_phosphokinase): "))
diabetes = int(input("Tiểu đường (diabetes) (0=không, 1=có): "))
ejection_fraction = float(input("Ejection fraction (%): "))
high_blood_pressure = int(input("Cao huyết áp (high_blood_pressure) (0=không,1=có): "))
platelets = float(input("Số lượng tiểu cầu (platelets): "))
serum_creatinine = float(input("Serum creatinine: "))
serum_sodium = float(input("Serum sodium: "))
sex = int(input("Giới tính (sex) (0=Nữ, 1=Nam): "))
smoking = int(input("Hút thuốc (smoking) (0=không, 1=có): "))
time = float(input("Thời gian theo dõi (time): "))

# 3. Tạo DataFrame từ dữ liệu nhập
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
prediction = model.predict(new_patient_scaled)[0]
probability = model.predict_proba(new_patient_scaled)[0][1]

# 6. Hiển thị kết quả
print("\n=== Kết quả dự đoán ===")
print(f"Dự đoán DEATH_EVENT: {prediction} (0=Sống, 1=Tử vong)")
print(f"Xác suất tử vong: {probability:.2f}")
