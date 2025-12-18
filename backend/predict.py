import tkinter as tk
from tkinter import messagebox
import pandas as pd
import joblib

# =============================
# Load model & scaler
# =============================
model = joblib.load("logistic_model.pkl")
scaler = joblib.load("scaler.pkl")

# =============================
# Prediction function
# =============================
def predict():
    try:
        data = [
            float(age_var.get()),
            int(anaemia_var.get()),
            float(cpk_var.get()),
            int(diabetes_var.get()),
            float(ef_var.get()),
            int(hbp_var.get()),
            float(platelets_var.get()),
            float(serum_creatinine_var.get()),
            float(serum_sodium_var.get()),
            int(sex_var.get()),
            int(smoking_var.get()),
            float(time_var.get())
        ]

        columns = [
            "age", "anaemia", "creatinine_phosphokinase", "diabetes",
            "ejection_fraction", "high_blood_pressure", "platelets",
            "serum_creatinine", "serum_sodium", "sex", "smoking", "time"
        ]

        new_patient = pd.DataFrame([data], columns=columns)
        new_patient_scaled = scaler.transform(new_patient)

        prediction = model.predict(new_patient_scaled)[0]
        probability = model.predict_proba(new_patient_scaled)[0][1]

        result_text = "TỬ VONG" if prediction == 1 else "SỐNG"
        result_label.config(
            text=f"Kết quả: {result_text}\nXác suất tử vong: {probability*100:.2f}%",
            fg="red" if prediction == 1 else "green"
        )

    except Exception as e:
        messagebox.showerror("Lỗi dữ liệu", str(e))


# =============================
# UI Setup
# =============================
root = tk.Tk()
root.title("Heart Failure Death Prediction")
root.geometry("520x650")

tk.Label(root, text="DỰ ĐOÁN NGUY CƠ TỬ VONG (HEART FAILURE)",
         font=("Arial", 14, "bold")).pack(pady=10)

frame = tk.Frame(root)
frame.pack(pady=10)

# =============================
# Variables
# =============================
age_var = tk.StringVar()
anaemia_var = tk.StringVar()
cpk_var = tk.StringVar()
diabetes_var = tk.StringVar()
ef_var = tk.StringVar()
hbp_var = tk.StringVar()
platelets_var = tk.StringVar()
serum_creatinine_var = tk.StringVar()
serum_sodium_var = tk.StringVar()
sex_var = tk.StringVar()
smoking_var = tk.StringVar()
time_var = tk.StringVar()

# =============================
# Helper function to create rows
# =============================
def create_row(label, var, row):
    tk.Label(frame, text=label, anchor="w", width=30).grid(row=row, column=0, pady=5)
    tk.Entry(frame, textvariable=var, width=20).grid(row=row, column=1)

# =============================
# Input fields
# =============================
create_row("Tuổi", age_var, 0)
create_row("Thiếu máu (0/1)", anaemia_var, 1)
create_row("CPK", cpk_var, 2)
create_row("Tiểu đường (0/1)", diabetes_var, 3)
create_row("Ejection Fraction (%)", ef_var, 4)
create_row("Cao huyết áp (0/1)", hbp_var, 5)
create_row("Platelets", platelets_var, 6)
create_row("Serum Creatinine", serum_creatinine_var, 7)
create_row("Serum Sodium", serum_sodium_var, 8)
create_row("Giới tính (0=Nữ,1=Nam)", sex_var, 9)
create_row("Hút thuốc (0/1)", smoking_var, 10)
create_row("Thời gian theo dõi", time_var, 11)

# =============================
# Predict button
# =============================
tk.Button(root, text="DỰ ĐOÁN", command=predict,
          font=("Arial", 12, "bold"),
          bg="#007BFF", fg="white",
          width=20).pack(pady=20)

# =============================
# Result label
# =============================
result_label = tk.Label(root, text="", font=("Arial", 12, "bold"))
result_label.pack(pady=10)

root.mainloop()
