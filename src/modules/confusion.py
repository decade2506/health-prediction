from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Dự đoán trên tập test
y_pred = model.predict(X_test_scaled)

# Tạo ma trận nhầm lẫn
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)
