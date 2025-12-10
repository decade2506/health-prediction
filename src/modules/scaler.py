#Template scaler for 1 dimensional only
#Might need to improve for multi-dimensional data
def scaler(x): #Chuẩn hóa từ 0 đến 1
    x_min = min(x)
    x_max = max(x)
    scaled_x = []
    for i in x:
        new_x = (i - x_min) / (x_max - x_min)
        scaled_x.append(new_x)
    return scaled_x
x = [6, 3, 9, 8, 7]
print(scaler(x))