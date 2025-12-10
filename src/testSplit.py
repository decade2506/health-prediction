import random
def train_test(X, Y, test_size = 0.3): #split data để train và test
     data = list(zip(X, Y))
     random.shuffle(data)
     
     split_index = int(len(data) * (1 - test_size))
     train_data = data[:split_index]
     test_data = data[split_index:]

     X_train, Y_train = zip(*train_data)
     X_test, Y_test = zip(*test_data)

     return list(X_train), list(Y_train), list(X_test), list(Y_test)