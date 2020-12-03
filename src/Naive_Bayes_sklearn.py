import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 

def read_file():
    dataset = pd.read_csv("../data_set/heart_failure_clinical_records_dataset.csv", delimiter=",")
    return dataset

def split_data(dataset):
    y = dataset.iloc[:, -1] # in this dataset, the last column is classes
    X = dataset.iloc[:, :-1] # all columns except the last column is atrribute column
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3.0, random_state=100)
    return X, y, X_train, X_test, y_train, y_test
    
def train_using_gaussian(X_train, y_train):
    model = GaussianNB()
    # performing training
    model.fit(X_train, y_train)
    return model

# Prediction
def prediction(X_test, model):
    y_pred = model.predict(X_test)
    return y_pred

# Calculating accuracy
def cal_accuracy(y_test, y_pred):
    # Calculating accuracy using confusion matrix
    print(confusion_matrix(y_test, y_pred, labels=[1.0, 0.0]))
    print("Accuracy is ", accuracy_score(y_test, y_pred)*100)

'''
While implementing the decision tree we will go through the following two phases:
    1. Building Phase
        Preprocess the dataset.
        Split the dataset from train and test using Python sklearn package.
        Train the classifier.
    2. Operational Phase
        Make predictions.
        Calculate the accuracy.
'''

def main():
    ''' Hold-out'''
    '''
    # 1. Building phrase
    data = read_file()
    # print(data)
    X, y, X_train, X_test, y_train, y_test = split_data(data) # Split the dataset from train and test using Python sklearn package.
    model = train_using_gaussian(X_train, y_train) # Train the classifier using Gaussian

    # 2. Opeational phrase
    # Prediction
    y_pred = prediction(X_test, model)
    # print(y_test)
    # print(y_pred)
    # Calculate the accuracy
    cal_accuracy(y_test, y_pred)
    '''
    
    ''' k-fold '''
    data = read_file()
    kf = KFold(n_splits=40, shuffle=True) # chia tap du lieu thanh 15 phan
    y = data.iloc[:, -1]
    X = data.iloc[:, :-1]

    scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index, ], X.iloc[test_index, ]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model = GaussianNB()
        model.fit(X_train, y_train) # train mô hình sử dụng phân phối xác suất Gauss
        scores.append(model.score(X_test, y_test)) # tính độ chính xác tổng thể ở mỗi vòng lặp và lưu vào mảng score
        # tính độ chính xác cho từng phân lớp của mỗi lần lặp
        y_pred = model.predict(X_test)
        cfm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))

    average_score = np.mean(scores) # tính trung bình độ chính xác tổng thể của 15 lần lặp
    print("Accuracy is ", average_score*100)
    # độ chính xác cho từng phần lớp ở lần lặp cuối cùng
    print(cfm)
    
# calling main function
if __name__=="__main__":
    main()