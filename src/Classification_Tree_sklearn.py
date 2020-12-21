import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 

def read_file():
    dataset = pd.read_csv("../dataset/heart_failure_clinical_records_dataset.csv", delimiter=",")
    return dataset

def split_data(dataset):
    y = dataset.iloc[:, -1] # in this dataset, the last column is classes
    X = dataset.iloc[:, :-1] # all columns except the last column is atrribute column
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3.0, random_state=100)
    return X, y, X_train, X_test, y_train, y_test
    
# Decision Tree model
def train_using_gini(X_train, y_train):
    # creating the classifier object
    clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=11, min_samples_leaf=1)
    # performing training
    clf_gini.fit(X_train, y_train)
    return clf_gini

def train_using_entropy(X_train, y_train):
    # decision tree with entropy
    clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=11, min_samples_leaf=1)
    # performing training
    clf_entropy.fit(X_train, y_train)
    return clf_entropy

# Prediction
def prediction(X_test, clf_object):
    y_pred = clf_object.predict(X_test)
    #y_pred = clf_gini.predict([[8, 3, 7, 2]])
    #print(y_pred)
    return y_pred

# Calculating accuracy
def cal_accuracy(y_test, y_pred):
    # Calculating accuracy using confusion matrix
    #print(confusion_matrix(y_test, y_pred, labels=[2,0,1])) # for iris data set in sklearn
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
    # 1. Building phrase
    data = read_file()
    X, y, X_train, X_test, y_train, y_test = split_data(data) # Split the dataset from train and test using Python sklearn package.
    clf_gini = train_using_gini(X_train, y_train) # Train the classifier using Gini index
    clf_entropy = train_using_entropy(X_train, y_train) # Train the classifier using Entropy

    # 2. Opeational phrase
    print("Results using Gini index: ")
    # Prediction using Gini index
    y_pred_gini = prediction(X_test, clf_gini)
    # Calculate the accuracy
    cal_accuracy(y_test, y_pred_gini)

    print("Result using Entropy: ")
    # prediction using Entropy
    y_pred_entropy = prediction(X_test, clf_entropy)
    # Calculate the accuracy
    cal_accuracy(y_test, y_pred_entropy)

# calling main function
if __name__=="__main__":
    main()