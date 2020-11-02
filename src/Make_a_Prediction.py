from sklearn.metrics import accuracy_score
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
wineWhite = pd.read_csv("./heart_failure_clinical_records_dataset .csv")


"""
def predict(node, row):
	if row[node['index']] < node['value']:
		if isinstance(node['left'], dict):
			return predict(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], row)
		else:
			return node['right']

"""

X = wineWhite.iloc[:,0:12]
y = wineWhite.iloc[:,12]

print(wineWhite)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 1/3.0,random_state = 5)

clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 5, max_depth = 8, min_samples_leaf =  2)
clf_gini.fit(X_train,y_train)
TreeDuBao =  clf_gini.predict(X_test)
print("Do chinh xac cua giai thuat cay quyet dinh hold- out la:", accuracy_score(y_test,TreeDuBao)*100)
