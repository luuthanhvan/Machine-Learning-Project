import csv
import random
'''
Các bước và các hàm cần định nghĩa khi xây dựng Cây quyết định:

1. Đọc dữ liệu (tiền xử lý nếu cần):
- Input: none
- Output: trả về tập dữ liệu
def read_file():
	# đọc file csv từ thư viện pandas <- dataset
    return dataset'''

def readFile(fileName):
    with open(fileName) as file:
        data = csv.reader(file) # Su dung phuong thuc reader trong thu vien csv de themmdu lieu vao bien data
        dataset = list(data) # Dua data vao mot list
        dataset.remove(dataset[0]) # Loai bo header
        return dataset


'''2. Phân chia tập DL theo nghi thức hold-out (tập dữ liệu được chia làm 3 phần, trong đó 2 phần train, 1 phần test)
- Input:
    + dataset: tập dữ liệu đọc từ file
    + test_size: kích thước tập dữ liệu kiểm tra
- Output:
	+ train_data: tập dữ liệu huấn luyện
	+ test_data: tập dữ liệu kiểm tra
import random
def train_test_split(dataset, test_size):
	# cần tìm hiểu hàm random
    return train_data, test_data'''

def train_test_split(dataset, test_size):
    train_data, test_data = list(), list()
    size = round(test_size*len(dataset)) #Vi tri de chia tap du lieu
    random.shuffle(dataset) #Xao tron dataset
    for i in range(len(dataset)): # Them du lieu vao 2 bien train_data va test_data
        if(i<=size):
            train_data.insert(i, dataset[i])
        else:
            test_data.insert(i,dataset[i])
    return train_data, test_data

'''#Kiem tra
dataset = readFile("../data_set/iris_data.csv")
print(dataset)
train, test = train_test_split(dataset, 2/3)
print(train)
print(test)'''


'''3. Xây dựng cây
def decision_tree_algorithm(data_train, counter, min_samples_leaf, max_depth):
    return tree

sub_tree = {"question": ["yes_answer", 
                         "no_answer"]}
                         
example_tree = {'petal_width <= 0.8': ['Iris-setosa',
                        {'petal_width <= 1.65': [{'petal_length <= 4.95': ['Iris-versicolor',
                                                                           'Iris-virginica']},
                                                 'Iris-virginica']}]}

4. Dự đoán nhãn cho tập dữ liệu kiểm tra
def predict(tree, data_test):
    return y_pred

5. Tính toán độ chính xác tổng thể
def cal_accuracy_all(y_pred, test_data):
    return accuracy_score

'''