import csv
import time
import random
import pandas as pd
import numpy as np
import array as arr
from pprint import pprint
import math
from collections import deque
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

'''
Các bước và các hàm cần định nghĩa khi xây dựng Cây quyết định:
1. Đọc dữ liệu (tiền xử lý nếu cần):
- Input: tên file hoặc đường dẫn đến file .csv
- Output: trả về tập dữ liệu
def read_file():
	# đọc file csv từ thư viện pandas <- dataset
    return dataset
'''
def read_file():
    # dataset = pd.read_csv("../dataset/iris_data.csv", delimiter=",")
    dataset = pd.read_csv("../dataset/test1.csv", delimiter=",")
    # dataset = pd.read_csv("../dataset/heart_failure_clinical_records_dataset.csv", delimiter=",")
    return dataset

'''
2. Phân chia tập DL theo nghi thức hold-out (tập dữ liệu được chia làm 3 phần, trong đó 2 phần train, 1 phần test)
- Input:
    + dataset: tập dữ liệu đọc từ file
    + test_size: kích thước tập dữ liệu kiểm tra
- Output:
	+ train_data: tập dữ liệu huấn luyện
	+ test_data: tập dữ liệu kiểm tra
import random
def train_test_split(dataset, test_size):
    return train_data, test_data
'''
def train_test_split(dataset, test_size):
    if isinstance(test_size, float):
        test_size = round(test_size * len(dataset))  # để đảm bảo test_size luôn là 1 số nguyên

    # lấy tất cả các chỉ số trong tập DL gốc, sau đó chuyển nó sang dạng list
    indices = dataset.index.tolist()
    # random các chỉ số cho tập test, lưu vào mảng test_indices
    test_indices = random.sample(population=indices, k=test_size)
    # print(test_indices)
    test_data = dataset.iloc[
        test_indices]  # lấy các giá trị và lưu vào tập DL test thông qua các chỉ số trong mảng test_indices
    train_data = dataset.drop(
        test_indices)  # lấy các giá trị bỏ đi các chỉ số trong mảng test_indices và lưu vào tập DL train
    return train_data, test_data


'''3. Xây dựng cây
def decision_tree_classifier(dt, counter, min_samples_leaf, max_depth):
    return sub_tree
sub_tree = {"question": ["yes_answer", 
                         "no_answer"]}
                         
example_tree = {'petal_width <= 0.8': ['Iris-setosa',
                        {'petal_width <= 1.65': [{'petal_length <= 4.95': ['Iris-versicolor',
                                                                           'Iris-virginica']},
                                                 'Iris-virginica']}]}
'''
# hàm kiểm tra dữ liệu trong 1 nút có thuần nhất hay không
# một nút được xem là có DL thuần nhất khi nút đó chỉ chứa duy nhất 1 nhãn
def check_purity(data):
    label_column = data[:, -1]  # lấy nguyên cột nhãn trong data
    # print(label_column)

    classes = np.unique(label_column)  # lấy giá trị các nhãn duy nhất từ cột nhãn
    # print(classes)
    # Ví dụ: đối với tập dữ liệu hoa Iris, classes = ['Iris-setosa' 'Iris-versicolor' 'Iris-virginica']

    # kiểm tra dữ liệu trong mảng classes, 
    # nếu số lượng phần tử trong mảng classes chỉ có 1 nhãn -> thuần nhất -> là nút lá
    # ngược lại không phải là nút lá 

    # print(len(classes))
    if len(classes) == 1:
        return True
    else:
        return False

# hàm tạo nút lá
def create_leaf_node(data):
    label_column = data[:, -1]

    # nếu dữ liệu trong 1 nút đã thuần nhất rồi thì gán nhãn nhãn đó cho nút lá luôn
    if check_purity(data) == True:
        leaf_node = np.unique(label_column)[0]
    else:
        # lấy giá trị các nhãn duy nhất từ cột nhãn cùng với số lượng nhãn tương ứng
        classes, counts_classes = np.unique(label_column, return_counts=True)
        # print(counts_classes)
        # tìm phần tử có số lượng nhãn lớn nhất có trong mảng counts_classes
        maxValue = counts_classes[0]
        index = 0
        for i in range(len(counts_classes)):
            if (maxValue < counts_classes[i]):
                maxValue = counts_classes[i]
                index = i  # lưu lại vị trí
        leaf_node = classes[index]  # lấy nhãn tương ứng với vị trí vừa tìm được ở bên trên

    return leaf_node

def get_point_splits(data):
    point_splits = {}
    no_rows, no_cols = data.shape

    for column_index in range(no_cols-1):
        column_values = data[:, column_index]
        # print(column_values)
        unique_column_values = np.unique(column_values) # loại bỏ các giá trị trùng

        type_of_feature = FEATURE_TYPES[column_index]

        if type_of_feature == "continuous":
            point_splits[column_index] = []
            for row_index in range(no_rows-1):
                if row_index != 0: # không xét dòng thứ 0 và dòng cuối cùng
                    previous_class_value = data[row_index-1, -1]
                    current_class_value = data[row_index, -1]

                    # nếu mà giá trị nhãn hiện tại khác với giá trị nhãn trước và sau đó thì chọn điểm hiện tại là điểm phân hoạch
                    if current_class_value != previous_class_value:
                        column_value = data[row_index, column_index]
                        point_splits[column_index].append(column_value)
        
        # feature is categorical
        else:
            point_splits[column_index] = unique_column_values
    
    # print(point_splits)
    return point_splits

# Phân hoạch nhị phân dựa trên 1 giá trị ngưỡng cho 1 thuộc tính (cột) trên tập dữ liệu
def binary_split_data(data, split_column, split_value):
    split_column_values = data[:, split_column]
    # print(split_column_value)
    type_of_feature = FEATURE_TYPES[split_column]
    if type_of_feature == "continuous":
        left = data[split_column_values <= split_value]
        right = data[split_column_values > split_value]
    else:
        left = data[split_column_values == split_value]
        right = data[split_column_values != split_value]
    
    return left, right

# 1. Sử dụng độ lợi thông tin, chọn thuộc tính phân hoạch có giá trị lớn nhất
# Tính độ hỗn loạn thông tin trước khi phân hoạch (entropy)
def info(data):
    label_column = data[:, -1]
    classes, counts_classes = np.unique(label_column, return_counts=True)

    entropy = 0.0
    for i in range(len(counts_classes)):
        # Tính xác suất xuất hiện cho từng phân lớp <- lưu vào biến p
        p = counts_classes[i] / sum(counts_classes)
        # Ta có công thức: Info(D) =  entropy(p1, p2,..., pn) = (-p1*log2(p1)) + (-p2*log2(p2)) + ... + (-pn*log2(pn))
        # Vì thế ta cần cộng dồn các giá trị [-p*log2(p)] vào biến entropy
        entropy += (-p * np.log2(p))

    return entropy

# Tính độ hỗn loạn thông tin sau khi phân hoạch (Info_A)
def info_A(left, right):
    # Ta có công thức: Info_A(D) = (D1/D)*Info(D1) + (D2/D)*Info(D2) + ... + (Dv/D)*Info(Dv)
    # Ta phân hoạch DL ra thành 2 phần: left và right
    # Do đó ta sẽ tính Info(left) và Info(right), sau đó cộng 2 kết quả này <- overall_entropy
    D = len(left) + len(right)  # tính tổng số lượng phần tử trong tập dữ liệu D
    D1 = len(left)
    D2 = len(right)

    overall_entropy = ((D1 / D) * info(left)) + ((D2 / D) * info(right))

    return overall_entropy

# Hàm chọn ra thuộc tính và giá trị của thuộc tính đó để phân hoạch dựa vào giá trị độ lợi thông tin lớn nhất
def choose_best_split(data, point_splits):
    # để tìm được độ lợi thông tin lớn, ta cần khởi tạo giá trị ban đầu cho biến information_gain có giá trị là âm vô cùng
    information_gain = -9999999
    # duyệt qua các cột trong point_splits
    for col_index in point_splits:
        # duyệt qua các giá trị trong từng cột
        for value in point_splits[col_index]:
            # phân hoạch dữ liệu và tính entropy tại các điểm trong point_splits
            left, right = binary_split_data(data, split_column = col_index, split_value = value)
            current_information_gain = info(data) - info_A(left, right)
            
            # nếu tìm được 1 giá trị info_gain mới lớn hơn giá trị info_gain cũ
            if current_information_gain > information_gain:
                information_gain = current_information_gain # cập nhật lại giá trị độ lợi thông tin
                best_split_column = col_index # lưu lại vị trí cột (thuộc tính) phân hoạch
                best_split_value = value # lưu lại giá trị phân hoạch
    
    # cuối cùng trả về thuộc tính (vị trí cột) và giá trị để phân hoạch
    return best_split_column, best_split_value

# hàm kiểm tra giá trị của 1 cột thuộc tính, trả về True nếu là giá trị liên tục, ngược lại trả về false
def is_continuous(column):
    unique_column_values = np.unique(column)
    if len(unique_column_values) > 2:
        return True
    return False

# phân chia lại tập dữ liệu thành hai phần: 1 phần chứa các thuộc tính có giá trị liên tục, 1 phần chứa các giá trị không liên tục
def determine_type_of_feature(data):
    feature_types = []
    no_rows, no_cols = data.shape # lấy số lượng hàng, cột

    # duyệt qua từng cột
    for column_index in range(no_cols-1):
        column_value = data.iloc[:, column_index] # lấy các giá trị tại 1 cột
        # kiểm tra nếu giá trị tại cột đó có giá trị liên tục
        if is_continuous(column_value):
            feature_types.append("continuous") # đặt tên là continuous (liên tục)
        else: # ngược lại đặt tên là categorical (không liên tục)
            feature_types.append("categorical")
    
    return feature_types

def decision_tree_classifier(dt, counter=0, min_samples_leaf=2, max_depth=5):    
    # nút gốc
    if counter == 0:
        global COLUMN_HEADERS, FEATURE_TYPES
        COLUMN_HEADERS = dt.columns
        FEATURE_TYPES = determine_type_of_feature(dt)
        data = dt.values
    else:
        data = dt         

    # trường hợp để dừng phân hoạch
    if (check_purity(data)) or (len(data) < min_samples_leaf) or (counter == max_depth):
        leaf_node = create_leaf_node(data)
        # print(leaf_node)
        return leaf_node
    
    else:    
        counter += 1
        # tìm các điểm phân hoạch trên toàn bộ tập DL truyền vào (data)
        point_splits = get_point_splits(data)
        # print("Các điểm phân hoạch:\n", point_splits)
        # chọn thuộc tính và giá trị của thuộc tính để phân hoạch
        split_column, split_value = choose_best_split(data, point_splits)
        # print(split_column, split_value)
        # phân hoạch nhị phân tập DL ta thành 2 cây con trái và phải dựa trên thuộc tính và giá trị của thuộc tính đó vừa tìm được ở trên
        left, right = binary_split_data(data, split_column, split_value)

        print(left)
        
        # kiểm tra dữ liệu rỗng
        if len(left) == 0 or len(right) == 0:
            leaf_node = create_leaf_node(data)
            return leaf_node
        
        # tạo cây con
        feature_name = COLUMN_HEADERS[split_column] # tên thuộc tính
        type_of_feature = FEATURE_TYPES[split_column]
        
        if type_of_feature == "continuous":
            question = "{} <= {}".format(feature_name, split_value) # tạo nút điều kiện
        else:
            question = "{} = {}".format(feature_name, split_value)
        
        sub_tree = {question: []}
        
        # lặp lại việc phân hoạch 1 cách đệ quy cho cây con trái và phải
        yes_answer = decision_tree_classifier(left, counter, min_samples_leaf, max_depth)
        no_answer = decision_tree_classifier(right, counter, min_samples_leaf, max_depth)
        
        if yes_answer == no_answer:
            sub_tree = yes_answer
        else:
            sub_tree[question].append(yes_answer)
            sub_tree[question].append(no_answer)
        
        return sub_tree

'''
4. Dự đoán nhãn cho tập dữ liệu kiểm tra
'''
# Dự đoán nhãn cho từng dòng dữ liệu
def predict_row(tree, row_data_test):
    left = 0
    right = 1
    pprint
    for label in tree:
        # Tách keys đầu tiên trong từ điển, xét nút gốc
        '''Ví dụ: Với cây bên dưới
        {'petalLength <= 2.5999999999999996': ['Iris-setosa',
                                       {'petalWidth <= 1.65': [{'petalLength <= 4.95': ['Iris-versicolor',
                                                                                        {'petalWidth <= 1.55': ['Iris-virginica',
                                                                                                                'Iris-versicolor']}]},
                                                               {'petalLength <= 4.85': [{'sepalWidth <= 3.1': ['Iris-virginica',
                                                                                                               'Iris-versicolor']},
                                                                                        'Iris-virginica']}]}]}
        Sau khi thực thi câu lệnh split_label = label.split() sẽ có kết quả:                                                                               
            ['petalLength', '<=', '2.5999999999999996']
            split_label[0]: thuộc tính
            split_label[2]: giá trị thuộc tính 
        '''
        split_label = label.split()
        properties = split_label[0]
        value = float(split_label[2])
    c = pd.to_numeric(row_data_test[properties])
    # Nếu giá trị dữ liệu test <= giá trị thuộc tính của cây ->Xét con bên trái
    if c <= value:
        # Nếu con trái chưa phải nút lá thì đệ quy cho đến khi tìm thấy nút lá
        if isinstance(tree.get(label)[left], dict):
            return predict_row(tree.get(label)[left], row_data_test)
        else:
            return tree.get(label)[left]
    else:
        if isinstance(tree.get(label)[right], dict):
            return predict_row(tree.get(label)[right], row_data_test)
        else:
            return tree.get(label)[right]

# Dự đoán nhãn cho tập dữ liệu test
def predict_DT(tree, data_test):
    y_pred = []
    for row in range(0, len(data_test)):
        row_data = data_test.iloc[row]
        y_pred.append(predict_row(tree, row_data))
    return y_pred


'''5. Tính toán độ chính xác tổng thể'''
def cal_accuracy_all_DT(y_pred, y_test):
    correct = 0
    # Lặp tất cả các nhãn trong tập test
    for i in range(0, len(y_test)):
        # So sánh nhãn tập test với giá trị dự đoán, nếu giống tăng 1, không giống không cần quan tâm đâu
        if y_pred[i] == y_test[i]:
            correct += 1
    accuracy_score = correct / len(y_pred) * 100
    return accuracy_score

'''6.Hàm tính độ chính xác cho từng thực thể'''
def confusion_matrix_DT(y_test, y_pred, label):
    #print(y_test,"\n",y_pred)
    # Tạo ma trận dự đoán cho từng giá trị thông qua ma trận 2 chiều
    arr = [[]]
    # Tạo từ điển để lưu số lượng dự đoán cho từng label
    # Ví dụ: correct = { 0:8,1:5}
    # Tức là, trong tập dự đoán tìm thấy 8 nhãn có label là 0 và 5 nhãn có label là 1
    correct = {}
    t = 0
    # Lặp lần lượt các nhãn có trong tập dữ liệu test
    for i in label:
        # Khởi tạo giá trị ban đầu cho từng label = 0
        for e in label:
            correct[e] = 0
    
        lb = i
        # Duỵet lần lượt tất cả cac nhãn trong tập test
        for j in range(0, len(y_test)):
            # Chi thực hiện kiểm tra nếu nhan phan trong tập test là label đang xét
            if y_test[j] == lb:
                for key in correct.keys():
                    # Tinh do chinh xac va sai  cho từng label  trong tung lop
                    if key == y_pred[j]:
                    
                        # Tang gia tri y_pred cho nhan = key
                        correct[key] = correct[key] + 1
        # Add gia tri từng label vào ma trận dự đoán
        for key in correct.keys():
            arr[t].append(correct[key])
        t += 1
        arr.append([])
    # print(correct)
    arr.pop()
    # print(arr)
    for i in arr:
        print(i)

def main():
    dataset = read_file()
    print(dataset)
    # print(is_continuous(dataset.iloc[:, 0]))
    # point_splits = get_point_splits(dataset.values)

    # print(point_splits)

    # print(info(dataset.values)) # 1.0

    # tree = decision_tree_classifier(dataset, max_depth=11)
    # pprint(tree)


# gọi hàm main
if __name__ == "__main__":
    main()
