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

'''
Các bước và các hàm cần định nghĩa khi xây dựng Cây quyết định:
    1. Đọc dữ liệu (tiền xử lý nếu cần)
    2. Phân chia tập DL theo nghi thức hold-out (tập dữ liệu được chia làm 3 phần, trong đó 2 phần train, 1 phần test)
    3. Xây dựng cây
    4. Dự đoán nhãn cho tập dữ liệu kiểm tra
    5. Tính toán độ chính xác tổng thể' và độ chính xác cho từng phân lớp
'''

'''
- Hàm đọc file
- Input: không
- Output: trả về tập dữ liệu
'''
def read_file():
    dataset = pd.read_csv("../data_set/heart_failure_clinical_records_dataset.csv", delimiter=",")
    return dataset

'''
- Hàm phân chia tập dữ liệu theo nghi thức hold-out
- Input:
    + dataset: tập dữ liệu đọc từ file
    + test_size: kích thước tập dữ liệu kiểm tra
- Output:
	+ train_data: tập dữ liệu huấn luyện
	+ test_data: tập dữ liệu kiểm tra
'''
def train_test_split(dataset, test_size, random_state):
    random.seed(random_state)

    if isinstance(test_size, float):
        test_size = round(test_size * len(dataset))  # để đảm bảo test_size luôn là 1 số nguyên

    # lấy tất cả các chỉ số trong tập DL gốc, sau đó chuyển nó sang dạng list
    indices = dataset.index.tolist()

    # random các chỉ số cho tập test, lưu vào test_indices
    # hàm random.sample dùng để lấy ngẫu nhiên k phần tử trong danh sách indices
    # ví dụ: test_size = 100 thì hàm này sẽ lấy ngẫu nhiên 100 phần tử trong danh sách indices 
    test_indices = random.sample(population=indices, k=test_size)
    # print(test_indices)

    # lấy các giá trị và lưu vào tập DL test thông qua các chỉ số trong mảng test_indices
    test_data = dataset.iloc[test_indices]
    
    # lấy các giá trị bỏ đi các chỉ số trong mảng test_indices và lưu vào tập DL train
    train_data = dataset.drop(test_indices)

    return train_data, test_data

'''
- Hàm kiểm tra dữ liệu trong 1 nút có thuần nhất hay không. Một nút được xem là có DL thuần nhất 
khi nút đó chỉ chứa duy nhất 1 nhãn.
- Input: dữ liệu cần kiểm tra
- Output: trả về True nếu dữ liệu là thuần nhất, ngược lại trả về False

'''
def check_purity(data):
    # lấy nguyên cột nhãn trong data
    label_column = data[:, -1]

    # loại bỏ các giá trị trùng trên cột nhãn
    # Ví dụ: đối với tập dữ liệu hoa Iris 
    # cột nhãn có dạng = ['Iris-setosa' 'Iris-setosa' 'Iris-setosa' 'Iris-versicolor' 'Iris-versicolor' 'Iris-virginica' 'Iris-virginica']
    classes = np.unique(label_column)
    # sau khi lấy unique trên cột nhãn thì classes = ['Iris-setosa' 'Iris-versicolor' 'Iris-virginica']
    # kiểm tra nếu len(classes) == 1 -> thuần nhất -> trả về True, ngược lại trả về False
    if len(classes) == 1:
        return True
    else:
        return False

'''
- Hàm tạo nút lá
- Input: dữ liệu
- Output: nhãn trên nút lá
'''
def create_leaf_node(data):
    label_column = data[:, -1]

    # nếu dữ liệu trong 1 nút đã thuần nhất rồi thì gán nhãn đó cho nút lá luôn
    if check_purity(data) == True:
        leaf_node = np.unique(label_column)
    
    # nếu dữ liệu chưa thuần nhất nhưng cần dừng sớm quá trình phân hoạch
    # thì sẽ lấy giá trị nhãn có số lượng nhiều nhất cho nút lá
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
        
        # lấy nhãn tương ứng với vị trí vừa tìm được ở bên trên
        leaf_node = classes[index]

    return leaf_node

'''
- Hàm lấy các điểm phân hoạch. Các điểm phân hoạch sẽ là các điểm có sự thay đổi về lớp
- Input: dữ liệu
- Output: một từ điển dạng {cột: mảng các điểm phân hoạch}
'''
def get_point_splits(data):
    # khởi tạo một từ điển rỗng
    point_splits = {}

    '''
    Kiểu từ điển (dictionary) là các phần tử của nó được biểu diễn dưới dạng {key: value}
    Ta có thể truy xuất giá trị của 1 phần tử trong từ điển thông qua key
    Trong từ điển, value của 1 key có thể là 1 số, chuỗi, list hoặc 1 từ điển...
    Ví dụ: 
    - Khởi tạo 1 từ điển gồm 3 phần tử  như sau:
        student_info = {
            fullname: "Luu Thanh Van",
            age: 21,
            major: "Computer Science"
        }
    - Truy xuất:
        student_name = student_info["fullname"] -> result: Luu Thanh Van
    
    Trong hàm này, point_splits có dạng {column_index: []}
        - column_index: chỉ số cột
        - []: 
            + Đối với cột thuộc tính có dữ liệu liểu liên tục thì mình sẽ xác định tất cả các điểm phân hoạch trong một cột là 
            các điểm có sự thay đổi về lớp sau đó lưu vào mảng này.

            + Đối với cột thuộc tính có dữ liệu liểu rời rạc thì mảng có dạng [0, 1]
    
            Ví dụ: point_splits = { 0: [49, 65, 50, ...], 
                                    1: [0, 1], ... }
    '''
    # lấy hình dạng của dữ liệu, shape sẽ trả về 2 giá trị: giá trị đầu tiên là tổng số  hàng (no_rows), giá trị thứ hai là tổng số cột (no_cols)
    no_rows, no_cols = data.shape

    # duyệt qua các cột trong data
    for column_index in range(no_cols-1):
        # lấy các giá trị trong 1 cột (col_index sẽ chạy từ 0 đến no_cols-1)
        column_values = data[:, column_index]

        # loại bỏ các giá trị trùng
        unique_column_values = np.unique(column_values)

        type_of_feature = FEATURE_TYPES[column_index]

        # nếu cột dữ liệu là cột thuộc tính có giá trị kiểu số  (giá trị liên tục)
        if type_of_feature == "continuous":
            # column_index là các key trong point_splits
            # khởi tạo giá trị tại key = column_index là 1 mảng rỗng
            point_splits[column_index] = []

            # duyệt qua từng dòng trong 1 cột thuộc tính
            for row_index in range(no_rows-1):
                # không xét dòng đầu tiên (dòng có chỉ số = 0), chỉ xét từ dòng có chỉ số 1 đến no_rows-1
                if row_index != 0:
                    # lấy giá trị trước đó
                    previous_class_value = data[row_index-1, -1] 
                    
                    # lấy giá trị hiện tại
                    current_class_value = data[row_index, -1]
                    
                    # lấy giá trị kế tiếp
                    next_class_value = data[row_index+1, -1]

                    # nếu mà giá trị nhãn hiện tại khác với giá trị nhãn trước và sau đó thì chọn điểm hiện tại là điểm phân hoạch
                    if (current_class_value != previous_class_value and current_class_value != next_class_value):
                        column_value = data[row_index, column_index]
                        point_splits[column_index].append(column_value)
        
        # trường hợp cho cột thuộc tính có giá trị kiểu rời rạc
        else:
            point_splits[column_index] = unique_column_values
    
    # print(point_splits)
    return point_splits

'''
- Hàm phân hoạch nhị phân dựa trên 1 giá trị ngưỡng cho 1 thuộc tính (cột) trên tập dữ liệu
- Input:
    + data: dữ liệu
    + split_column: chỉ số cột cần phân hoạch
    + split_value: giá trị phân hoạch (ngưỡng)
- Output: hai mảng left và right
    + Đối với thuộc tính có giá trị kiểu liên tục
        - left là các giá trị nhỏ hơn hoặc bằng giá trị phân hoạch
        - right là các giá trị lớn hơn giá trị phân hoạch
    + Đối với thuộc tính có giá trị kiểu rời rạc
        - left là các giá trị bằng giá trị phân hoạch
        - right là các giá trị khác giá trị phân hoạch
'''
def binary_split_data(data, split_column, split_value):
    split_column_values = data[:, split_column]

    type_of_feature = FEATURE_TYPES[split_column]

    if type_of_feature == "continuous":
        left = data[split_column_values <= split_value]
        right = data[split_column_values > split_value]
    else:
        left = data[split_column_values == split_value]
        right = data[split_column_values != split_value]
    
    return left, right

'''
- Hàm tính độ hỗn loạn thông tin trước khi phân hoạch
- Input: dữ liệu
- Output: entropy (Info(D))
'''
def info(data):
    label_column = data[:, -1]
    classes, counts_classes = np.unique(label_column, return_counts=True)

    entropy = 0.0
    for i in range(len(counts_classes)):
        # Tính xác suất xuất hiện cho từng phân lớp -> lưu vào biến p
        p = counts_classes[i] / sum(counts_classes)
        # Ta có công thức: Info(D) =  entropy(p1, p2,..., pn) = (-p1*log2(p1)) + (-p2*log2(p2)) + ... + (-pn*log2(pn))
        # Vì thế ta cần cộng dồn các giá trị [-p*log2(p)] vào biến entropy
        entropy += (-p * np.log2(p))

    return entropy

'''
- Hàm tính độ hỗn loạn thông tin sau khi phân hoạch
- Input: dữ liệu đã phân hoạch hai phần left và right
- Output: overall_entropy (Info_A(D))
'''
def info_A(left, right):
    '''
    Ta có công thức: Info_A(D) = (D1/D)*Info(D1) + (D2/D)*Info(D2) + ... + (Dv/D)*Info(Dv)
    Ta phân hoạch DL ra thành 2 phần: left và right. 
    Do đó ta sẽ tính Info(left) và Info(right) sau đó cộng 2 kết quả này -> lưu vào biến overall_entropy
    '''
    D = len(left) + len(right)  # tính tổng số lượng phần tử trong tập dữ liệu D
    D1 = len(left)
    D2 = len(right)

    overall_entropy = ((D1 / D) * info(left)) + ((D2 / D) * info(right))

    return overall_entropy
'''
- Hàm chọn ra thuộc tính và giá trị của thuộc tính đó để phân hoạch dựa vào giá trị độ lợi thông tin lớn nhất
- Input:
    + data: dữ liệu
    + point_splits: các điểm phân hoạch (output của hàm get_point_splits)
- Output:
    + best_split_column: thuộc tính (chỉ số cột) mà ta sẽ phân hoạch dữ liệu trên cột đó
    + best_split_value: giá trị phân hoạch
'''
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

'''
- Hàm kiểm tra giá trị của 1 cột thuộc tính xem các giá trị của cột đó là giá trị liên tục hay rời rạc
- Input: cột dữ liệu cần kiểm tra 
- Output: trả về True nếu cột đó có giá trị liên tục, ngược lại trả về False
'''
def is_continuous(column):
    unique_column_values = np.unique(column)
    if len(unique_column_values) > 2:
        return True
    return False

'''
- Hàm xét xem giá trị của một cột thuộc tính trong 1 tập dữ liệu là kiểu liên tục hay rời rạc
- Input: dữ liệu
- Output: mảng feature_types
    + chỉ số mảng tương ứng với chỉ số cột
    + giá trị ứng với các chỉ số mảng là continuous hoặc categorical
        - continuous: là các cột có giá trị kiểu liên tục
        - categorical: là các cột có giá trị kiểu rời rạc
'''
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

'''    
- Hàm xây dựng cây
- Input:
    + dt: dữ liệu
    + counter: đếm độ sâu của cây
    + min_samples_leaf: dùng để dừng sớm quá trình phân hoạch
    + max_depth: độ sâu của cây
- Output: cây
Ví dụ:                     
example_tree = {'petal_width <= 0.8': ['Iris-setosa',
                        {'petal_width <= 1.65': [{'petal_length <= 4.95': ['Iris-versicolor',
                                                                           'Iris-virginica']},
                                                 'Iris-virginica']}]}
'''
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
        return leaf_node
    
    else:    
        counter += 1
        # tìm các điểm phân hoạch trên toàn bộ tập DL truyền vào (data)
        point_splits = get_point_splits(data)
        # chọn thuộc tính và giá trị của thuộc tính để phân hoạch
        split_column, split_value = choose_best_split(data, point_splits)
        # phân hoạch nhị phân tập DL ta thành 2 cây con trái và phải dựa trên thuộc tính và giá trị của thuộc tính đó vừa tìm được ở trên
        left, right = binary_split_data(data, split_column, split_value)
        
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
- Hàm dự đoán nhãn cho từng dòng dữ liệu
- Input:
    +Cây
    +Dòng dữ liệu cần dự đoán trong tập test.
- Output:Giá trị nhãn sau khi dự đoán.
'''
def predict_row(tree, row_data_test):
    left = 0
    right = 1

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
            split_label[2]: ngưỡng (giá trị phân hoạch dữ liệu)
        '''
        split_label = label.split()
        properties = split_label[0]
        value = float(split_label[2])

    # row_data_test[properties]: lấy giá trị của thuộc tính thông qua tên thuộc tính, sau đó chuyển nó sang dạng số
    c = pd.to_numeric(row_data_test[properties])
    
    # Nếu giá trị dữ liệu test <= giá trị thuộc tính của cây -> Xét con bên trái
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

'''
- Hàm dự đoán nhãn cho tập dữ liệu test
- Input:
    +Cây
    +Tập dữ liệu test
- Output: Mảng chứa các giá trị nhãn cho cả tập test
'''
def predict_DT(tree, data_test):
    y_pred = []
    for row in range(0, len(data_test)):
        row_data = data_test.iloc[row]
        y_pred.append(predict_row(tree, row_data))
    return y_pred

''' 
- Hàm tính toán độ chính xác tổng thể'
- Input:
    +Tập các giá trị nhãn dự đoán được
    +Tập các giá trị nhãn của tập test
- Output: Độ chính xác tổng thể (Đơn vị %)
'''
def cal_accuracy_all_DT(y_pred, y_test):
    correct = 0
    # Lặp tất cả các nhãn trong tập test
    for i in range(0, len(y_test)):
        # So sánh nhãn tập test với giá trị dự đoán, nếu giống tăng 1, không giống không cần quan tâm đâu
        if y_pred[i] == y_test[i]:
            correct += 1
    accuracy_score = correct / len(y_pred) * 100
    return accuracy_score

''' 
- Hàm tính độ chính xác cho từng thực thể 
- Input:
    +Tập các giá trị nhãn của tập test
    +Tập các giá trị nhãn dự đoán được
    +Các thực thể cần tính độ chính xác theo thứ tự mong muốn
- Output: Ma trận độ chính xác của từng thực thể
'''
def confusion_matrix_DT(y_test, y_pred, label):
    # Tạo ma trận dự đoán cho từng giá trị thông qua ma trận 2 chiều
    arr = [[]]
    # Tạo từ điển để lưu số lượng dự đoán cho từng label
    # Ví dụ: correct = { 0:8, 1:5}
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
                    # Tinh do chinh xac va sai cho từng label trong tung lop
                    if key == y_pred[j]:
                        # Tang gia tri y_pred cho nhan = key
                        correct[key] = correct[key] + 1

        # Add gia tri từng label vào ma trận dự đoán
        for key in correct.keys():
            arr[t].append(correct[key])

        t += 1
        arr.append([])

    arr.pop()
    for i in arr:
        print(i)

# Giải thuật Naive Bayes - thư viện sklearn
def naive_bayes_classifier(X_train, y_train):
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model

def predict_NB(X_test, model):
    y_pred = model.predict(X_test)
    return y_pred

def cal_accuracy_NB(y_test, y_pred):
    print(confusion_matrix(y_test, y_pred, labels=[1, 0]))
    print("Accuracy is ", accuracy_score(y_test, y_pred)*100)

def main():
    dataset = read_file()

    np.random.seed(2)
    random_state = np.random.choice(100, size=11, replace=False)
    max_depth = np.random.choice(11, size=11, replace=False)
    min_samples_leaf = np.random.choice(100, size=11, replace=False)

    # loại bỏ số 0
    random_state = np.delete(random_state, np.where(random_state == 0))
    max_depth = np.delete(max_depth, np.where(max_depth == 0))
    min_samples_leaf = np.delete(min_samples_leaf, np.where(min_samples_leaf == 0))
    # print("random_state", random_state)
    # print("max_depth", max_depth)
    # print("min_samples_leaf", min_samples_leaf)
    
    dataset = read_file()
    time1 = []
    time2 = []
    phantram1 = []
    phantram2 = []
    z = []
    i=0
    for i in range(0, 10):
        random_num = random_state[i] + i
        
        train_data, test_data = train_test_split(dataset, test_size=1/3.0, random_state=random_num)

        y_test = test_data.iloc[:, -1]  # lấy cột cuối cùng
        X_test = test_data.iloc[:, :-1]  # bỏ cột cuối cùng, lấy các cột còn lại

        y_train = train_data.iloc[:, -1]
        X_train = train_data.iloc[:, :-1] 

        print("Decision Tree")
        print(i+1, "random state =", random_num)
        tree = decision_tree_classifier(train_data, min_samples_leaf=min_samples_leaf[i], max_depth=max_depth[i])
        # pprint(tree)
        seconds = time.time()
        y_pred_DT = predict_DT(tree, X_test)
        time1.append(time.time()-seconds)
        # print(y_pred)
        # Chuyển kiểu dữ liệu y_test để dễ dàng tính độ chính xác tổng thể
        y_test_DT = y_test.tolist()
        print("Do chinh xac: ", cal_accuracy_all_DT(y_pred_DT, y_test_DT))
        phantram1.append(cal_accuracy_all_DT(y_pred_DT, y_test_DT))
        confusion_matrix_DT(y_test_DT, y_pred_DT ,[1, 0])
        print("===========================================")

        print("Naive Bayes")
        model = naive_bayes_classifier(X_train, y_train)
        seconds = time.time()
        y_pred_NB = predict_NB(X_test, model)
        time2.append(time.time()-seconds)
        cal_accuracy_NB(y_test, y_pred_NB)
        phantram2.append(accuracy_score(y_test, y_pred_NB)*100)
        print("===========================================")
        z.append(i+1)
        i+=1
    
    print("\n\nBảng thống kê độ chính xác và thời gian của 2 giải thuật\n")
    print("|---|---------------------|-------------------------------------------------|")
    print("|   |          Độ chính xác             |             Thời gian             |")
    print("|STT|-----------------------------------|-----------------------------------|")
    print("|   |  Decision Tree  |   Naive Bayes   |  Decision Tree  |   Naive Bayes   |")
    print("|---|-----------------|-----------------|-----------------|-----------------|")
    for i in range(len(phantram1)):
        print("|",'{0:3d}'.format(i+1),"|",'{0:17.5f}'.format(phantram1[i]),"|",'{0:17.5f}'.format(phantram2[i]),"|",'{0:17.5f}'.format(time1[i]),"|",'{0:17.5f}'.format(time2[i]),"|",sep='')
        print("|---|-----------------|-----------------|-----------------|-----------------|")
    print("\nĐộ chính và thời gian trung bình của giải thuật cây quyết định là:",'{0:10.5f}'.format(sum(phantram1)/len(phantram1)),'{0:10.5f}'.format(sum(time1)/len(time1)))
    print("\nĐộ chính và thời gian trung bình của giải thuật Bayes là:",'{0:10.5f}'.format(sum(phantram2)/len(phantram2)),'{0:10.5f}'.format(sum(time2)/len(time2)))

    plt.axis([0,10,65,100])
    plt.plot(z,phantram1,color="blue")
    plt.plot(z,phantram2,color="red")
    plt.xlabel("Lần lặp")
    plt.ylabel("Độ chính xác")
    plt.show() 
    
# gọi hàm main
if __name__ == "__main__":
    main()
