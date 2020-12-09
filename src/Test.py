import csv
import random
import pandas as pd
import numpy as np
import array as arr
from pprint import pprint
import math
from collections import deque

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
    # dataset = pd.read_csv("../data_set/test1.csv", delimiter=",")
    dataset = pd.read_csv("../data_set/heart_failure_clinical_records_dataset.csv", delimiter=",")
    return dataset

# hàm kiểm tra dữ liệu trong 1 nút có thuần nhấy hay không
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

def get_point_splits_2(data):
    point_splits = {} # khởi tạo 1 từ điển rỗng
    
    # lấy hình dạng của dữ liệu, shape sẽ trả về 2 giá trị: giá trị đầu tiên là tổng số  hàng (no_rows), giá trị thứ hai là tổng số cột (no_cols)
    no_rows, no_cols = data.shape
    
    # duyệt qua các cột trong data
    for col_index in range(no_cols - 1):
        # col_index là các key trong point_splits
        # khởi tạo giá trị tại key = col_index là 1 mảng rỗng
        point_splits[col_index] = []
        
        # lấy các giá trị trong 1 cột (col_index sẽ chạy từ 0,1, đến no_cols-1)
        column_values = data[:, col_index]
        
        unique_column_values = np.unique(column_values) # loại bỏ các giá trị trùng

        for index in range(len(unique_column_values)):
            # dòng này để tránh index = 0, mình sẽ chạy từ index = 1
            # vì mình cần lấy giá trị trước đó + giá trị hiện tại, mà tại index = 0 thì không có giá trị trước nó
            if index != 0:
                # lấy giá trị hiện tại trong mảng unique_column_values
                current_value = unique_column_values[index]
                # lấy giá trị trước đó trong mảng unique_column_values
                previous_value = unique_column_values[index - 1]
                # xác định điểm phân hoạch
                point_split = (current_value + previous_value) / 2 # công thức này tham khảo trên mạng
                # thêm điểm phân hoạch vào mảng, tại vị trí col_index
                point_splits[col_index].append(point_split)
    
    # xem cụ thể output ở phần test hàm này bên dưới nhe
    return point_splits

def get_point_splits(data):
    point_splits = {}
    no_rows, no_cols = data.shape
    for col_index in range(no_cols-1):
        point_splits[col_index] = []
        for row_index in range(no_rows):
            if row_index != 0:
                current_column_class = data[row_index, -1]
                previous_column_class = data[row_index-1, -1]

                if current_column_class != previous_column_class:
                    column_value = data[row_index, col_index]
                    point_splits[col_index].append(column_value)

    # print(point_splits)
    return point_splits

# Phân hoạch nhị phân dựa trên 1 giá trị ngưỡng cho 1 thuộc tính (cột) trên tập dữ liệu
def binary_split_data(data, split_column, split_value):
    split_column_values = data[:, split_column]
    # print(split_column_value)
    # type_of_feature = FEATURE_TYPES[split_column]
    # if type_of_feature == "continuous":
    # left = data[split_column_values >= split_value]
    # right = data[split_column_values < split_value]
    # else:
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
            question = "{} < {}".format(feature_name, split_value)
        
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
    # Tạo ma trận dự đoán cho từng giá trị thông qua ma trận 2 chiều
    arr = [[]]
    # Tạo từ điển để lưu số lượng dự đoán cho từng label
    # Ví dụ: correct = { 0:8,1:5}
    # Tức là, trong tập dự đoán tìm thấy 8 nhãn có label là 0 và 5 nhãn có label là 1
    correct = {}
    t = 0
    # Lặp lần lượt các nhãn có trong tập dữ liệu test
    for i in np.unique(y_test):
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

# Giải thuật Naive Bayes - thư viện sklearn
def naive_bayes_classifier(X_train, y_train):
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model

def predict_NB(X_test, model):
    y_pred = model.predict(X_test)
    return y_pred

# Calculating accuracy
def cal_accuracy_NB(y_test, y_pred):
    # Calculating accuracy using confusion matrix
    print(confusion_matrix(y_test, y_pred, labels=[1, 0]))
    print("Accuracy is ", accuracy_score(y_test, y_pred)*100)

def main():
    # dataset = read_file()
    # print(dataset)
    # point_split = get_point_splits(dataset.values)
    # print(point_split[4])
    # left, right = binary_split_data(dataset.values, 9, point_split[9][0])
    # print(left)
    # print(right)
    # print(info(left))
    # print(info_A(left, right))

    # tree = decision_tree_classifier(dataset, max_depth=10)
    # print(tree)
    dataset = read_file()
    #print(dataset)

    for i in range(11):
        print(i)
        random.seed(i)
        train_data, test_data = train_test_split(dataset, test_size=0.1)

        y_test = test_data.iloc[:, -1]  # lấy cột cuối cùng
        X_test = test_data.iloc[:, :-1]  # bỏ cột cuối cùng, lấy các cột còn lại

        y_train = train_data.iloc[:, -1]
        X_train = train_data.iloc[:, :-1]


        print("Decision Tree")
        tree = decision_tree_classifier(train_data, max_depth=11)
        #pprint(tree)
        y_pred_DT = predict_DT(tree, X_test)
        # print(y_pred)
        # Chuyển kiểu dữ liệu y_test để dễ dàng tính độ chính xác tổng thể
        y_test_DT = y_test.tolist()
        print("Do chinh xac: ", cal_accuracy_all_DT(y_pred_DT, y_test_DT))
        confusion_matrix_DT(y_test_DT, y_pred_DT ,[0, 1])
        print("===========================================")

        print("Naive Bayes")
        model = naive_bayes_classifier(X_train, y_train)
        y_pred_NB = predict_NB(X_test, model)
        cal_accuracy_NB(y_test, y_pred_NB)
        print("===========================================")


# gọi hàm main
if __name__ == "__main__":
    main()


    '''
    for i in range(11):
        print(i)
        random.seed(i)
        train_data, test_data = train_test_split(dataset, test_size=0.1)

        # feature_types = determine_type_of_feature(train_data)
        # print(feature_types)

        # print("Train data: ", train_data)
        # print("Test data: ", test_data)
        # y_test = test_data.iloc[:,4]
        # X_test = test_data.iloc[:,0:4]
        y_test = test_data.iloc[:, -1]  # lấy cột cuối cùng
        X_test = test_data.iloc[:, :-1]  # bỏ cột cuối cùng, lấy các cột còn lại

        y_train = train_data.iloc[:, -1]
        X_train = train_data.iloc[:, :-1]

        # print(y_test)
        # print(X_test)
        # Test hàm check_purity(data)
        # print(check_purity(train_data.values)) # kết quả là False, bởi vì giá trị nhãn trong tập DL train bao gồm 3 nhãn -> chưa thuần nhất
        # dt = train_data.head().values # lấy 5 giá trị đầu tiên trong tập DL train
        # print(dt)'''
        '''
        sepalLength  sepalWidth  petalLength  petalWidth      species
        0          5.1         3.5          1.4         0.2  Iris-setosa
        1          4.9         3.0          1.4         0.2  Iris-setosa
        2          4.7         3.2          1.3         0.2  Iris-setosa
        3          4.6         3.1          1.5         0.2  Iris-setosa
        4          5.0         3.6          1.4         0.2  Iris-setosa
        '''
        # print(check_purity(dt))
        # kết quả là True, bởi vì 5 phần tử đầu trong tập DL huấn luyện đều có cùng 1 nhãn là Iris-setosa(check_purity(dt))

        # Test hàm create_leaf_node()
        # leaf_node = create_leaf_node(train_data.values)
        # print(leaf_node)

        # Test hàm get_point_splits(data) (t)
        # point_splits = get_point_splits(train_data.values)
        # print(point_splits)
        ''' Result: point_splits = {key: value, } <=> {col_index: [], }
        {
            0: [4.35, 4.45, 4.55, 4.65, 4.75, 4.85, 4.95, 5.05, 5.15, 5.25, 5.35, 5.45, 5.55, 5.65, 5.75, 5.85, 5.95, 6.05, 6.15, 6.25, 6.35, 6.45, 6.55, 6.65, 6.75, 6.85, 6.95, 7.05, 7.15, 7.4, 7.65, 7.800000000000001], 
            1: [2.1, 2.25, 2.3499999999999996, 2.45, 2.55, 2.6500000000000004, 2.75, 2.8499999999999996, 2.95, 3.05, 3.1500000000000004, 3.25, 3.3499999999999996, 3.45, 3.55, 3.6500000000000004, 3.75, 3.8499999999999996, 3.95, 4.05, 4.15, 4.300000000000001], 
            2: [1.05, 1.15, 1.25, 1.35, 1.45, 1.55, 1.65, 1.7999999999999998, 2.5999999999999996, 3.4, 3.55, 3.6500000000000004, 3.75, 3.8499999999999996, 3.95, 4.05, 4.15, 4.25, 4.35, 4.45, 4.55, 4.65, 4.75, 4.85, 4.95, 5.05, 5.15, 5.25, 5.35, 5.45, 5.55, 5.65, 5.75, 5.85, 5.95, 6.05, 6.25, 6.5, 6.65, 6.800000000000001], 
            3: [0.15000000000000002, 0.25, 0.35, 0.45, 0.55, 0.8, 1.05, 1.15, 1.25, 1.35, 1.45, 1.55, 1.65, 1.75, 1.85, 1.95, 2.05, 2.1500000000000004, 2.25, 2.3499999999999996, 2.45]
        }
        '''
        '''
        print("Decision Tree")
        tree = decision_tree_classifier(train_data, max_depth=11)
        #pprint(tree)
        y_pred_DT = predict_DT(tree, X_test)
        # print(y_pred)
        # Chuyển kiểu dữ liệu y_test để dễ dàng tính độ chính xác tổng thể
        y_test_DT = y_test.tolist()
        print("Do chinh xac: ", cal_accuracy_all_DT(y_pred_DT, y_test_DT))
        confusion_matrix_DT(y_test_DT, y_pred_DT ,[0, 1])
        print("===========================================")

        print("Naive Bayes")
        model = naive_bayes_classifier(X_train, y_train)
        y_pred_NB = predict_NB(X_test, model)
        cal_accuracy_NB(y_test, y_pred_NB)
        print("===========================================")'''