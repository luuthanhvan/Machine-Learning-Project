import csv
import random
import pandas as pd
import numpy as np
import array as arr
from pprint import pprint

'''
Các bước và các hàm cần định nghĩa khi xây dựng Cây quyết định:
1. Đọc dữ liệu (tiền xử lý nếu cần):
- Input: tên file hoặc đường dẫn đến file .csv
- Output: trả về tập dữ liệu
def read_file():
	# đọc file csv từ thư viện pandas <- dataset
    return dataset
'''
def readFile(fileName):
    dataset = pd.read_csv(fileName)
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
        test_size = round(test_size * len(dataset)) # để đảm bảo test_size luôn là 1 số nguyên

    indices = dataset.index.tolist() # lấy tất cả các chỉ số trong tập DL gốc, sau đó chuyển nó sang dạng list
    test_indices = random.sample(population = indices, k = test_size) # random các chỉ số cho tập test, lưu vào mảng test_indices
    # print(test_indices)
    test_data = dataset.iloc[test_indices] # lấy các giá trị và lưu vào tập DL test thông qua các chỉ số trong mảng test_indices
    train_data = dataset.drop(test_indices) # lấy các giá trị bỏ đi các chỉ số trong mảng test_indices và lưu vào tập DL train

    return train_data, test_data

'''3. Xây dựng cây
def decision_tree_algorithm(data_train, counter, min_samples_leaf, max_depth):
    return sub_tree

sub_tree = {"question": ["yes_answer", 
                         "no_answer"]}
                         
example_tree = {'petal_width <= 0.8': ['Iris-setosa',
                        {'petal_width <= 1.65': [{'petal_length <= 4.95': ['Iris-versicolor',
                                                                           'Iris-virginica']},
                                                 'Iris-virginica']}]}
'''

# hàm kiểm tra dữ liệu trong 1 nút có thuần nhấy hay không
# một nút được xem là có DL thuần nhất khi nút đó chỉ chứa duy nhất 1 nhãn
def check_purity(data):
    label_column = data[:, -1] # lấy nguyên cột nhãn trong data
    #print(label_column)
    
    classes = np.unique(label_column) # lấy giá trị các nhãn duy nhất từ cột nhãn
    #print(classes)
    # Ví dụ: đối với tập dữ liệu hoa Iris, classes = ['Iris-setosa' 'Iris-versicolor' 'Iris-virginica']
     
    # kiểm tra dữ liệu trong mảng classes, 
    # nếu số lượng phần tử trong mảng classes chỉ có 1 nhãn -> thuần nhất -> là nút lá
    # ngược lại không phải là nút lá 
    
    #print(len(classes))
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
        classes, counts_classes = np.unique(label_column, return_counts = True) # lấy giá trị các nhãn duy nhất từ cột nhãn cùng với số lượng nhãn tương ứng
        # print(counts_classes)
        # tìm phần tử có số lượng nhãn lớn nhất có trong mảng counts_classes
        maxValue = counts_classes[0]
        index = 0
        for i in range(len(counts_classes)):
            if(maxValue < counts_classes[i]):
                maxValue = counts_classes[i] 
                index = i # lưu lại vị trí
        leaf_node = classes[index] # lấy nhãn tương ứng với vị trí vừa tìm được ở bên trên
    
    return leaf_node

def get_point_splits(data):
    point_splits = {} # khởi tạo 1 từ điển rỗng
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
    
    Trong hàm này, potential_splits có dạng {col_index: []}
    - col_index: chỉ số cột
    - []: mình sẽ xác định tất cả các điểm phân hoạch trong một cột = (giá trị hiện tại + giá trị trước đó)/2 và lưu vào mảng này
    '''
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

# Phân hoạch nhị phân dựa trên 1 giá trị ngưỡng cho 1 thuộc tính (cột) trên tập dữ liệu
def binary_split_data(data, split_column, split_value):
    split_column_values = data[:, split_column]
    # print(split_column_value)
    left = data[split_column_values <= split_value]
    right = data[split_column_values > split_value]
    
    return left, right

# 1. Sử dụng độ lợi thông tin, chọn thuộc tính phân hoạch có giá trị lớn nhất
# Tính độ hỗn loạn thông tin trước khi phân hoạch (entropy)
def info(data):
    label_column = data[:, -1]
    classes, counts_classes = np.unique(label_column, return_counts = True)

    entropy = 0.0
    for i in range(len(counts_classes)):
        # Tính xác suất xuất hiện cho từng phân lớp <- lưu vào biến p
        p = counts_classes[i]/sum(counts_classes)
        # Ta có công thức: Info(D) =  entropy(p1, p2,..., pn) = (-p1*log2(p1)) + (-p2*log2(p2)) + ... + (-pn*log2(pn))
        # Vì thế ta cần cộng dồn các giá trị [-p*log2(p)] vào biến entropy
        entropy += (-p * np.log2(p))
        
    return entropy

# Tính độ hỗn loạn thông tin sau khi phân hoạch (Info_A)
def info_A(left, right):
    # Ta có công thức: Info_A(D) = (D1/D)*Info(D1) + (D2/D)*Info(D2) + ... + (Dv/D)*Info(Dv)
    # Ta phân hoạch DL ra thành 2 phần: left và right
    # Do đó ta sẽ tính Info(left) và Info(right), sau đó cộng 2 kết quả này <- overall_entropy
    D = len(left) + len(right) # tính tổng số lượng phần tử trong tập dữ liệu D
    D1 = len(left)
    D2 = len(right)
    
    overall_entropy = ((D1 / D) * info(left)) + ((D2 / D) * info(right))
    
    return overall_entropy

# Hàm chọn ra thuộc tính và giá trị của thuộc tính đó để phân hoạch dựa vào giá trị độ lợi thông tin lớn nhất
def choose_best_split(data, point_splits):
    information_gain = -9999999
    for col_index in point_splits:
        for value in point_splits[col_index]:
            left, right = binary_split_data(data, split_column = col_index, split_value = value)
            current_information_gain = info(data) - info_A(left, right)

            if current_information_gain > information_gain:
                information_gain = current_information_gain
                best_split_column = col_index
                best_split_value = value

    return best_split_column, best_split_value

def decision_tree_algorithm(df, counter=0, min_samples=2, max_depth=5):    
    # nút gốc
    if counter == 0:
        global COLUMN_HEADERS
        COLUMN_HEADERS = df.columns
        data = df.values
    else:
        data = df          

    # trường hợp để dừng phân hoạch
    if (check_purity(data)) or (len(data) < min_samples) or (counter == max_depth):
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
        
        # tạo cây con
        feature_name = COLUMN_HEADERS[split_column] # tên thuộc tính
        question = "{} <= {}".format(feature_name, split_value) # điều kiện
        sub_tree = {question: []}
        
        # lặp lại việc phân hoạch 1 cách đệ quy cho cây con trái và phải
        yes_answer = decision_tree_algorithm(left, counter, min_samples, max_depth)
        no_answer = decision_tree_algorithm(right, counter, min_samples, max_depth)
        
        if yes_answer == no_answer:
            sub_tree = yes_answer
        else:
            sub_tree[question].append(yes_answer)
            sub_tree[question].append(no_answer)
        
        return sub_tree

'''
4. Dự đoán nhãn cho tập dữ liệu kiểm tra
def predict(tree, data_test):
    return y_pred

5. Tính toán độ chính xác tổng thể
def cal_accuracy_all(y_pred, test_data):
    return accuracy_score

'''

dataset = readFile("../data_set/iris_data.csv")
# print(dataset)
random.seed(0)
train_data, test_data = train_test_split(dataset, test_size=0.1)
# print(train_data)
# print(test_data)

# Test hàm check_purity(data)
#print(check_purity(train_data.values)) # kết quả là False, bởi vì giá trị nhãn trong tập DL train bao gồm 3 nhãn -> chưa thuần nhất
#dt = train_data.head().values # lấy 5 giá trị đầu tiên trong tập DL train
#print(dt)
'''
   sepalLength  sepalWidth  petalLength  petalWidth      species
0          5.1         3.5          1.4         0.2  Iris-setosa
1          4.9         3.0          1.4         0.2  Iris-setosa
2          4.7         3.2          1.3         0.2  Iris-setosa
3          4.6         3.1          1.5         0.2  Iris-setosa
4          5.0         3.6          1.4         0.2  Iris-setosa
'''
#print(check_purity(dt)) 
# kết quả là True, bởi vì 5 phần tử đầu trong tập DL huấn luyện đều có cùng 1 nhãn là Iris-setosa(check_purity(dt))


# Test hàm create_leaf_node()
# leaf_node = create_leaf_node(train_data.values)
# print(leaf_node)


# Test hàm get_point_splits(data)
#point_splits = get_point_splits(train_data.values)
#print(point_splits)
''' Result: point_splits = {key: value, } <=> {col_index: [], }
{
    0: [4.35, 4.45, 4.55, 4.65, 4.75, 4.85, 4.95, 5.05, 5.15, 5.25, 5.35, 5.45, 5.55, 5.65, 5.75, 5.85, 5.95, 6.05, 6.15, 6.25, 6.35, 6.45, 6.55, 6.65, 6.75, 6.85, 6.95, 7.05, 7.15, 7.4, 7.65, 7.800000000000001], 
    1: [2.1, 2.25, 2.3499999999999996, 2.45, 2.55, 2.6500000000000004, 2.75, 2.8499999999999996, 2.95, 3.05, 3.1500000000000004, 3.25, 3.3499999999999996, 3.45, 3.55, 3.6500000000000004, 3.75, 3.8499999999999996, 3.95, 4.05, 4.15, 4.300000000000001], 
    2: [1.05, 1.15, 1.25, 1.35, 1.45, 1.55, 1.65, 1.7999999999999998, 2.5999999999999996, 3.4, 3.55, 3.6500000000000004, 3.75, 3.8499999999999996, 3.95, 4.05, 4.15, 4.25, 4.35, 4.45, 4.55, 4.65, 4.75, 4.85, 4.95, 5.05, 5.15, 5.25, 5.35, 5.45, 5.55, 5.65, 5.75, 5.85, 5.95, 6.05, 6.25, 6.5, 6.65, 6.800000000000001], 
    3: [0.15000000000000002, 0.25, 0.35, 0.45, 0.55, 0.8, 1.05, 1.15, 1.25, 1.35, 1.45, 1.55, 1.65, 1.75, 1.85, 1.95, 2.05, 2.1500000000000004, 2.25, 2.3499999999999996, 2.45]
}
'''

tree = decision_tree_algorithm(train_data)
pprint(tree)