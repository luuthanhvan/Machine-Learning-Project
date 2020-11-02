''' 
Tính chỉ số Gini
- groups: mảng chứa các nhóm giá trị của 1 thuộc tính
- classes: mảng chứa các giá trị nhãn
ví dụ ở tập dữ liệu weather: thuộc tính outlook có 3 nhóm giá trị là sunny, overcast và rain và cột nhãn có 2 giá trị là Yes và No
groups = [sunny, outlook, rain]
classes = [yes, no]
'''
def giniIndex(groups, classes):
    # Tổng số lượng phần tử (instance) của 1 thuộc tính
    nbInstances = float(sum([len(group) for group in groups])) # tương ứng với N
    
    # Dòng 11 được viết lại như sau:
    # nbInstances = 0.0
    # for group in groups:
    #     nbInstances += len(group)
    
    gini_split = 0.0
    for group in groups:       
        size = float(len(group))
        if size == 0: continue

        sum_p = 0.0
        for classVal in classes:
            p = [row[-1] for row in group].count(classVal)/size
            # Dòng số 25 được viết lại như sau:
            # cnt = 0
            # for row in group:
            #     if(row[-1] == classVal):
            #         cnt += 1
            # p = cnt/size
            sum_p += p*p
        
        gini_T = 1.0 - sum_p
        gini_split += (size/nbInstances)*gini_T

    return gini_split

# test Gini values for Outlook attribute
print("gini_split(Outlook) =", giniIndex(
[
    [ # sunny
        ["sunny", "no"], 
        ["sunny", "no"],
        ["sunny", "no"],
        ["sunny", "yes"],
        ["sunny", "yes"],
    ],
    [ # overcast
        ["overcast", "yes"],
        ["overcast", "yes"],
        ["overcast", "yes"],
        ["overcast", "yes"],
    ],
    [ # rain
        ["rain", "yes"],
        ["rain", "yes"],
        ["rain", "yes"],
        ["rain", "no"],
        ["rain", "no"],
    ]
], ["yes", "no"])) # RESULT: gini_split(Outlook) = 0.34285714285714286 ~ 0.343


# test Gini values for Temperature attribute
print("gini_split(Temperature) =", giniIndex(
[
    [ # hot
        ["hot", "yes"], 
        ["hot", "yes"],
        ["hot", "no"],
        ["hot", "no"],
    ],
    [ # mild
        ["mild", "yes"],
        ["mild", "yes"],
        ["mild", "yes"],
        ["mild", "yes"],
        ["mild", "no"],
        ["mild", "no"],
    ],
    [ # cool
        ["cool", "yes"],
        ["cool", "yes"],
        ["cool", "yes"],
        ["cool", "no"],
    ]
], ["yes", "no"])) # RESULT: gini_split(Temperature) = 0.44047619047619047 ~ 0.44

# test Gini values for Humidity attribute
print("gini_split(Humidity) =", giniIndex(
[
    [ # high
        ["high", "yes"], 
        ["high", "yes"],
        ["high", "yes"],
        ["high", "no"],
        ["high", "no"],
        ["high", "no"],
        ["high", "no"],
    ],
    [ # normal
        ["normal", "yes"],
        ["normal", "yes"],
        ["normal", "yes"],
        ["normal", "yes"],
        ["normal", "yes"],
        ["normal", "yes"],
        ["normal", "no"],
    ],
], ["yes", "no"])) # RESULT: gini_split(Humidity) = 0.3673469387755103 ~ 0.367

# test Gini values for Windy attribute
print("gini_split(Windy) =", giniIndex(
[
    [ # true
        ["true", "yes"], 
        ["true", "yes"],
        ["true", "yes"],
        ["true", "no"],
        ["true", "no"],
        ["true", "no"],
    ],
    [ # false
        ["false", "yes"],
        ["false", "yes"],
        ["false", "yes"],
        ["false", "yes"],
        ["false", "yes"],
        ["false", "yes"],
        ["false", "no"],
        ["false", "no"],
    ],
], ["yes", "no"])) # RESULT: gini_split(Windy) = 0.42857142857142855 ~ 0.429