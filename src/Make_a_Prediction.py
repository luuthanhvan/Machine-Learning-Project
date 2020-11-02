# Dữ liệu vào của hàm predict gồm 2 tham số: node và row
#   +Node: nút cần xét của cây
#   +Row: số thứ tự hàng cần dự đoán trong tập dữ liệu.
#        Tức là mỗi lần hàm predict được gọi nó chỉ dự đoán cho một dòng thôi.
#        Để dự đoán cho một tập dữ liệu test để tính độ chính xác ta cần sử dụng vòng lặp "for row in test:"
def predict(node, row):
	if row[node['index']] < node['value']:
        # Dòng 7, kiểm tra xem nên duyệt con bên trái hay con bên phải của nút
		if isinstance(node['left'], dict):
            #Dòng 9 isinstance: kiểm tra nút con bên trái có phải là kiểu dict (từ điển) không
            #   Tức là kiểm tra xem nút con bên trái đó có phải là nút lá hay chưa.
            #   Nếu chưa, thì gọi đệ quy đến khi là nút là thì đưa ra kết quả dụ đoán
			return predict(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], row)
		else:
			return node['right']
