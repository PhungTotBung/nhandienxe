# Nhập các thư viện cần thiết
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import cv2
import torch
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt
import os
from tkinter import ttk

# Hàm hiển thị hộp thoại thông báo
def show_message(message, title="Thông báo"):
    tk.messagebox.showinfo(title, message)

# Hàm để nhận giá trị ngưỡng từ người dùng
def get_threshold():
    # Tạo cửa sổ Tkinter để chọn giá trị ngưỡng
    root = tk.Tk()
    root.title("Chọn giá trị ngưỡng")

    # Lấy kích thước màn hình và cửa sổ
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    window_width = 300  # Đặt kích thước cửa sổ theo ý muốn
    window_height = 75

    # Tính toán vị trí để cửa sổ nằm giữa màn hình
    x_position = (screen_width - window_width) // 2
    y_position = (screen_height - window_height) // 2

    # Đặt vị trí cửa sổ
    root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")

    # Biến để lưu giá trị ngưỡng được chọn
    threshold_var = tk.DoubleVar()

    # Hàm được gọi khi nút OK được nhấn
    def on_okay():
        selected_threshold = threshold_var.get()
        root.destroy()

    # Các giá trị ngưỡng có thể chọn
    values = [0.50, 0.75, 0.90]

    # Tạo frame để chứa các radiobutton và sắp xếp chúng ngang
    frame = ttk.Frame(root)
    frame.pack(pady=5)

    # Tạo style để cài đặt font cho Radiobutton
    style = ttk.Style()
    style.configure("TRadiobutton", font=('Arial', 10))

    # Thêm các radiobutton vào frame
    for value in values:
        radio_button = ttk.Radiobutton(frame, text=str(value), variable=threshold_var, value=value, style="TRadiobutton")
        radio_button.pack(side=tk.LEFT, padx=5)

    okay_button = ttk.Button(root, text="OK", command=on_okay)
    okay_button.pack(pady=10)

    # Chờ cho đến khi cửa sổ được đóng
    root.wait_window()

    # Trả về giá trị ngưỡng được chọn
    return threshold_var.get()

# Hàm để chọn tệp hình ảnh sử dụng hộp thoại tệp
def select_image():
    # Hiển thị hộp thoại để chọn tệp hình ảnh
    file_path = filedialog.askopenfilename(title="Chọn hình ảnh")

    # Chuẩn hóa đường dẫn tệp
    normalized_path = os.path.normpath(file_path)

    # Kiểm tra nếu người dùng không chọn tệp
    if not file_path:
        show_message("Bạn chưa chọn hình ảnh.")
    else:
        # Nhận giá trị ngưỡng từ người dùng
        threshold = get_threshold()
        if threshold is not None:
            # Hiển thị kết quả nhận diện với ngưỡng đã chọn
            show_result(normalized_path, threshold)

# Hàm để hiển thị kết quả nhận diện cho một hình ảnh và ngưỡng đã chọn
def show_result(image_path, threshold):
    # Tải một mô hình đã được huấn luyện trước trên COCO
    model = fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
    WEIGHTS_FILE = "model.pth"
    num_classes = 10
    classes = {1: 'bus', 2: 'car', 6: 'truck', 4: 'motorbike'}

    # Nhận số lượng đặc trưng đầu vào cho bộ phân loại
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Thay thế phần đầu được huấn luyện trước bằng một phần đầu mới
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Tải trọng số đã được huấn luyện
    model.load_state_dict(torch.load(WEIGHTS_FILE, map_location=torch.device('cpu')))
    
    # Xác định thiết bị sẽ sử dụng (GPU hoặc CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Hàm thực hiện việc phát hiện đối tượng trên một hình ảnh
    def obj_detector(img):
        img = cv2.imread(img)
        if img is None:
            show_message(f"Không thể đọc hình ảnh từ đường dẫn: {img}")
            return None, None, None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

        img /= 255.0
        img = torch.from_numpy(img)
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2)

        model.eval()

        img = list(im.to(device) for im in img)
        output = model(img)

        # Lặp qua từng hình ảnh đầu ra để xử lý kết quả
        for i, im in enumerate(img):
            # Lấy thông tin về hộp giới hạn (boxes), điểm (scores), và nhãn (labels)
            boxes = output[i]['boxes'].data.cpu().numpy()
            scores = output[i]['scores'].data.cpu().numpy()
            labels = output[i]['labels'].data.cpu().numpy()
            # Lọc các đối tượng dự đoán với xác suất lớn hơn ngưỡng đã chỉ định
            labels = labels[scores >= threshold]
            boxes = boxes[scores >= threshold].astype(np.int32)
            scores = scores[scores >= threshold]
            # Chuyển đổi hộp giới hạn sang định dạng (x, y, width, height)
            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
        # Lấy hình ảnh mẫu (sample) và thông tin về hộp giới hạn, nhãn và điểm của đối tượng
        sample = img[0].permute(1, 2, 0).cpu().numpy()
        sample = np.array(sample)
        boxes = output[0]['boxes'].data.cpu().numpy()
        name = output[0]['labels'].data.cpu().numpy()
        scores = output[0]['scores'].data.cpu().numpy()
        boxes = boxes[scores >= threshold].astype(np.int32)
        names = name.tolist()

        return names, boxes, sample

    # Chọn hình ảnh để hiển thị
    selected_image = image_path

    # Gọi hàm obj_detector với ngưỡng đã chọn
    names, boxes, sample = obj_detector(selected_image)

    if names is not None:
        for i, box in enumerate(boxes):
            cv2.rectangle(sample, (box[0], box[1]), (box[2], box[3]), (0, 220, 0), 2)
            if names[i] in classes:
                cv2.putText(sample, str(classes[names[i]]), (box[0], box[1]-5), cv2.FONT_HERSHEY_COMPLEX, 0.7, (220, 0, 0), 1, cv2.LINE_AA)
            else:
                print(f"Không tìm thấy tên lớp cho khóa: {names[i]}")
        plt.imshow(sample)
        plt.axis('off')
        plt.show()

# Chạy hàm để chọn hình ảnh và hiển thị kết quả
select_image()
