import os
import cv2
import torch
import numpy as np
import tkinter as tk
from tkinter import Tk, filedialog, simpledialog
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt
from tkinter import ttk

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

def show_message(message, title="Thông báo"):
    # Hiển thị hộp thoại thông báo
    tk.messagebox.showinfo(title, message)

def obj_detector(img, detection_threshold):
    # Hàm để thực hiện phát hiện đối tượng trên một hình ảnh
    img = cv2.imread(img)
    if img is None:
        show_message("Không thể đọc hình ảnh từ đường dẫn:", img)
        raise ValueError("Không thể đọc hình ảnh từ đường dẫn:", img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img /= 255.0
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)
    img = img.permute(0, 3, 1, 2)

    model.eval()

    img = [image.to(device) for image in img]
    output = model(img)

    for i, image in enumerate(img):
        boxes = output[i]['boxes'].detach().cpu().numpy()
        scores = output[i]['scores'].detach().cpu().numpy()
        labels = output[i]['labels'].detach().cpu().numpy()

        labels = labels[scores >= detection_threshold]
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        scores = scores[scores >= detection_threshold]

        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

    sample = img[0].permute(1, 2, 0).cpu().numpy()
    sample = np.array(sample)
    boxes = output[0]['boxes'].detach().cpu().numpy()
    name = output[0]['labels'].detach().cpu().numpy()
    scores = output[0]['scores'].detach().cpu().numpy()
    boxes = boxes[scores >= detection_threshold].astype(np.int32)
    names = name.tolist()

    return names, boxes, sample

# Hiển thị hộp thoại để chọn thư mục
pred_path = filedialog.askdirectory(title="Chọn thư mục chứa hình ảnh")

# Kiểm tra nếu người dùng không chọn thư mục
if not pred_path:
    show_message("Bạn chưa chọn thư mục.")
else:
    # Tải mô hình 
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

    # Nhận giá trị ngưỡng từ người dùng
    detection_threshold = get_threshold()

    if detection_threshold is not None:
        # Lặp qua tất cả các hình ảnh trong thư mục
        pred_files = [os.path.join(pred_path, f) for f in os.listdir(pred_path)]
        num_images = min(20, len(pred_files))  # Giới hạn số lượng hình ảnh là 20

        if num_images > 0:
            num_columns = int(np.ceil(np.sqrt(num_images)))  # Tính toán số lượng cột

            # Đảm bảo số lượng dòng là một số nguyên dương
            num_rows = int(np.ceil(num_images / num_columns))

            plt.figure(figsize=(20, 20))
            for i, images in enumerate(pred_files[:num_images]):
                plt.subplot(num_rows, num_columns, i + 1)
                try:
                    names, boxes, sample = obj_detector(images, detection_threshold)
                except ValueError as e:
                    show_message(str(e))
                    break

                if all(arr is not None for arr in (names, boxes, sample)):  # Kiểm tra xem tất cả các phần tử trong tuple có khác None không
                    for i, box in enumerate(boxes):
                        cv2.rectangle(sample, (box[0], box[1]), (box[2], box[3]), (0, 220, 0), 2)
                        # Thay đổi đoạn này để hiển thị tên class thay vì chỉ số class
                        cv2.putText(sample, str(classes[names[i]]), (box[0], box[1] - 5), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                                    (220, 0, 0), 1, cv2.LINE_AA)

                    plt.axis('off')
                    plt.imshow(sample)

            plt.show()
