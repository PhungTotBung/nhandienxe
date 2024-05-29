import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import cv2
import torch
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import os
from tkinter import ttk

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

def select_image():
    # Hiển thị hộp thoại để chọn file
    file_path = filedialog.askopenfilename(title="Chọn hình ảnh")

    # Chuẩn hóa đường dẫn
    normalized_path = os.path.normpath(file_path)

    # Kiểm tra nếu người dùng không chọn file
    if not file_path:
        show_message("Bạn chưa chọn hình ảnh.")
    else:
        # Nhận giá trị ngưỡng từ người dùng
        threshold = get_threshold()
        if threshold is not None:
            # Hiển thị kết quả nhận diện với ngưỡng đã nhập
            show_result(normalized_path, threshold)

def show_result(image_path, threshold):
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

    def obj_detector(img):
        img = cv2.imread(img)
        if img is None:
            show_message("Không thể đọc hình ảnh từ đường dẫn:", img)
            return None, None, None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

        img /= 255.0
        img = torch.from_numpy(img)
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2)

        model.eval()

        img = list(im.to(device) for im in img)
        output = model(img)

        for i, im in enumerate(img):
            boxes = output[i]['boxes'].data.cpu().numpy()
            scores = output[i]['scores'].data.cpu().numpy()
            labels = output[i]['labels'].data.cpu().numpy()

            labels = labels[scores >= threshold]
            boxes = boxes[scores >= threshold].astype(np.int32)
            scores = scores[scores >= threshold]

            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

        sample = img[0].permute(1, 2, 0).cpu().numpy()
        sample = np.array(sample)
        boxes = output[0]['boxes'].data.cpu().numpy()
        name = output[0]['labels'].data.cpu().numpy()
        scores = output[0]['scores'].data.cpu().numpy()
        boxes = boxes[scores >= threshold].astype(np.int32)
        names = name.tolist()

        return names, boxes, sample

    # Hàm xử lý sự kiện chọn vùng
    def onselect(eclick, erelease):
        global selected_image, names, boxes, sample
        # Lấy tọa độ của điểm bắt đầu và kết thúc khi người dùng chọn vùng
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

        # Lưu đường dẫn hình ảnh đã chọn vào biến toàn cục
        selected_image = image_path
        # Gọi hàm phát hiện đối tượng trên hình ảnh đã chọn
        names, boxes, sample = obj_detector(selected_image)

        # Kiểm tra xem có đối tượng nào được phát hiện hay không
        if names is not None:
            for i, box in enumerate(boxes):
                # Kiểm tra xem điểm giữa của hộp nằm trong phạm vi đã chọn hay không
                if x1 < (box[0] + box[2]) / 2 < x2 and y1 < (box[1] + box[3]) / 2 < y2:
                    # Vẽ hộp giới hạn và gán nhãn lên hình ảnh đã chọn
                    cv2.rectangle(sample, (box[0], box[1]), (box[2], box[3]), (0, 220, 0), 2)
                    cv2.putText(sample, str(classes[names[i]]), (box[0], box[1] - 5),
                                cv2.FONT_HERSHEY_COMPLEX, 0.7, (220, 0, 0), 1, cv2.LINE_AA)

            # Hiển thị hình ảnh đã chọn và gán nhãn
            plt.imshow(sample)
            plt.axis('off')
            plt.show()

    # Hiển thị hình ảnh cùng với khả năng chọn vùng
    fig, ax = plt.subplots()
    plt.imshow(plt.imread(image_path))
    plt.axis('off')
    RS = RectangleSelector(ax, onselect, useblit=True, button=[1],
                            minspanx=5, minspany=5, spancoords='pixels', interactive=True)
    
    # Gọi hàm để hiển thị cửa sổ nhập giá trị ngưỡng
    get_threshold()
    
    plt.show()

# Chạy hàm để chọn hình ảnh và hiển thị kết quả
select_image()
