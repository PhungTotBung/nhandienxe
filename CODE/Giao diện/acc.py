import tkinter as tk
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

# Gọi hàm để thử nghiệm
selected_threshold = get_threshold()
print("Giá trị ngưỡng được chọn:", selected_threshold)
