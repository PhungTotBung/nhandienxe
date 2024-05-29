import subprocess
import time
import os
import datetime

def crawl_image():
    try:
        # Nhập tham số đầu vào
        link = input("Nhập link camera: ")
        num = input("Nhập số ảnh: ")
    
        # Kiểm tra xem đường link và số lượng ảnh có hợp lệ hay không
        if not link or not num.isdigit() or int(num) <= 0:
            print("Link hoặc số lượng không hợp lệ!")
            return

        # Gán giá trị của link cho url
        url = link
        count = 0
        download_dir = "image_vehicle_dark_1"

        # Tạo thư mục chứa ảnh nếu chưa tồn tại
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)

        # Vòng lặp cho đến khi số lượng ảnh đã tải đủ
        while count < int(num):
            try:
                now = datetime.datetime.now()
                #formatted_now = str(now).replace(" ", "_").replace(":", "")
                # Tạo tên file ảnh mới dựa trên thời gian và số lượng ảnh đã tải
                image_filename = f"{'20234'}_{count + 1}.jpg"
                image_path = os.path.join(download_dir, image_filename)
                # Tạo lệnh curl để tải ảnh về
                curl_command = f'curl -o "{image_path}" "{url}"'

                # Thực thi lệnh curl và chờ 12 giây trước khi tiếp tục
                subprocess.run(curl_command, shell=True, check=True)
                time.sleep(12)

                # In thông báo về việc tải ảnh thành công
                print(f"Ảnh {image_filename} đã được tải về và lưu vào {image_path}.")
                print(f"Số lượng tải: {count + 1}/{num}")
                count += 1
            except subprocess.CalledProcessError as e:
                # Xử lý lỗi khi thực thi lệnh curl
                print(f"Lỗi khi thực thi lệnh curl: {e}")
            except Exception as e:
                # Xử lý lỗi không xác định
                print(f"Lỗi không xác định: {e}")

        # In thông báo khi tải ảnh thành công
        print("Thành công tải dữ liệu!")
    except Exception as e:
        # Xử lý lỗi chung không xác định
        print(f"Lỗi không xác định: {e}")

# Kiểm tra xem đoạn mã được thực thi trực tiếp hay được gọi từ một tệp khác
if __name__ == "__main__":
    # Gọi hàm download_images nếu đoạn mã được thực thi trực tiếp
    crawl_image()
