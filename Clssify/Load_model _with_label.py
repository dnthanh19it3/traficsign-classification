# import the necessary packages
from preprocessing import imagetoarraypreprocessor
from preprocessing import simplepreprocessor
from datasets import simpledatasetloader
from keras.models import load_model
from imutils import paths
import numpy as np
import cv2
import argparse

#  Load_model.py -d <folder chứa ảnh phân loại>
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Nhập vào folder để phân loại")
args = vars(ap.parse_args())

# Khởi tạo danh sách nhãn
classLabels = ['Toc do toi da cho phep 5 km/h', 'Toc do toi da cho phep 15 km/h', 'Cam di thang', 'Cam re trai',
               'Cam re trai va re phai', 'Cam re phai', 'Cam vuot', 'Cam quay dau xe','Cam xe o to', 'Cam bop coi', 'Het han che toc do toi da 40km/h', 'Het han che toc do toi da 50km/h',
               'Toc do toi da cho phep 30 km/h', 'Di thang va re phai', 'Di thang',
               'Re trai', 'Re trai va re phai', 'Re phai', 'Huong phai di vong chuong ngai vat phai',
               'Huong phai di vong chuong ngai vat phai',
               'Noi giao nhau chay theo vong xuyen', 'Duong gioi han', 'An coi', 'Toc do toi da cho phep 40 km/h',
               'Chi danh cho xe dap',
               'Duoc phep quay dau xe', 'Re trai va/hoac phai de di duong vong', 'Giao nhau co tin hieu den',
               'Nguy hiem khac', 'Duong danh cho nguoi di bo phia truoc',
               'Co xe dap phia truoc', 'Truong hoc o phia truoc', 'Duong cong ben phai', 'Duong cong ben trai',
               'Toc do toi da cho phep 50 km/h', 'Xuong doc',
               'Len doc', 'Cham lai', 'Nga ba duong phu phia truoc ben trai', 'Nga ba duong phu phia truoc ben phai',
               'Duong xuyen lang',
               'Duong cong doi, re trai truoc, sau do re phai', 'Giao nhau voi duong sat khong co rao chan',
               'Phia truoc dang thi cong', 'Nhieu duong cong',
               'Toc do toi da cho phep 60 km/h', 'Giao nhau voi duong sat co rao chan', 'Doan duong hay xay ra tai nan',
               'Dung', 'Duong cam', 'Khong dung lai',
               'Khong co loi vao cho giao thong xe co', 'Cho di', 'Dieu khien', 'Toc do toi da cho phep 70 km/h',
               'Toc do toi da cho phep 70 km/h', 'Cam di thang va re trai', 'Cam di thang va re phai']

# Lấy danh sách các hình ảnh trong tập dữ liệu sau đó lấy mẫu ngẫu nhiên
# ảnh theo chỉ số để đưa vào đường dẫn hình ảnh
print("[INFO] Đang nạp ảnh mẫu để phân lớp (dự đoán)...")
imagePaths = np.array(list(paths.list_images(args["dataset"])))  # xác định số file trong dataset
idxs = range(0, len(imagePaths))  # Lấy tất cả các chỉ số idxs của ảnh
imagePaths = imagePaths[idxs]

sp = simplepreprocessor.SimplePreprocessor(32, 32)  # Thiết lập kích thước ảnh 32 x 32
iap = imagetoarraypreprocessor.ImageToArrayPreprocessor()  # Gọi hàm để chuyển ảnh sang mảng

# Nạp dataset từ đĩa rồi co dãn mức xám của pixel trong vùng [0,1]
sdl = simpledatasetloader.SimpleDatasetLoader(preprocessors=[sp, iap])
print("Paths", imagePaths)
(data, labels) = sdl.load(imagePaths)
data = data.astype("float") / 255.0

# Nạp model đã pre-trained
print("[INFO] Nạp model mạng pre-trained ...")
model = load_model("miniVGGNet.hdf5")

# Dự đoán
print("[INFO] Đang dự đoán để phân lớp...")
print(data)
preds = model.predict(data, batch_size=32).argmax(axis=1)

# Lặp qua tất cả các file ảnh trong imagePaths
# Nạp ảnh ví dụ --> Vẽ dự đoán --> Hiển thị ảnh
for (i, imagePath) in enumerate(imagePaths):
    try:
        image = cv2.imread(imagePath)
        cv2.putText(image, "label: {}".format(classLabels[preds[i]]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)
        cv2.imshow("Image", image)
        cv2.waitKey(0)
    except:
        print(preds[i], classLabels)
