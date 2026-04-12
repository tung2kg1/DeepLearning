<div align="center">

| ![CIFAR-10](https://github.com/tung2kg1/DeepLearning/blob/main/app/dog_cfar10.png?raw=true) | ![Plant Disease](https://github.com/tung2kg1/DeepLearning/blob/main/app/leafscorch_plantvillage.png?raw=true) | ![Cat vs Dog](https://github.com/tung2kg1/DeepLearning/blob/main/app/cat_catvsdog.png?raw=true) |
| :---: | :---: | :---: |
| *Phân loại CIFAR-10* | *Nhận diện Bệnh cây trồng* | *Phân biệt Chó vs Mèo* |

| ![Default UI](https://github.com/tung2kg1/DeepLearning/blob/main/app/defult_ui.png?raw=true) | ![Model Selector](https://github.com/tung2kg1/DeepLearning/blob/main/app/model_selector.png?raw=true) |
| :---: | :---: |
| *Giao diện ứng dụng chính* | *Menu lựa chọn mô hình* |

</div>

Plant Disease: Nhận diện 38 loại bệnh cây trồng (Dữ liệu PlantVillage).

Cat vs Dog:    Phân biệt chó và mèo.

CIFAR-10:      Phân loại 10 lớp đối tượng cơ bản (Máy bay, ô tô, chim, ...).

Tiền xử lý (Pre-processing):
   - Auto-size Detection: App tự động truy vấn 'model.input_shape' để
     thực hiện Resize ảnh (Ví dụ: 224x224 hoặc 32x32) một cách chính xác.
   - Normalization: Chuyển đổi giá trị Pixel về đoạn [0, 1].
   - Dimension Expansion: Chuyển ảnh về dạng Batch (1, H, W, C).

Inference:
   - Sử dụng TensorFlow để dự đoán xác suất (Softmax Vector).

Post-processing:
   - Mapping: Ánh xạ Index sang tên lớp (Class Name).
   - Prettify: Định dạng lại văn bản (ví dụ: "Apple___Black_rot" -> "Apple - Black Rot").
