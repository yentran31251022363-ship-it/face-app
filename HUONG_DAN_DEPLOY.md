# 🚀 Deploy lên Streamlit Cloud (miễn phí, vĩnh viễn)

---

## Bước 1 — Chuẩn bị GitHub repo

### 1a. Tạo tài khoản GitHub (nếu chưa có)
→ https://github.com/signup

### 1b. Tạo repo mới
1. Vào https://github.com/new
2. Đặt tên repo, ví dụ: `face-recognition-app`
3. Để **Public** (bắt buộc cho Streamlit Cloud free)
4. Bấm **Create repository**

### 1c. Upload file lên repo
Bấm **Add file → Upload files**, upload 3 file:
```
app.py
requirements.txt
face_recognition_model.h5   ← file model của bạn
```

> ⚠️ File .h5 ~9MB → GitHub cho phép tối đa 100MB/file, không vấn đề gì.

---

## Bước 2 — Tạo tài khoản Streamlit Cloud

1. Vào https://share.streamlit.io
2. Bấm **Sign up** → chọn **Continue with GitHub**
3. Authorize cho Streamlit truy cập GitHub

---

## Bước 3 — Deploy app

1. Vào https://share.streamlit.io → bấm **New app**
2. Điền thông tin:
   - **Repository**: `your-username/face-recognition-app`
   - **Branch**: `main`
   - **Main file path**: `app.py`
3. Bấm **Deploy!**

Streamlit Cloud sẽ tự động:
- Cài packages từ `requirements.txt`
- Chạy `app.py`
- Tạo URL dạng: `https://your-username-face-recognition-app-app-xxxx.streamlit.app`

Chờ khoảng **3–7 phút** là xong.

---

## Bước 4 — SỬA TÊN CLASS (QUAN TRỌNG!)

Mở `app.py` trên GitHub, tìm:
```python
CLASS_NAMES = [
    "Person_01", "Person_02", ...
]
```
Thay bằng **tên thật 22 người**, **đúng thứ tự** khi train model.

Ví dụ:
```python
CLASS_NAMES = [
    "Nguyen_Van_A",   # class index 0
    "Tran_Thi_B",     # class index 1
    "Le_Van_C",       # class index 2
    ...
]
```

Sau khi save trên GitHub, Streamlit Cloud tự động redeploy.

---

## Sau khi deploy

| | Chi tiết |
|---|---|
| **URL** | Vĩnh viễn, chia sẻ được ngay |
| **Miễn phí** | 1 app / tài khoản (free tier) |
| **Sleep** | App sleep sau 7 ngày không dùng → mở lại tự boot (~30s) |
| **Wake up** | Vào Streamlit Cloud dashboard → bấm **Wake app** |

---

## Tính năng của app

| Tab | Chức năng |
|-----|-----------|
| 📷 Upload Ảnh | Upload jpg/png → detect khuôn mặt → hiển thị bounding box + tên + confidence |
| 📹 Webcam | Chụp ảnh từ camera → nhận diện tương tự |

---

## Troubleshoot thường gặp

| Lỗi | Nguyên nhân | Fix |
|-----|------------|-----|
| `ModuleNotFoundError: tensorflow` | Version conflict | Đổi thành `tensorflow-cpu==2.15.0` trong requirements.txt |
| Kết quả nhận diện sai | CLASS_NAMES sai thứ tự | Kiểm tra lại thứ tự label_encoder khi train |
| App crash khi load model | Version TF khác lúc train | Đổi version TF trong requirements.txt khớp với lúc train |
| File .h5 không load được | Path sai | Đảm bảo file .h5 nằm cùng thư mục với app.py trong repo |
