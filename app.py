import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

# ── Cấu hình trang ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Face Recognition",
    page_icon="🎭",
    layout="wide",
)

# ── Cấu hình model ─────────────────────────────────────────────────────────────
MODEL_PATH  = "face_recognition_model.h5"
IMG_SIZE    = (180, 180)
THRESHOLD   = 0.60   # confidence tối thiểu

# ⚠️ SỬA LẠI tên 22 người đúng thứ tự label khi train!
CLASS_NAMES =[
    "Đinh Hữu Khánh Anh", "Đoàn Hùng", "Đỗ An Phúc", "HoangKyAnh", "Lê Quang Dũng",
    "Lê Tuấn Thành", "Lương Ngọc Thuận", "Ngô Quốc Trung", "Nguyen Ngọc Bao", "Nguyễn Đặng Vinh Phúc",
    "Nguyễn Hoàng Quế Châu", "Nguyễn Phạm Hoàng An", "Nguyễn Thị Khánh Lê", "Nguyễn Thị Ngọc Tuyết", "Nguyễn Tiến Mạnh",
    "Nguyễn Việt Đức", "Phạm Gia Thành Duy", "Phạm Hứa Nhật Minh", "Phạm Nguyễn Bảo Châu", "Phạm Phú Hoà",
    "Trần Hải Yến", "Vũ Quang Thái"
]

# ── Load model (cache) ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Đang load model...")
def load_model():
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    return tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"preprocess_input": preprocess_input},
    )
 
model = load_model()
 
# Download haar cascade nếu chưa có (Streamlit Cloud không có sẵn)
import urllib.request, os
 
CASCADE_PATH = "haarcascade_frontalface_default.xml"
if not os.path.exists(CASCADE_PATH):
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml",
        CASCADE_PATH,
    )
 
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
# ── Helpers ────────────────────────────────────────────────────────────────────
def preprocess(img_rgb: np.ndarray) -> np.ndarray:
    img = Image.fromarray(img_rgb).resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def predict(img_rgb: np.ndarray):
    preds = model.predict(preprocess(img_rgb), verbose=0)[0]
    idx   = int(np.argmax(preds))
    conf  = float(preds[idx])
    label = CLASS_NAMES[idx] if conf >= THRESHOLD else "Unknown"
    return label, conf, preds

def detect_and_annotate(img_rgb: np.ndarray):
    """Detect khuôn mặt, nhận diện từng khuôn mặt, vẽ bounding box."""
    img_bgr  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    gray     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces    = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
    results  = []

    if len(faces) == 0:
        # Không detect được → predict toàn ảnh
        label, conf, preds = predict(img_rgb)
        results.append({"label": label, "conf": conf, "preds": preds, "face": None})
        return img_rgb, results

    for (x, y, w, h) in faces:
        crop        = img_rgb[y:y+h, x:x+w]
        label, conf, preds = predict(crop)
        results.append({"label": label, "conf": conf, "preds": preds, "face": (x, y, w, h)})

        color = (0, 200, 0) if label != "Unknown" else (0, 0, 220)
        cv2.rectangle(img_bgr, (x, y), (x+w, y+h), color, 2)
        text  = f"{label}  {conf*100:.1f}%"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(img_bgr, (x, y - th - 10), (x + tw + 6, y), color, -1)
        cv2.putText(img_bgr, text, (x + 3, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    annotated = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return annotated, results

# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("🎭 Face Recognition App")
st.caption(f"Model nhận diện **{len(CLASS_NAMES)} người** · Confidence threshold: {THRESHOLD*100:.0f}%")

tab1, tab2 = st.tabs(["📷 Upload Ảnh", "📹 Webcam"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — UPLOAD ẢNH
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    uploaded = st.file_uploader(
        "Chọn ảnh (jpg / png / jpeg)",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded:
        img_pil = Image.open(uploaded).convert("RGB")
        img_rgb = np.array(img_pil)

        col_in, col_out = st.columns(2)

        with col_in:
            st.subheader("Ảnh gốc")
            st.image(img_rgb, use_container_width=True)

        with st.spinner("🔍 Đang nhận diện..."):
            annotated, results = detect_and_annotate(img_rgb)

        with col_out:
            st.subheader("Kết quả")
            st.image(annotated, use_container_width=True)

        st.divider()
        st.subheader("📊 Chi tiết")

        if len(results) == 1 and results[0]["face"] is None:
            r = results[0]
            st.warning("⚠️ Không detect được khuôn mặt rõ ràng — nhận diện toàn ảnh")
            if r["label"] != "Unknown":
                st.success(f"✅ **{r['label']}** — {r['conf']*100:.1f}%")
            else:
                st.error(f"❓ Unknown (confidence thấp: {r['conf']*100:.1f}%)")
        else:
            for i, r in enumerate(results, 1):
                with st.expander(f"Khuôn mặt #{i} — **{r['label']}** ({r['conf']*100:.1f}%)",
                                 expanded=True):
                    if r["label"] != "Unknown":
                        st.success(f"✅ {r['label']}  —  {r['conf']*100:.1f}%")
                    else:
                        st.error(f"❓ Unknown  —  confidence chỉ {r['conf']*100:.1f}%")

                    # Top 5 predictions
                    top5_idx  = np.argsort(r["preds"])[::-1][:5]
                    st.markdown("**Top 5 dự đoán:**")
                    for j in top5_idx:
                        bar_val = float(r["preds"][j])
                        st.progress(bar_val, text=f"{CLASS_NAMES[j]}: {bar_val*100:.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — WEBCAM
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.info("📸 Chụp ảnh từ webcam rồi nhận diện (Streamlit Cloud không hỗ trợ streaming realtime)")

    cam_img = st.camera_input("Bấm để chụp ảnh")

    if cam_img:
        img_pil = Image.open(cam_img).convert("RGB")
        img_rgb = np.array(img_pil)

        with st.spinner("🔍 Đang nhận diện..."):
            annotated, results = detect_and_annotate(img_rgb)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Ảnh gốc")
            st.image(img_rgb, use_container_width=True)
        with col2:
            st.subheader("Kết quả")
            st.image(annotated, use_container_width=True)

        st.divider()
        for i, r in enumerate(results, 1):
            if r["label"] != "Unknown":
                st.success(f"Khuôn mặt #{i}: ✅ **{r['label']}** — {r['conf']*100:.1f}%")
            else:
                st.error(f"Khuôn mặt #{i}: ❓ Unknown (confidence thấp: {r['conf']*100:.1f}%)")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Built with Streamlit · Model: Keras/TensorFlow · Deploy: Streamlit Cloud")
