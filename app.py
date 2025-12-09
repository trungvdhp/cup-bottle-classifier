import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import tempfile
from moviepy.editor import VideoFileClip

# -------------------------
# Load Keras Model (Teachable Machine)
# -------------------------
model = tf.keras.models.load_model("model/keras_model.h5")

# Edit this list based on your TM training labels
class_labels = ["Class A", "Class B", "Class C"]


# -------------------------
# Preprocessing
# -------------------------
def preprocess(image):
    img = image.resize((224, 224))
    img = np.array(img).astype(np.float32)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def predict(image):
    x = preprocess(image)
    preds = model.predict(x)[0]
    idx = np.argmax(preds)
    return class_labels[idx], preds[idx]


# -------------------------
# Streamlit UI
# -------------------------
st.title("Teachable Machine Image Classifier")
st.write("Upload an image, capture photo from camera, or upload a video.")

option = st.sidebar.selectbox(
    "Choose mode",
    ["Image Upload", "Camera Capture", "Video Upload"]
)


# -------------------------
# MODE A — Upload Image
# -------------------------
if option == "Image Upload":
    st.header("Upload an Image")

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        label, conf = predict(image)

        st.success(f"Prediction: **{label}** ({conf*100:.2f}%)")


# -------------------------
# MODE B — Capture from Camera
# -------------------------
elif option == "Camera Capture":
    st.header("Take a Photo")

    picture = st.camera_input("Capture Image")

    if picture:
        image = Image.open(picture)
        st.image(image, caption="Captured Photo", use_column_width=True)

        label, conf = predict(image)
        st.success(f"Prediction: **{label}** ({conf*100:.2f}%)")


# -------------------------
# MODE C — Upload Video (Frame-by-frame)
# -------------------------
elif option == "Video Upload":
    st.header("Upload a Video")

    file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])

    if file:
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.read())
            video_path = tmp.name

        st.video(video_path)

        st.write("Processing video... please wait.")

        # Process frame-by-frame
        clip = VideoFileClip(video_path)
        fps = clip.fps

        def process_frame(frame):
            # Convert BGR -> RGB
            image = Image.fromarray(frame)
            label, conf = predict(image)

            # Draw text onto the frame
            frame = cv2.putText(
                frame.copy(),
                f"{label} {conf*100:.1f}%",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            return frame

        processed_clip = clip.fl_image(process_frame)

        # Save processed video
        output_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        processed_clip.write_videofile(output_path, codec="libx264")

        st.success("Video processing completed!")
        st.video(output_path)
