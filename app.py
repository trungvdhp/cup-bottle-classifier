import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import tempfile
from moviepy.editor import VideoFileClip
import os


# --------------------------------------
# Load Teachable Machine SavedModel
# --------------------------------------
# Use pure TensorFlow API (not tf.keras.models.load_model)
model = tf.saved_model.load("model/")

# The model uses the "serving_default" signature
infer = model.signatures["serving_default"]

# --------------------------------------
# Print model signatures
# --------------------------------------
print(model.signatures)

# --------------------------------------
# Get input tensor name
# --------------------------------------
input_tensor_name = list(infer.structured_input_signature[1].keys())[0]
output_tensor_name = list(infer.structured_outputs.keys())[0]

# --------------------------------------
# Load class labels
# --------------------------------------

label_path = 'model/labels.txt'

if os.path.exists(label_path):
    with open(label_path) as f:
        labels = [label.strip().split() for label in f.readlines()]
        # Convert to dictionary
        labels = dict((label[0], label[1]) for label in labels)
        print("\nClass labels:", labels)
else:
    print("\nNo labels file found. You may manually create a list of class labels.")
    labels = {"0":"Cup",  "1":"Bottle"}


# --------------------------------------
# Preprocessing Function
# --------------------------------------
def preprocess(image: Image.Image):
    """
    Convert PIL image → tensor with shape (1, 224, 224, 3)
    """
    img = image.resize((224, 224))
    img = np.array(img).astype(np.float32)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # add batch dimension
    # Create input dictionary
    input_dict = {input_tensor_name: tf.convert_to_tensor(img, dtype=tf.float32)}

    return input_dict


# --------------------------------------
# Prediction Function (TensorFlow API)
# --------------------------------------
def predict(image):
    x = preprocess(image)

    # Infer using TensorFlow SavedModel signature
    outputs = infer(**x)

    # Teachable Machine SavedModel returns tensor "outputs"
    preds = outputs[output_tensor_name].numpy()[0]
    
    #for i, score in enumerate(preds):
        #print(labels[f'{i}'], f'{score*100:.2f}%')
        #st.success(f"Prediction: **{labels[f'{i}']}** ({score*100:.2f}%)")
    
    idx = int(np.argmax(preds))
    conf = preds[idx]

    return labels[f'{idx}'], float(conf)


# --------------------------------------
# Streamlit App UI
# --------------------------------------
st.title("Teachable Machine Image Classifier (TensorFlow SavedModel API)")
st.write("Upload an image, capture a photo, or upload a video.")

mode = st.sidebar.selectbox(
    "Select Mode",
    ["Image Upload", "Camera Capture", "Video Upload"]
)


# --------------------------------------
# Mode A — Upload Image
# --------------------------------------
if mode == "Image Upload":
    st.header("Upload Image")

    uploaded = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        label, conf = predict(image)
        st.success(f"Prediction: **{label}** ({conf*100:.2f}%)")


# --------------------------------------
# Mode B — Camera Capture
# --------------------------------------
elif mode == "Camera Capture":
    st.header("Take a Photo")

    picture = st.camera_input("Capture Image")
    if picture:
        image = Image.open(picture).convert("RGB")
        st.image(image, caption="Captured Photo", use_column_width=True)

        label, conf = predict(image)
        st.success(f"Prediction: **{label}** ({conf*100:.2f}%)")


# --------------------------------------
# Mode C — Video Upload (Frame-by-frame)
# --------------------------------------
elif mode == "Video Upload":
    st.header("Upload Video")

    file = st.file_uploader("Choose a video", type=["mp4", "mov", "avi"])
    if file:
        # save uploaded video temporarily
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.read())
            video_path = tmp.name

        st.video(video_path)
        st.write("Processing video… This may take a moment.")

        clip = VideoFileClip(video_path)
        fps = clip.fps

        # Frame-by-frame processing
        def process_frame(frame):
            # moviepy provides frames as RGB np arrays
            img = Image.fromarray(frame)

            label, conf = predict(img)

            # Draw label on frame
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

        # save output video
        output_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        processed_clip.write_videofile(output_path, codec="libx264")

        st.success("Video processing completed!")
        st.video(output_path)
