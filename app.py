import cv2
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
import numpy as np


def load_pretrained_model():
    return load_model("unet_model.h5")


def process_frame_with_model(frame, model):
    input_frame = cv2.resize(frame, (256, 256))
    input_frame = input_frame / 255.0
    input_frame = np.expand_dims(input_frame, axis=0)
    segmentation_output = model.predict(input_frame)[0]
    segmentation_output_resized = cv2.resize(segmentation_output, (frame.shape[1], frame.shape[0]))

    color_map = {
        0: (0, 0, 0),
        1: (255, 0, 0),
        2: (0, 255, 0),
        3: (0, 0, 255),
        4: (255, 255, 0),
        5: (255, 0, 255),
        6: (0, 255, 255)
    }

    segmented_frame = np.zeros_like(frame)

    for class_id, color in color_map.items():
        segmented_frame[segmentation_output_resized.argmax(axis=-1) == class_id] = color

    blended_frame = cv2.addWeighted(frame, 0.6, segmented_frame, 0.4, 0)
    return blended_frame


def main():
    st.title("Real-Time Multi-Object Colorization with Semantic Segmentation")

    st.sidebar.title("Options")
    use_model = st.sidebar.checkbox("Use Pretrained Model")
    upload_video = st.sidebar.file_uploader("Upload a Video File", type=["mp4", "avi"])
    start_stream = st.sidebar.button("Start Processing")
    model = None

    if use_model:
        st.sidebar.write("Loading pretrained model...")
        model = load_pretrained_model()
        st.sidebar.success("Pretrained model loaded!")

    if start_stream:
        if upload_video:
            video_path = upload_video.name
            with open(video_path, "wb") as f:
                f.write(upload_video.getbuffer())
            video_capture = cv2.VideoCapture(video_path)
        else:
            video_capture = cv2.VideoCapture(0)

        stframe = st.empty()

        while video_capture.isOpened():
            ret, frame = video_capture.read()

            if not ret:
                st.warning("No video frame detected or end of video. Stopping stream.")
                break

            if use_model and model:
                processed_frame = process_frame_with_model(frame, model)
            else:
                processed_frame = colorize_and_track_face(frame)

            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)

            stframe.image(frame_pil, caption="Processed Video Stream", use_column_width=True)

        video_capture.release()


if __name__ == "__main__":
    main()
