import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import urllib.request

# ESP32 CAM URL
url = 'http://192.168.1.6/cam-hi.jpg'

# Object classes
classNames = ["Immature Sawi", "Mature sawi", "Non-sawi", "Partially mature sawi", "Rotten"]

def process_frame(frame, model, min_confidence):
    # Proses frame untuk deteksi objek
    results = model(frame)
    
    # Gambarkan hasil deteksi pada frame
    for detection in results[0].boxes.data:
        x0, y0 = (int(detection[0]), int(detection[1]))
        x1, y1 = (int(detection[2]), int(detection[3]))
        score = round(float(detection[4]), 2)
        cls = int(detection[5])
        object_name = classNames[cls]
        label = f'{object_name} {score}'

        if score > min_confidence:  # Gunakan ambang batas untuk menampilkan deteksi
            cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 0), 2)
            cv2.putText(frame, label, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return frame

def main():
    st.title("Object Detection with YOLOv8")

    # Create 2 rows and 2 columns layout
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Widget 1")
        widget_url1 = "https://stem.ubidots.com/app/dashboards/public/widget/n2DJ6zraCJkvxZYYAQ5egHCTgZLe6E3XBpVtLGnZsoQ"
        st.components.v1.iframe(widget_url1, width=500, height=400, scrolling=True)

        st.subheader("Widget 2")
        widget_url2 = "https://stem.ubidots.com/app/dashboards/public/widget/xggZMNEOQq-32hJmgV9ovScYSPNe9iyO77bi1lB5oE4"
        st.components.v1.iframe(widget_url2, width=400, height=400, scrolling=True)

    with col2:
        st.subheader("Widget 3")
        widget_url3 = "https://stem.ubidots.com/app/dashboards/public/widget/XXSQaCPoG41tQ1W33PDj9xphZOO7DwF6tvflxiKnSkE"
        st.components.v1.iframe(widget_url3, width=500, height=400, scrolling=True)

        st.subheader("Widget 4")
        widget_url4 = "https://stem.ubidots.com/app/dashboards/public/widget/ST57XPDVjOhWeqD1GHC1ejT2zCuxr078rU-tQH6WNKo"
        st.components.v1.iframe(widget_url4, width=700, height=400, scrolling=True)

    st.write("This is an Ubidots widget embedded in a Streamlit app.")

    # Muat model YOLOv8
    model = YOLO('trained_model.pt')

    # Slider untuk ambang batas kepercayaan
    min_confidence = st.slider('Confidence threshold', 0.0, 1.0, 0.3)

    st.write("Starting video stream from ESP32 CAM...")

    stframe = st.empty()
    stop_button = st.button('Stop', key='stop_button')

    while True:
        # Ambil gambar dari ESP32 CAM
        img_resp = urllib.request.urlopen(url)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        frame = cv2.imdecode(imgnp, -1)

        if frame is None:
            st.error("Error: Unable to capture video from ESP32 CAM.")
            break

        # Proses frame
        frame = process_frame(frame, model, min_confidence)

        # Convert frame BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Tampilkan frame dengan Streamlit
        stframe.image(frame_rgb, channels="RGB", use_column_width=True)

        # Hentikan loop jika tombol 'Stop' ditekan
        if stop_button:
            break

    st.write("Video stream stopped")

if __name__ == "__main__":
    main()