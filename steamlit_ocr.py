import streamlit as st
import cv2
import threading
from PIL import Image

# Function to capture frames from the webcam
def capture_webcam():
    cap = cv2.VideoCapture(0)

    while st.session_state.capture:
        ret, frame = cap.read()

        if not ret:
            st.error("Failed to capture frame from webcam.")
            break

        # Convert the frame to RGB format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the webcam feed in Streamlit
        st.image(rgb_frame, caption="Webcam Feed", use_column_width=True)

    # Release the webcam when done
    cap.release()

# Streamlit app with interactive window
def main():
    st.title("Interactive Webcam Window with Streamlit")

    # Display a button to start capturing frames from the webcam
    if not st.session_state.get("capture"):
        st.session_state.capture = False

    if st.button("Start Webcam"):
        st.session_state.capture = True
        # Create a thread to run the webcam capture function
        webcam_thread = threading.Thread(target=capture_webcam)
        webcam_thread.start()

    # Display a button to stop capturing frames
    if st.button("Stop Webcam"):
        st.session_state.capture = False

if __name__ == "__main__":
    main()
