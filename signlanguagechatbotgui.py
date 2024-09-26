import streamlit as st
import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
import google.generativeai as genai
import os
import warnings
import av
import queue
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoTransformerBase

# Suppress warnings related to protobuf
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

# Configure the Google Generative AI API
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

# Load the sign language model
model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3, min_tracking_confidence=0.5)

# Mapping of labels (A-Y)
labels_dict = {i: chr(65 + i) for i in range(25)}  # A to Y

# Streamlit page configuration
st.set_page_config(page_title="Sign Language to Chatbot", layout="wide")

# Define Streamlit layout
st.title("Real-Time Sign Language Recognition and Chatbot Interaction")
st.text("Use the webcam to sign and see the chatbot response below!")

# Display areas for chatbot response and detected letters
chatbox = st.empty()
confirmed_letters_box = st.empty()  # Box for displaying confirmed letters

# Variables for tracking sign language input
sign_string = ""  # To store the detected letters
last_sign_time = time.time()  # Timer for inactivity
chatbot_response_threshold = 5  # Seconds to wait before sending to chatbot
letter_confirmation_time = 3  # Seconds to confirm the same letter

# Result queue for thread-safe communication
result_queue = queue.Queue()

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.current_letter = None
        self.letter_start_time = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        data_aux = []
        H, W, _ = img.shape
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the frame using MediaPipe Hands
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # Extract landmarks and normalize
                for landmark in hand_landmarks.landmark:
                    x = landmark.x
                    y = landmark.y
                    data_aux.append(x)
                    data_aux.append(y)

                # Predict sign language letter
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]

                # Confirm the letter
                if predicted_character == self.current_letter:
                    if time.time() - self.letter_start_time > letter_confirmation_time:
                        global sign_string
                        sign_string += self.current_letter
                        self.current_letter = None
                        result_queue.put(sign_string)  # Thread-safe update
                else:
                    self.current_letter = predicted_character
                    self.letter_start_time = time.time()

                # Draw bounding box and the predicted letter on the frame
                x1, y1 = int(landmark.x * W) - 10, int(landmark.y * H) - 10
                x2, y2 = int(landmark.x * W) + 10, int(landmark.y * H) + 10
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(img, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Start the webcam stream using streamlit-webrtc
try:
    webrtc_ctx = webrtc_streamer(
        key="example",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoTransformer,
        media_stream_constraints={"video": True, "audio": False},  # Disable audio if not needed
        async_processing=True,  # Run the video processing asynchronously
    )
except Exception as e:
    st.error(f"Error starting the webcam: {e}")

# Display detected letters and handle chatbot response
if webrtc_ctx and webrtc_ctx.state.playing:
    while True:
        try:
            sign_string = result_queue.get_nowait()  # Get the latest sign string
            confirmed_letters_box.text(f"Confirmed letters: {sign_string}")

            # Send the accumulated words to the chatbot after inactivity
            if time.time() - last_sign_time > chatbot_response_threshold and sign_string.strip():
                response = genai.GenerativeModel(model_name="gemini-1.5-flash").generate_content(sign_string.strip())
                chatbox.text(f"Chatbot response: {response.text}")
                sign_string = ""  # Clear sign string after processing
                last_sign_time = time.time()  # Reset the inactivity timer

        except queue.Empty:
            continue
        except Exception as e:
            st.error(f"An error occurred: {e}")
