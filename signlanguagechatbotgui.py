import streamlit as st
import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
import google.generativeai as genai
import os
import warnings
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer, WebRtcMode
from sklearn.preprocessing import StandardScaler

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

# Mapping of labels (A-Y)
labels_dict = {i: chr(65 + i) for i in range(25)}  # A to Y

# Streamlit page configuration
st.set_page_config(page_title="Sign Language to Chatbot", layout="wide")

# Define Streamlit layout
st.title("Real-Time Sign Language Recognition and Chatbot Interaction")
st.text("Use the webcam to sign and see the chatbot response below!")

# Display area for the chatbot response and detected letters
chatbox = st.empty()
confirmed_letters_box = st.empty()  # Box for displaying confirmed letters

# Variables
sign_string = ""  # To store the detected letters
last_sign_time = time.time()  # Timer for inactivity
inactive_threshold = 3  # Seconds to wait before adding a space
chatbot_response_threshold = 5  # Seconds to wait before sending to chatbot
letter_confirmation_time = 3  # Seconds to confirm the same letter
current_letter = None
letter_start_time = None

# Video processing class for Streamlit WebRTC
class SignLanguageProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5)
        self.model = model
        self.labels_dict = labels_dict
        self.sign_string = ""
        self.current_letter = None
        self.letter_start_time = None
        self.last_sign_time = time.time()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W, _ = img.shape

        # Process the frame using MediaPipe Hands
        results = self.hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                # Extract landmarks
                data_aux = []
                x_ = []
                y_ = []
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    data_aux.append(hand_landmarks.landmark[i].x - min(x_))
                    data_aux.append(hand_landmarks.landmark[i].y - min(y_))

                # Predict sign language letter
                prediction = self.model.predict([np.asarray(data_aux)])
                predicted_character = self.labels_dict[int(prediction[0])]

                # Check if the predicted letter is the same as the current letter
                if predicted_character == self.current_letter:
                    # Confirm the letter if the same letter is held for the confirmation time
                    if time.time() - self.letter_start_time > letter_confirmation_time:
                        self.sign_string += self.current_letter  # Confirm the letter
                        self.current_letter = None  # Reset the current letter after confirming
                        confirmed_letters_box.text(f"Confirmed letters: {self.sign_string}")  # Display confirmed letters
                else:
                    # New letter detected
                    self.current_letter = predicted_character
                    self.letter_start_time = time.time()  # Reset timer for new letter

                # Draw bounding box and the predicted letter on the frame
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(img, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                            cv2.LINE_AA)
                self.last_sign_time = time.time()  # Reset last sign time

        else:
            # If no hand is detected, check for inactivity
            if time.time() - self.last_sign_time > inactive_threshold:
                if self.sign_string and self.sign_string[-1] != " ":  # Prevent multiple spaces
                    self.sign_string += " "  # Add space after inactivity
                    confirmed_letters_box.text(f"Confirmed letters: {self.sign_string}")  # Display confirmed letters
                    self.last_sign_time = time.time()

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# Start the WebRTC streamer
webrtc_ctx = webrtc_streamer(
    key="sign-language",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=SignLanguageProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Send the accumulated words to the chatbot after inactivity
if time.time() - last_sign_time > chatbot_response_threshold and sign_string.strip():
    response = genai.GenerativeModel(model_name="gemini-1.5-flash").generate_content(sign_string.strip())
    chatbox.text(f"Chatbot response: {response.text}")
    sign_string = ""  # Clear sign string after processing
