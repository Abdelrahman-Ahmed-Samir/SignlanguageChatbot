import streamlit as st
import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
import google.generativeai as genai
import os
import warnings

# Suppress warnings related to protobuf
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

# Configure the Google Generative AI API
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

# Load the sign language model
model_dict = pickle.load(open('C:/Users/Adelrahman/.spyder-py3/model.p', 'rb'))
model = model_dict['model']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

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

# Button to start/stop webcam
start_button = st.button('Start Webcam')
stop_button = st.button('Stop Webcam')

# Variables
sign_string = ""  # To store the detected letters
last_sign_time = time.time()  # Timer for inactivity
inactive_threshold = 3  # Seconds to wait before adding a space
chatbot_response_threshold = 5  # Seconds to wait before sending to chatbot
letter_confirmation_time = 3  # Seconds to confirm the same letter
current_letter = None
letter_start_time = None

# Initialize video capture
cap = cv2.VideoCapture(0)

if start_button:
    # Start processing video
    frame_placeholder = st.empty()  # Placeholder for video frames

    while stop_button == False:
        ret, frame = cap.read()
        if not ret:
            break

        data_aux = []
        x_ = []
        y_ = []
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame using MediaPipe Hands
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                # Extract landmarks
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                # Predict sign language letter
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]

                # Check if the predicted letter is the same as the current letter
                if predicted_character == current_letter:
                    # Confirm the letter if the same letter is held for the confirmation time
                    if time.time() - letter_start_time > letter_confirmation_time:
                        sign_string += current_letter  # Confirm the letter
                        current_letter = None  # Reset the current letter after confirming
                        confirmed_letters_box.text(f"Confirmed letters: {sign_string}")  # Display confirmed letters
                else:
                    # New letter detected
                    current_letter = predicted_character
                    letter_start_time = time.time()  # Reset timer for new letter

                # Draw bounding box and the predicted letter on the frame
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                            cv2.LINE_AA)
                last_sign_time = time.time()  # Reset last sign time

        else:
            # If no hand is detected, check for inactivity
            if time.time() - last_sign_time > inactive_threshold:
                if sign_string and sign_string[-1] != " ":  # Prevent multiple spaces
                    sign_string += " "  # Add space after inactivity
                    confirmed_letters_box.text(f"Confirmed letters: {sign_string}")  # Display confirmed letters
                    last_sign_time = time.time()

        # Show the video stream in Streamlit
        frame_placeholder.image(frame, channels="BGR")

        # Send the accumulated words to the chatbot after inactivity
        if time.time() - last_sign_time > chatbot_response_threshold and sign_string.strip():
            response = genai.GenerativeModel(model_name="gemini-1.5-flash").generate_content(sign_string.strip())
            chatbox.text(f"Chatbot response: {response.text}")
            sign_string = ""  # Clear sign string after processing

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
