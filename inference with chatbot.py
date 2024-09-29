# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 23:08:48 2024

@author: Adelrahman
"""

import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
import warnings
import google.generativeai as genai
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

# Suppress warnings related to protobuf
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

# Configure the Google Generative AI API
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

# Load the sign language model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Mapping of labels (A-Y)
labels_dict = {i: chr(65 + i) for i in range(25)}  # A to Y
sign_string = ""  # To store the detected letters
last_sign_time = time.time()  # Timer for inactivity
inactive_threshold = 3  # Seconds to wait before adding a space
chatbot_response_threshold = 5  # Seconds to wait before sending to chatbot
letter_confirmation_time = 3  # Seconds to confirm the same letter

current_letter = None
letter_start_time = None

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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

            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            # Check if the predicted letter is the same as the current letter
            if predicted_character == current_letter:
                # If it is the same letter, check if we are within the confirmation time
                if time.time() - letter_start_time > letter_confirmation_time:
                    sign_string += current_letter  # Confirm the letter
                    print("Confirmed letter:", current_letter)
                    current_letter = None  # Reset the current letter after confirming
            else:
                # New letter detected
                current_letter = predicted_character
                letter_start_time = time.time()  # Reset timer for new letter
                print("Current sign:", current_letter)

            # Draw the prediction on the frame
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)
            last_sign_time = time.time()  # Reset the last sign time for activity
    else:
        # If no hand is detected, check for inactivity
        if time.time() - last_sign_time > inactive_threshold:
            if sign_string and sign_string[-1] != " ":  # Prevent adding multiple spaces
                sign_string += " "  # Add a space after 3 seconds of inactivity
                last_sign_time = time.time()  # Reset the timer after adding space

    cv2.imshow('frame', frame)
    
    # If 5 seconds of inactivity and user is done signing
    if time.time() - last_sign_time > chatbot_response_threshold and sign_string:
        # Send the accumulated words to the chatbot
        response = genai.GenerativeModel(model_name="gemini-1.5-flash").generate_content(sign_string.strip())
        print("Chatbot response:", response.text)
        sign_string = ""  # Clear the sign_string after processing
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit and print the string
        print("Final sign string:", sign_string)
        break

cap.release()
cv2.destroyAllWindows()
