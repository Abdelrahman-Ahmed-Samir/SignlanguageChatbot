# Sign Language Chatbot

This project is a real-time **Sign Language Recognition System** that converts sign language gestures (A-Z, space, and delete) into text. The system uses MediaPipe for hand landmark detection and a custom-trained machine learning model to recognize each sign. It also integrates with the Google Generative AI to simulate a chatbot that responds to the text formed by the recognized sign language.

# Demo
click on thumbnail to watch the demo
[![Watch the demo](https://img.youtube.com/vi/a8_PHeJ6Yps/maxresdefault.jpg)](https://www.youtube.com/watch?v=a8_PHeJ6Yps)


## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Data Collection](#data-collection)
- [Model Training](#model-training)
- [Model Inference](#model-inference)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)


## Overview
The **Sign Language Chatbot** project aims to bridge the communication gap between people who use sign language and those who don’t understand it. It captures hand signs via webcam, predicts the corresponding letter using a machine learning model, and sends the complete text to a chatbot powered by Google Generative AI for interaction.

The model supports:
- Letters from **A** to **Z**
- Special gestures for **space** and **delete**

## Features
- **Real-Time Hand Gesture Recognition**: Detects hand landmarks using MediaPipe and predicts the corresponding letter using a trained model.
- **Confirmation Timer**: Ensures that signs are confirmed only after holding the gesture for a set duration (3 seconds for letters, 2 seconds for space and delete).
- **Chatbot Interaction**: Converts detected letters into a string and sends it to a chatbot API to generate meaningful responses.
- **Webcam Integration**: Live feed from the webcam processes the hand gestures in real-time.

## How It Works
1. **Sign Detection**: Using MediaPipe's Hand module, the hand landmarks are detected in real-time.
2. **Prediction**: These landmarks are fed into a pre-trained machine learning model that predicts the signed letter (A-Z, space, delete).
3. **Confirmation**: After confirming the same gesture for a specific time (3 seconds for letters, 2 seconds for space/delete), the letter is added to the sign string.
4. **Chatbot Response**: When the user stops signing for a defined period, the system sends the accumulated string to Google Generative AI, and the chatbot responds.

## Project Structure

```plaintext
.
├── README.md               # This file
├── model.p                 # The pre-trained model for sign language recognition (loaded using pickle)
├── main.py                 # Main application file to run the real-time sign recognition and chatbot interaction
├── requirements.txt        # Dependencies and libraries needed for the project
├── collect_imgs.py         # Script to collect images for hand signs
├── create_dataset.py       # Script to create a dataset and save it to a pickle file
├── inference.py            # Inference script to test sign language recognition without chatbot
├── inference_with_chatbot.py # Inference script to test sign recognition with chatbot interaction
├── Train_Model.py # Train script to train the model on sign language dataset for recognition
```
## Key Files

- **`model.p`**: This file contains the pre-trained model (loaded using Python's `pickle` module) that predicts the sign language gestures.
- **`main.py`**: The main file that initializes the webcam feed, processes hand landmarks, predicts the sign language gestures, and interacts with the Google Generative AI chatbot.
- **`requirements.txt`**: This file lists all the dependencies required to run the project. You can install them using:
  ```bash
  pip install -r requirements.txt

## Data Collection

This project includes two files to help create a dataset of hand signs for training the sign language recognition model:

1. **collect_imgs.py**: This script captures real-time images of hand signs using the webcam. It allows you to manually collect images for each sign (A-Z, space, delete), which are then saved locally for further processing.

   **To run the image collection script**:
   ```bash
   python collect_imgs.py
Functionality:
Opens a webcam feed for real-time hand sign capture.
Allows you to manually label and save images for each sign gesture (A-Z, space, delete).
The images are saved in separate folders, each representing a specific sign.
Usage Scenario: Use this script to collect your own hand sign data for model training.

2. **create_dataset.py**: This script processes the collected hand sign images and creates a dataset by extracting landmarks (using MediaPipe) from the images. The processed dataset is saved as a pickle file for easy use during model training.

To run the dataset creation script:
```bash
  python create_dataset.py
```
Functionality:
Reads the images collected by collect_imgs.py.
Extracts hand landmarks from each image using MediaPipe.
Saves the processed data (landmarks and labels) into a pickle file (dataset.pkl), which can be used for training the machine learning model.
Usage Scenario: After collecting hand sign images, use this script to create the dataset by extracting features and storing them in a format ready for model training.


### Explanation:
- Each file is described with its purpose, command to run, functionality, and usage scenario.
- The workflow explains the sequential process of collecting images and then creating a dataset, giving users a clear understanding of how to use the scripts.


## Model Training

The model used in this project is trained to recognize 29 classes (A-Z, space, delete) using hand landmarks as input features. The training process involves:

- Capturing hand sign images.
- Extracting hand landmarks using **MediaPipe**.
- Training a machine learning model on these features to predict the sign.
  
**To run the Train_Model.py**:
   ```bash
   python Train_Model.py
  ```

## Model Inference

This project provides two separate inference scripts for testing the sign language recognition system:

1. **inference.py**: This script is used to test the sign language recognition model independently, without the chatbot. It captures real-time hand gestures, predicts the corresponding sign, and displays the predicted output.

   **To run the inference for sign language recognition**:
   ```bash
   python inference.py
Functionality: It opens a webcam feed, detects hand gestures using MediaPipe, and displays the recognized letter (A-Z, space, delete) in real-time.
Usage Scenario: Use this script if you're only interested in testing the sign recognition model without chatbot interaction.

2. **inference_with_chatbot.py**: This script tests the sign language recognition model integrated with the Google Generative AI chatbot. After detecting and recognizing a sign language sequence, the system sends the recognized string to the chatbot, which responds in real-time.

To run the inference with the chatbot:
```bash
   python inference_with_chatbot.py
```
Functionality: It opens a webcam feed, detects hand gestures, and predicts the corresponding sign. After forming a sentence, it sends the string to the Google Generative AI chatbot, which responds to the signed text.
Usage Scenario: Use this script when you want to test both sign language recognition and chatbot interaction in real-time.
Key Feature: Once a sentence is formed and confirmed, the chatbot responds after 5 seconds of inactivity.


### Explanation:
- Each file is described with its purpose, command to run, functionality, and appropriate use case.
- The differences between the two inference scripts are clearly highlighted, so users know which one to use based on their goals.


## Setup and Installation

### Prerequisites

- **Python 3.x**
- A **webcam** (for real-time gesture detection)
- A **Google Cloud account** with access to the Google Generative AI API (for chatbot interaction)

### Installation Steps

1. Clone the repository:
    ```bash
    git clone https://github.com/Abdelrahman-Ahmed-Samir/SignlanguageChatbot.git
    cd SignlanguageChatbot
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up the Google Generative AI API by obtaining an API key and storing it in an environment variable:
    ```bash
    export GOOGLE_API_KEY='your_gemini_api_key'
    ```

4. Run the application:
    ```bash
    streamlit run main.py
    ```

## Setting Up API Keys

This project uses the Google Generative AI API to generate chatbot responses. To use this feature:

1. Sign up for **Google Cloud** and enable the **Generative AI API**.
2. Create an API key and store it in your environment variables as `GOOGLE_API_KEY`.

## Usage

```bash
streamlit run main.py
```
The Streamlit interface will open in your default web browser. Use the webcam feed to start signing letters (A-Z), "space," and "delete" gestures. After forming a sentence, wait for 5 seconds, and the system will send the string to the chatbot, which will respond in real time.

### Key Controls:

- **Sign a letter**: Hold the hand gesture for 3 seconds to confirm a letter.
- **Space/Delete**: Sign the space or delete gesture and hold for 2 seconds.
- **Chatbot response**: After 5 seconds of inactivity, the chatbot will respond to the signed text.

---

### Technologies Used:

- **Python**: The core programming language.
- **Streamlit**: Framework for creating the real-time web application.
- **MediaPipe**: For real-time hand landmark detection.
- **Google Generative AI**: For chatbot interaction.
- **OpenCV**: For handling webcam input and image processing.
- **NumPy**: For handling numerical operations.
- **Pickle**: For loading the trained machine learning model.

---

### Contributing:

Contributions are welcome! Feel free to open a pull request or issue if you find a bug or have a suggestion for improving the project.

#### How to Contribute:

1. **Fork** the repository.
2. **Create** a new feature branch.
3. **Commit** your changes.
4. **Push** your branch and **create** a pull request.

## License

This project is licensed under the MIT License. 

### MIT License

Copyright (c) 2024 Abdelrahman Ahmed Samir

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
