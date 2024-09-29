
import numpy as np
import pickle
from scipy.interpolate import interp1d
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the data from the pickle file
with open('data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

data = data_dict['data']  # List of landmark data
labels = np.asarray(data_dict['labels'])  # Corresponding labels

# Define the fixed number of points (landmarks) we want
fixed_length = 42  # Example: 21 landmarks, each with x and y coordinates (21 * 2 = 42)

# Function to interpolate landmark data to a fixed length
def interpolate_landmarks(landmarks, fixed_length):
    current_length = len(landmarks)
    
    if current_length == fixed_length:
        return landmarks
    
    x = np.linspace(0, 1, num=current_length)
    x_new = np.linspace(0, 1, num=fixed_length)

    # Interpolate for x and y values separately
    landmarks = np.array(landmarks)
    f = interp1d(x, landmarks, axis=0, fill_value="extrapolate")
    landmarks_interpolated = f(x_new)
    
    return landmarks_interpolated

# Interpolate each sequence of landmarks
processed_data = [interpolate_landmarks(d, fixed_length) for d in data]

# Convert to numpy arrays for model training
data_array = np.asarray(processed_data)
labels_array = np.asarray(labels)

print(f"Processed data shape: {data_array.shape}")
print(f"Processed labels shape: {labels_array.shape}")


# Split the data
x_train, x_test, y_train, y_test = train_test_split(data_array, labels_array, test_size=0.2, shuffle=True, stratify=labels_array)

# Train the RandomForest model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Predict and evaluate
y_predict = model.predict(x_test)
score = accuracy_score(y_test, y_predict)
print(f'{score * 100:.2f}% of samples were classified correctly!')

# Save the trained model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
