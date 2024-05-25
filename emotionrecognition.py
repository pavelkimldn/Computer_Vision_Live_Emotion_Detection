import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Define paths
data_dir = 'Desktop/emotion recognition'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')
emotions = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
image_size = (48, 48)
max_images_per_emotion = 500  # Change this value to limit the number of images per emotion

# Function to load images and labels with HOG features
def load_data_with_hog(data_dir, emotions, max_images=None):
    images = []
    labels = []
    for emotion_id, emotion in enumerate(emotions):
        emotion_train_dir = os.path.join(data_dir, 'train', emotion)
        emotion_images = []
        for image_file in os.listdir(emotion_train_dir)[:max_images]:
            image_path = os.path.join(emotion_train_dir, image_file)
            image = cv2.imread(image_path)
            image = cv2.resize(image, image_size)
            # Convert image to grayscale
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hog_features = hog(gray_image, orientations=8, pixels_per_cell=(8, 8),
                               cells_per_block=(1, 1), transform_sqrt=True)
            images.append(hog_features)
            labels.append(emotion_id)
    return np.array(images), np.array(labels)

# Load training data with HOG features
X_train, y_train = load_data_with_hog(data_dir, emotions, max_images=max_images_per_emotion)

# Function to load test images and labels with HOG features
def load_test_data_with_hog(data_dir, emotions, max_images=None):
    images = []
    labels = []
    for emotion_id, emotion in enumerate(emotions):
        emotion_test_dir = os.path.join(data_dir, 'test', emotion)
        emotion_images = []
        for image_file in os.listdir(emotion_test_dir)[:max_images]:
            image_path = os.path.join(emotion_test_dir, image_file)
            image = cv2.imread(image_path)
            image = cv2.resize(image, image_size)
            # Convert image to grayscale
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hog_features = hog(gray_image, orientations=8, pixels_per_cell=(8, 8),
                               cells_per_block=(1, 1), transform_sqrt=True)
            images.append(hog_features)
            labels.append(emotion_id)
    return np.array(images), np.array(labels)

# Load test data with HOG features
X_test, y_test = load_test_data_with_hog(data_dir, emotions, max_images=max_images_per_emotion)

# Define a pipeline with feature scaling and Random Forest classifier
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Feature scaling
    ('clf', RandomForestClassifier(random_state=42))
])

# Define hyperparameters for grid search
param_grid = {
    'clf__n_estimators': [50 ],
    'clf__max_depth': [10]
}

#param_grid = {
  #  'clf__n_estimators': [50,100,200 ],
 #   'clf__max_depth': [none,10,20]
#}
#
# Perform grid search with cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', verbose=1)
grid_search.fit(X_train, y_train)

# Get the best model from grid search
best_model = grid_search.best_estimator_

# Evaluate the best model on the test set
test_accuracy = best_model.score(X_test, y_test)
print("Test Accuracy (after hyperparameter tuning):", test_accuracy)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_emotion_in_video(video_path):
    # Load the video
    video_capture = cv2.VideoCapture(video_path)
    while True:
        # Read a frame from the video
        ret, frame = video_capture.read()
        if not ret:
            break  # End of video
        # Detect faces in the frame
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = gray_frame[y:y+h, x:x+w]
            # Resize face region to match training image size
            resized_face_roi = cv2.resize(face_roi, image_size)
            # Extract HOG features
            hog_features = hog(resized_face_roi, orientations=8, pixels_per_cell=(8, 8),
                               cells_per_block=(1, 1), transform_sqrt=True)
            # Predict emotion using the trained model
            predicted_emotion_id = best_model.predict([hog_features])[0]
            predicted_emotion = emotions[predicted_emotion_id]
            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            # Display the predicted emotion on the frame
            cv2.putText(frame, predicted_emotion, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        # Display the frame
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release the video capture object
    video_capture.release()
    cv2.destroyAllWindows()

# Path to the video file
video_path = 'path_to_your_video_file.mp4'

# Detect emotion in the video
#detect_emotion_in_video('Desktop/video.mp4')
