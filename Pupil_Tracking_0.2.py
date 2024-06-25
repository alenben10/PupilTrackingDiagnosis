import cv2
import mediapipe as mp
import pandas as pd
import time
import pyautogui
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Initialize MediaPipe Face Mesh and drawing utilities
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 60)  # Set the FPS to 60

# Initialize a list to store coordinates
eye_coords = []

# Define the landmark indices for the left and right irises
LEFT_IRIS = [474, 475, 476, 477]  # Indices for left iris
RIGHT_IRIS = [469, 470, 471, 472]  # Indices for right iris

# Get screen dimensions
screen_width, screen_height = pyautogui.size()

# Calibration points (e.g., 9 points on the screen for better accuracy)
calibration_points_screen = [
    (screen_width // 4, screen_height // 4),
    (screen_width // 2, screen_height // 4),
    (3 * screen_width // 4, screen_height // 4),
    (screen_width // 4, screen_height // 2),
    (screen_width // 2, screen_height // 2),
    (3 * screen_width // 4, screen_height // 2),
    (screen_width // 4, 3 * screen_height // 4),
    (screen_width // 2, 3 * screen_height // 4),
    (3 * screen_width // 4, 3 * screen_height // 4)
]

calibration_data = []

def calibrate():
    global calibration_data
    calibration_data = []
    for point in calibration_points_screen:
        pyautogui.moveTo(point[0], point[1])
        pyautogui.click()
        print(f"Looking at point: {point}")
        time.sleep(1)
        
        # Collect multiple frames for each calibration point
        frame_coords = []
        for _ in range(10):  # Adjust the number of frames to average over
            ret, frame = cap.read()
            if not ret:
                continue
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    left_iris_x = sum(face_landmarks.landmark[i].x for i in LEFT_IRIS) / len(LEFT_IRIS)
                    left_iris_y = sum(face_landmarks.landmark[i].y for i in LEFT_IRIS) / len(LEFT_IRIS)
                    right_iris_x = sum(face_landmarks.landmark[i].x for i in RIGHT_IRIS) / len(RIGHT_IRIS)
                    right_iris_y = sum(face_landmarks.landmark[i].y for i in RIGHT_IRIS) / len(RIGHT_IRIS)

                    height, width, _ = frame.shape
                    left_iris_x_px = int(left_iris_x * width)
                    left_iris_y_px = int(left_iris_y * height)
                    right_iris_x_px = int(right_iris_x * width)
                    right_iris_y_px = int(right_iris_y * height)
                    
                    frame_coords.append((left_iris_x_px, left_iris_y_px, right_iris_x_px, right_iris_y_px))
        
        # Average the coordinates over the collected frames
        avg_coords = np.mean(frame_coords, axis=0)
        calibration_data.append((*avg_coords, point[0], point[1]))

calibrate()

# Convert calibration data to numpy arrays for easier manipulation
eye_data = np.array(calibration_data)
pupil_coords = eye_data[:, :4]
screen_coords = eye_data[:, 4:]

# Use polynomial regression to map pupil coordinates to screen coordinates
degree = 2  # Adjust the degree of the polynomial as needed
poly_model_x = make_pipeline(PolynomialFeatures(degree), LinearRegression())
poly_model_y = make_pipeline(PolynomialFeatures(degree), LinearRegression())

poly_model_x.fit(pupil_coords, screen_coords[:, 0])
poly_model_y.fit(pupil_coords, screen_coords[:, 1])

def map_pupil_to_gaze(pupil_x, pupil_y):
    pupil_data = np.array([pupil_x, pupil_y, pupil_x, pupil_y]).reshape(1, -1)
    screen_x = poly_model_x.predict(pupil_data)[0]
    screen_y = poly_model_y.predict(pupil_data)[0]
    return int(screen_x), int(screen_y)

# Start the gaze tracking for 21 seconds at 60fps
start_time = time.time()
record_duration = 21  # in seconds

while cap.isOpened() and (time.time() - start_time < record_duration):
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_iris_x = sum(face_landmarks.landmark[i].x for i in LEFT_IRIS) / len(LEFT_IRIS)
            left_iris_y = sum(face_landmarks.landmark[i].y for i in LEFT_IRIS)
            right_iris_x = sum(face_landmarks.landmark[i].x for i in RIGHT_IRIS) / len(RIGHT_IRIS)
            right_iris_y = sum(face_landmarks.landmark[i].y for i in RIGHT_IRIS)

            height, width, _ = frame.shape
            left_iris_x_px = int(left_iris_x * width)
            left_iris_y_px = int(left_iris_y * height)

            screen_x, screen_y = map_pupil_to_gaze(left_iris_x_px, left_iris_y_px)

            timestamp = time.time()
            eye_coords.append([timestamp, screen_x, screen_y])

            cv2.circle(frame, (left_iris_x_px, left_iris_y_px), 3, (0, 255, 0), -1)
            cv2.putText(frame, f'Gaze: ({screen_x}, {screen_y})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Eye Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

df = pd.DataFrame(eye_coords, columns=['timestamp', 'screen_x', 'screen_y'])
df.to_csv('gaze_coordinates.csv', index=False)

print("Gaze coordinates saved to gaze_coordinates.csv")
