import cv2
import mediapipe as mp
import pandas as pd
import time

# Initialize MediaPipe Face Mesh and drawing utilities
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 60)  # Set the FPS to 60

# Initialize a list to store coordinates
eye_coords = []

# Record for 20 seconds
start_time = time.time()
record_duration = 20  # in seconds

# Define the landmark indices for the left and right irises
LEFT_IRIS = [474, 475, 476, 477]  # Indices for left iris
RIGHT_IRIS = [469, 470, 471, 472]  # Indices for right iris

while cap.isOpened() and (time.time() - start_time < record_duration):
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Face Mesh
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extracting the coordinates of the left and right irises
            left_iris_x = sum(face_landmarks.landmark[i].x for i in LEFT_IRIS) / len(LEFT_IRIS)
            left_iris_y = sum(face_landmarks.landmark[i].y for i in LEFT_IRIS) / len(LEFT_IRIS)
            right_iris_x = sum(face_landmarks.landmark[i].x for i in RIGHT_IRIS) / len(RIGHT_IRIS)
            right_iris_y = sum(face_landmarks.landmark[i].y for i in RIGHT_IRIS) / len(RIGHT_IRIS)

            # Convert normalized coordinates to pixel coordinates
            height, width, _ = frame.shape
            left_iris_x_px = int(left_iris_x * width)
            left_iris_y_px = int(left_iris_y * height)
            right_iris_x_px = int(right_iris_x * width)
            right_iris_y_px = int(right_iris_y * height)

            # Save the coordinates along with the timestamp
            timestamp = time.time()
            eye_coords.append([timestamp, left_iris_x_px, left_iris_y_px, right_iris_x_px, right_iris_y_px])

            # Draw the iris landmarks on the frame
            cv2.circle(frame, (left_iris_x_px, left_iris_y_px), 3, (0, 255, 0), -1)
            cv2.circle(frame, (right_iris_x_px, right_iris_y_px), 3, (0, 255, 0), -1)

    # Display the frame
    cv2.imshow('Eye Tracking', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Save the coordinates to a CSV file
df = pd.DataFrame(eye_coords, columns=['timestamp', 'left_iris_x', 'left_iris_y', 'right_iris_x', 'right_iris_y'])
df.to_csv('eye_movement_coordinates.csv', index=False)

print("Eye movement coordinates saved to eye_movement_coordinates.csv")