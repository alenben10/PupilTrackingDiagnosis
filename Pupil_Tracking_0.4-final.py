# Initialization of MediaPipe Face Mesh and drawing utilities
import cv2
import mediapipe as mp
import pandas as pd
import time

# Initialize MediaPipe Face Mesh and drawing utilities
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, 
                                  max_num_faces=1, 
                                  refine_landmarks=True, 
                                  min_detection_confidence=0.5)

# Initialize video capture
cap = cv2.VideoCapture(0)
fps = 30  # Specify the desired FPS
cap.set(cv2.CAP_PROP_FPS, fps)  # Set the capture FPS

# Get the width and height of the frames
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize video writer with MP4 codec
out = cv2.VideoWriter('eye_tracking_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Initialize a list to store coordinates
eye_coords = []

# Record for 20 seconds
start_time = time.time()
record_duration = 40  # in seconds

# Define the landmark indices for the left and right irises and for head position reference
LEFT_IRIS = [474, 475, 476, 477]  # Indices for left iris
RIGHT_IRIS = [469, 470, 471, 472]  # Indices for right iris
HEAD_POSITION = [1, 33, 263, 61, 291]  # Indices for head position reference points (nose tip and outer eyes)

# Initialize variables for gaze prediction
initial_left_iris_pos = None
initial_right_iris_pos = None
initial_head_pos = None
red_dot_pos = [frame_width // 2, frame_height // 2]  # Start at the center

# Define constants for displacement multipliers and head movement threshold
DISPLACEMENT_MULTIPLIER_X = 4  # Multiplier for horizontal movement
DISPLACEMENT_MULTIPLIER_Y = 4  # Multiplier for vertical movement (adjust this value as needed)
HEAD_MOVEMENT_THRESHOLD = 50   # Threshold for head movement 

frame_counter = 0
red_dot_coords = []

while cap.isOpened() and (time.time() - start_time < record_duration):
    loop_start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    # Increment frame counter
    frame_counter += 1

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

            # Extracting the coordinates of head position reference points
            head_pos_x = sum(face_landmarks.landmark[i].x for i in HEAD_POSITION) / len(HEAD_POSITION)
            head_pos_y = sum(face_landmarks.landmark[i].y for i in HEAD_POSITION) / len(HEAD_POSITION)

            # Convert normalized coordinates to pixel coordinates
            height, width, _ = frame.shape
            left_iris_x_px = int(left_iris_x * width)
            left_iris_y_px = int(left_iris_y * height)
            right_iris_x_px = int(right_iris_x * width)
            right_iris_y_px = int(right_iris_y * height)
            head_pos_x_px = int(head_pos_x * width)
            head_pos_y_px = int(head_pos_y * height)

            # Save the coordinates along with the timestamp
            timestamp = time.time()
            eye_coords.append([timestamp, left_iris_x_px, left_iris_y_px, right_iris_x_px, right_iris_y_px, head_pos_x_px, head_pos_y_px])

            # Initialize the initial positions if they are not set
            if initial_left_iris_pos is None:
                initial_left_iris_pos = (left_iris_x_px, left_iris_y_px)
                initial_right_iris_pos = (right_iris_x_px, right_iris_y_px)
                initial_head_pos = (head_pos_x_px, head_pos_y_px)

            # Calculate the movement of the head
            head_movement_x = abs(head_pos_x_px - initial_head_pos[0])
            head_movement_y = abs(head_pos_y_px - initial_head_pos[1])

            # Check if head movement is within the threshold
            if head_movement_x < HEAD_MOVEMENT_THRESHOLD and head_movement_y < HEAD_MOVEMENT_THRESHOLD:
                # Calculate the movement of the irises relative to the initial head position
                relative_left_iris_x = left_iris_x_px - (head_pos_x_px - initial_head_pos[0])
                relative_left_iris_y = left_iris_y_px - (head_pos_y_px - initial_head_pos[1])
                relative_right_iris_x = right_iris_x_px - (head_pos_x_px - initial_head_pos[0])
                relative_right_iris_y = right_iris_y_px - (head_pos_y_px - initial_head_pos[1])

                # Calculate the average relative movement of the irises with the multipliers
                avg_relative_iris_movement_x = DISPLACEMENT_MULTIPLIER_X * ((relative_left_iris_x - initial_left_iris_pos[0]) + (relative_right_iris_x - initial_right_iris_pos[0])) / 2
                avg_relative_iris_movement_y = DISPLACEMENT_MULTIPLIER_Y * ((relative_left_iris_y - initial_left_iris_pos[1]) + (relative_right_iris_y - initial_right_iris_pos[1])) / 2

                # Update the red dot position based on the average relative iris movement
                red_dot_pos[0] += int(avg_relative_iris_movement_x)
                red_dot_pos[1] += int(avg_relative_iris_movement_y)

                # Ensure the red dot stays within the frame bounds
                red_dot_pos[0] = max(0, min(red_dot_pos[0], frame_width))
                red_dot_pos[1] = max(0, min(red_dot_pos[1], frame_height))

            # Draw the iris landmarks on the frame
            cv2.circle(frame, (left_iris_x_px, left_iris_y_px), 3, (0, 255, 0), -1)
            cv2.circle(frame, (right_iris_x_px, right_iris_y_px), 3, (0, 255, 0), -1)
            
            # Draw the red dot for gaze prediction
            cv2.circle(frame, (red_dot_pos[0], red_dot_pos[1]), 5, (0, 0, 255), -1)

            # Save the red dot coordinates along with the frame counter (flip x-coordinate)
            red_dot_coords.append([frame_counter, frame_width - red_dot_pos[0], red_dot_pos[1]])

    # Flip the frame horizontally
    flipped_frame = cv2.flip(frame, 1)

    # Add a filled rectangle as a background for the text
    cv2.rectangle(flipped_frame, (5, 5), (250, 50), (0, 0, 0), cv2.FILLED)

    # Add text to display the coordinates on the flipped frame (flip x-coordinate)
    cv2.putText(flipped_frame, f'({frame_width - red_dot_pos[0]}, {red_dot_pos[1]})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Write the flipped frame to the video file
    out.write(flipped_frame)

    # Display the frames in real time (comment so algorithm runs fast enough)
    cv2.imshow('Eye Tracking', flipped_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Measure the loop duration and print it
    loop_duration = time.time() - loop_start_time
    print(f"Frame {frame_counter} processed in {loop_duration:.4f} seconds")


# Release the video capture and writer, and close all OpenCV windows
cap.release()
out.release()
cv2.destroyAllWindows()

# Save the coordinates to a CSV file

#df = pd.DataFrame(eye_coords, columns=['timestamp', 'left_iris_x', 'left_iris_y', 'right_iris_x', 'right_iris_y', 'head_pos_x', 'head_pos_y'])
#df.to_csv('eye_movement_coordinates.csv', index=False)

df_red_dot = pd.DataFrame(red_dot_coords, columns=['frame_id', 'x', 'y'])
df_red_dot.to_csv('red_dot_coordinates.csv', index=False)

print("Eye movement coordinates saved to eye_movement_coordinates.csv")
print("Red dot coordinates saved to red_dot_coordinates.csv")
print("Video saved to eye_tracking_video.mp4")