import cv2
import face_recognition
import numpy as np
import time
from collections import defaultdict

# Dictionary to store head tracking information
head_tracker = defaultdict(lambda: {'encodings': None, 'enter_time': None, 'last_seen': None, 'engagement_time': 0})

# Video capture
cap = cv2.VideoCapture(0)

# Head count variable
head_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Find face locations in the frame
    face_locations = face_recognition.face_locations(frame)

    # Loop through detected faces
    for face_location in face_locations:
        top, right, bottom, left = face_location

        # Get face encoding
        face_encoding = face_recognition.face_encodings(frame, [(top, right, bottom, left)])[0]

        # Check if this face encoding already exists in head_tracker
        distances = [np.linalg.norm(face_encoding - known_encoding['encodings']) for known_encoding in head_tracker.values()]

        # Set a threshold distance (you may need to adjust this)
        threshold_distance = 0.6

        if all(distance > threshold_distance for distance in distances):
            # Extract head ID from face encoding
            head_id = hash(tuple(face_encoding))

            # Check if this head is already being tracked
            if head_id not in head_tracker:
                head_tracker[head_id]['enter_time'] = time.time()

                # Increment head count
                head_count += 1

            # Store face encoding for tracking
            head_tracker[head_id]['encodings'] = face_encoding
            head_tracker[head_id]['last_seen'] = time.time()

            # Draw bounding box around the head
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    # Check if a head is no longer visible and update the engagement time
    for head_id, data in head_tracker.items():
        if data['last_seen'] is not None and time.time() - data['last_seen'] > 1.0:  # Assuming a head is not visible for more than 1 second
            data['engagement_time'] += time.time() - data['last_seen']
            data['last_seen'] = None

    # Display engagement time for each head
    for i, (head_id, data) in enumerate(head_tracker.items()):
        if data['enter_time'] is not None:
            engagement_time = data['engagement_time'] + (time.time() - data['enter_time'])
            y_position = 60 + i * 30  # Adjust the vertical position dynamically
            cv2.putText(frame, f'Head {i + 1} - Engagement Time: {int(engagement_time)}s', (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display head count at the top left of the screen
    cv2.putText(frame, f'Total Head Count: {head_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow('Head Counting', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()