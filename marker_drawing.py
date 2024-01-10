import cv2
import mediapipe as mp
import numpy as np

class GeometricShapesMatcher:
    def __init__(self, trail_length=150, min_points_for_recognition=30, min_distance_between_points=10):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.trail_length = trail_length
        self.min_points_for_recognition = min_points_for_recognition
        self.min_distance_between_points = min_distance_between_points

        # List to store the trail of hand points
        self.hand_points = []

        # List of recognized shapes
        self.shapes = ['circle', 'square', 'triangle']
        self.current_shape = None

    def recognize_shape(self):
        if len(self.hand_points) < self.min_points_for_recognition:
            return None

        # Check the minimum distance between consecutive points for shape recognition
        if any(np.linalg.norm(np.array(self.hand_points[i]) - np.array(self.hand_points[i-1])) < self.min_distance_between_points
               for i in range(1, len(self.hand_points))):
            return None

        # Convert hand points to a NumPy array
        points = np.array(self.hand_points)

        # Calculate the centroid of the hand trail
        centroid = np.mean(points, axis=0)

        # Calculate the distance from each point to the centroid
        distances = np.linalg.norm(points - centroid, axis=1)

        # Calculate the standard deviation of the distances
        std_dev = np.std(distances)

        # Check if the hand trail resembles a circle, square, or triangle based on standard deviation
        if std_dev < 15:
            return 'circle'
        elif std_dev < 30:
            return 'square'
        else:
            return 'triangle'

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            index_finger_tip = hand_landmarks.landmark[8]
            index_finger_tip_x, index_finger_tip_y = int(index_finger_tip.x * frame.shape[1]), int(index_finger_tip.y * frame.shape[0])
            hand_position = (index_finger_tip_x, index_finger_tip_y)

            self.hand_points.append(hand_position)

            # Draw the trail of hand points
            for point in self.hand_points:
                cv2.circle(frame, point, 2, (0, 255, 0), -1)

            # Recognize the shape based on the hand trail
            recognized_shape = self.recognize_shape()
            if recognized_shape:
                self.current_shape = recognized_shape
                self.hand_points = []  # Clear the hand trail when a shape is recognized

            # Draw the recognized shape
            if self.current_shape:
                cv2.putText(frame, f"Shape: {self.current_shape}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Geometric Shapes Matcher", frame)

        # Remove the oldest hand point if the trail exceeds a certain length
        if len(self.hand_points) > self.trail_length:
            self.hand_points.pop(0)

    def run(self):
        cap = cv2.VideoCapture(0)
        cv2.namedWindow("Geometric Shapes Matcher", cv2.WINDOW_NORMAL)

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            self.process_frame(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# Create an instance of the GeometricShapesMatcher class and run the application
shapes_matcher = GeometricShapesMatcher(trail_length=200, min_points_for_recognition=50, min_distance_between_points=15)
shapes_matcher.run()
