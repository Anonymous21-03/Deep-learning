import cv2 as cv
import mediapipe as mp
from deepface import DeepFace
import numpy as np

# Initialize the webcam
cap = cv.VideoCapture(0)

# Initialize MediaPipe Face Mesh and Drawing Utils
mp_facemesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Configure the FaceMesh module and drawing specifications
face_mesh = mp_facemesh.FaceMesh(max_num_faces=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
draw_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if not results.multi_face_landmarks:
        cv.putText(frame, "No Face Detected", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    if results.multi_face_landmarks:
        for lm in results.multi_face_landmarks:
            # Draw the face mesh landmarks
            mp_drawing.draw_landmarks(
                frame, lm, mp_facemesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=draw_spec,
                connection_drawing_spec=draw_spec
            )
            
            # Convert landmarks to bounding box
            h, w, _ = frame.shape
            face_coords = [(int(point.x * w), int(point.y * h)) for point in lm.landmark]
            x_min = max(0, min([coord[0] for coord in face_coords]))
            y_min = max(0, min([coord[1] for coord in face_coords]))
            x_max = min(w, max([coord[0] for coord in face_coords]))
            y_max = min(h, max([coord[1] for coord in face_coords]))
            
            # Crop the face ROI
            face_roi = frame[y_min:y_max, x_min:x_max]

            # Resize and preprocess face ROI for DeepFace
            try:
                if face_roi.size > 0:
                    face_roi_resized = cv.resize(face_roi, (224, 224))  # DeepFace often expects 224x224 input
                    face_roi_rgb = cv.cvtColor(face_roi_resized, cv.COLOR_BGR2RGB)  # Convert to RGB if needed
                    
                    analysis = DeepFace.analyze(face_roi_rgb, actions=['emotion'], enforce_detection=False)
                    emotion = analysis[0]['dominant_emotion']
                    cv.putText(frame, f'Emotion: {emotion}', (x_min, y_min-10), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            except Exception as e:
                # Print the exception details for debugging
                print(f"Error during emotion analysis: {e}")
                cv.putText(frame, "Error in Emotion Analysis", (10, 90), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv.imshow("Webcam", frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
