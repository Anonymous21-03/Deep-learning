import cv2 as cv
import mediapipe as mp
from deepface import DeepFace

cap = cv.VideoCapture("stress.mp4")
mpdraw = mp.solutions.drawing_utils
mpfacemesh = mp.solutions.face_mesh
facemesh = mpfacemesh.FaceMesh(max_num_faces=2)
drawspec = mpdraw.DrawingSpec(thickness=1, circle_radius=2, color=(0, 255, 0))

while cap.isOpened():
    ret, frame = cap.read()
    frame=cv.resize(frame, (500,500))
    
    if not ret:
        print("No video loaded or end of video reached")
        break
    
    rgbframe = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    
    results = facemesh.process(rgbframe)
    
    if results.multi_face_landmarks:
        
        for lm in results.multi_face_landmarks:
            # mpdraw.draw_landmarks(frame, lm, mpfacemesh.FACEMESH_CONTOURS, drawspec)
            
            h, w, _ = frame.shape
            face_coords = [(int(point.x * w), int(point.y * h)) for point in lm.landmark]
            x_min = max(0, min([coord[0] for coord in face_coords]))
            y_min = max(0, min([coord[1] for coord in face_coords]))
            x_max = min(w, max([coord[0] for coord in face_coords]))
            y_max = min(h, max([coord[1] for coord in face_coords]))
            face_roi = frame[y_min:y_max, x_min:x_max]
            
            cv.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 4)
            
            try:
                if face_roi.size:
                    analyse = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                    # print("Analyse Dictionary:", analyse)  # Debug: Print the full structure of the analyse dictionary
                    
                    if isinstance(analyse, list) and len(analyse) > 0:
                        # Access the first item in the list
                        result = analyse[0]
                        if 'dominant_emotion' in result:
                            emo = result['dominant_emotion']
                            cv.putText(frame, f'Emotion: {emo}', (x_min, y_min - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        # else:
                        #     print("Key 'dominant_emotion' not found in result")
                    else:
                        print("Analyse did not return a list or list is empty")
            except Exception as e:
                # print(f"Emotion detection error: {e}")
                cv.putText(frame, "Error in emo", (10, 90), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
    cv.imshow("Video", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
