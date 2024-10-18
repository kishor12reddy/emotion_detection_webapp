from flask import Flask, render_template, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the pre-trained emotion detection model
model = load_model('emotion_model.h5')

# Define emotion mapping and song recommendations
emotion_dict = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprise",
}

song_recommendations = {
    "happy": ["Happy - Pharrell Williams", "YouTube Link 1"],
    "sad": ["Someone Like You - Adele", "YouTube Link 2"],
    "angry": ["Killing in the Name - Rage Against the Machine", "YouTube Link 3"],
    "surprise": ["Uptown Funk - Mark Ronson", "YouTube Link 4"],
    "fear": ["Scary Monsters - Skrillex", "YouTube Link 5"],
    "neutral": ["Weightless - Marconi Union", "YouTube Link 6"],
    "disgust": ["Mad World - Gary Jules", "YouTube Link 7"],
}

# Function to detect emotion in the frame
def detect_emotion(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray_frame[y:y + h, x:x + w]
        resized_frame = cv2.resize(roi_gray, (128, 128))
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_GRAY2RGB)
        normalized_frame = rgb_frame.astype("float32") / 255.0
        reshaped_frame = np.reshape(normalized_frame, (1, 128, 128, 3))
        predictions = model.predict(reshaped_frame)
        emotion_index = np.argmax(predictions)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, emotion_dict[emotion_index], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return emotion_dict[emotion_index], frame

    return None, frame

# Generate frames from the webcam
def gen_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Could not read frame from webcam.")
            break

        emotion, frame = detect_emotion(frame)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("Error: Could not encode frame.")
            break

        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# Route to render the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to stream video frames
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
