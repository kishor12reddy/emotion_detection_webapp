from flask import Flask, render_template, Response, jsonify  # Import jsonify
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
    "happy": ["Happy - Pharrell Williams",
        "Walking on Sunshine - Katrina and the Waves",
        "Can't Stop the Feeling! - Justin Timberlake",
        "Good Vibrations - The Beach Boys",
        "Uptown Funk - Mark Ronson feat. Bruno Mars",
        "Shake It Off - Taylor Swift"],
    "sad": [ "Someone Like You - Adele",
        "Fix You - Coldplay",
        "The Night We Met - Lord Huron",
        "Tears Dry on Their Own - Amy Winehouse",
        "Hallelujah - Jeff Buckley",
        "Back to December - Taylor Swift"],
    "angry": [  "Killing in the Name - Rage Against the Machine",
        "Break Stuff - Limp Bizkit",
        "F**k You - CeeLo Green",
        "Bodies - Drowning Pool",
        "Walk - Pantera",
        "Given Up - Linkin Park"],
    "surprise": [ "Uptown Funk - Mark Ronson",
        "Shake It Off - Taylor Swift",
        "I Gotta Feeling - The Black Eyed Peas",
        "Party in the USA - Miley Cyrus",
        "Get Lucky - Daft Punk",
        "Shut Up and Dance - WALK THE MOON"],
    "fear": [ "Scary Monsters - Skrillex",
        "Disturbia - Rihanna",
        "Monster - Imagine Dragons",
        "Bury a Friend - Billie Eilish",
        "Creep - Radiohead",
        "Run Boy Run - Woodkid"],
    "neutral": ["Weightless - Marconi Union",
        "Blue Monday - New Order",
        "The Sound of Silence - Simon & Garfunkel",
        "Chill Bill - Shreveport - Funky P - OMI",
        "Sunset Lover - Petit Biscuit",
        "Ocean Eyes - Billie Eilish"],
    "disgust": [ "Mad World - Gary Jules",
        "Boulevard of Broken Dreams - Green Day",
        "Cough Syrup - Young the Giant",
        "Stressed Out - Twenty One Pilots",
        "Irreplaceable - Beyoncé",
        "Take a Bow - Rihanna"],
}

# Variable to store the latest emotion
latest_emotion = 'unknown'

# Function to detect emotion in the frame
def detect_emotion(frame):
    global latest_emotion  # Use the global variable to store the latest emotion
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

        # Update the latest emotion variable
        latest_emotion = emotion_dict[emotion_index]
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, latest_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return latest_emotion, frame

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

# New route to get the latest detected emotion
@app.route('/get_emotion')
def get_emotion():
    return jsonify({'emotion': latest_emotion})  # Return the latest emotion as JSON

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
