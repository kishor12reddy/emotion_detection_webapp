import numpy as np
import cv2
from tensorflow.keras.models import load_model

model = load_model('emotion_model.h5')

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

def detect_emotion(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        roi_gray = gray_frame[y:y + h, x:x + w]

        resized_frame = cv2.resize(roi_gray, (128, 128))  # Change this size

        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_GRAY2RGB)  # Ensure it's 3 channels

        normalized_frame = rgb_frame.astype("float32") / 255.0
        reshaped_frame = np.reshape(normalized_frame, (1, 128, 128, 3))  # Ensure shape is (1, 128, 128, 3)


        predictions = model.predict(reshaped_frame)
        emotion_index = np.argmax(predictions)


        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, emotion_dict[emotion_index], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return emotion_dict[emotion_index], frame
    
    return None, frame


def recommend_song(emotion):
    return song_recommendations.get(emotion, ["No song recommendation available"])


cap = cv2.VideoCapture(0)

desired_width = 800
desired_height = 600

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to capture video feed.")
        break

    emotion, frame_with_emotion = detect_emotion(frame)
    
    if emotion:
        print(f"Detected Emotion: {emotion}")
        songs = recommend_song(emotion)
        print(f"Recommended songs: {songs}")
        
        for i, song in enumerate(songs):
            cv2.putText(frame_with_emotion, song, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    frame_with_emotion = cv2.resize(frame_with_emotion, (desired_width, desired_height))
    cv2.imshow("Emotion Detector", frame_with_emotion)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
