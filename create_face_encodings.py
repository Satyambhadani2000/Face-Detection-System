# database.py
import face_recognition
import os
import pickle

def create_face_encodings(known_faces_dir):
    known_face_encodings = []
    known_face_names = []

    for filename in os.listdir(known_faces_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(known_faces_dir, filename)
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)

            if encoding:
                known_face_encodings.append(encoding[0])
                known_face_names.append(os.path.splitext(filename)[0])  # Use the filename as the name

    # Save encodings and names to a file
    with open("face_encodings.pkl", "wb") as f:
        pickle.dump((known_face_encodings, known_face_names), f)

if __name__ == "__main__":
    create_face_encodings("known_faces")
