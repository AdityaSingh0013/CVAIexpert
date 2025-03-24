import os
import re
import cv2
import numpy as np
import pdfplumber
import speech_recognition as sr
import spacy
from collections import Counter
from nltk.sentiment import SentimentIntensityAnalyzer
from keras.models import model_from_json
from moviepy.editor import VideoFileClip
from nltk import download

# Setup
download('vader_lexicon')
nlp = spacy.load("en_core_web_sm")

# Load emotion recognition model
with open("emotiondetector.json", "r") as json_file:
    model = model_from_json(json_file.read())
model.load_weights("emotiondetector.h5")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Utilities
def extract_features(image):
    return image.reshape(1, 48, 48, 1) / 255.0

def extract_text_from_video(video_file):
    try:
        video = VideoFileClip(video_file)
        audio_file = "temp_audio.wav"
        video.audio.write_audiofile(audio_file)
        with sr.AudioFile(audio_file) as source:
            return sr.Recognizer().recognize_google(sr.Recognizer().record(source))
    except Exception as e:
        return f"Error processing video: {e}"

def analyze_text(text):
    sia = SentimentIntensityAnalyzer()
    return {"word_count": len(text.split()), "sentiment": sia.polarity_scores(text)}

def process_resume(cv_text, required_skills):
    doc = nlp(cv_text)
    skills_found = [token.text for token in doc if token.text.lower() in required_skills]
    return {
        "Name": next((ent.text for ent in doc.ents if ent.label_ == "PERSON"), "Name not found"),
        "Emails": re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', cv_text) or ["Email not found"],
        "Phone Numbers": re.findall(r'\b\d{10,13}\b', cv_text) or ["Phone number not found"],
        "Skills Found": skills_found or ["No skills found"]
    }

def extract_text_from_pdfs(folder_path):
    pdf_text = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pdf"):
            try:
                with pdfplumber.open(os.path.join(folder_path, file_name)) as pdf:
                    pdf_text[file_name] = "".join([page.extract_text() or "" for page in pdf.pages])
            except Exception as e:
                print(f"Error reading {file_name}: {e}")
    return pdf_text

def process_folder(folder_path, required_skills):
    resumes = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith((".txt", ".pdf")):
            file_path = os.path.join(folder_path, file_name)
            if file_name.endswith(".pdf"):
                text = extract_text_from_pdfs(folder_path).get(file_name, "")
            else:
                with open(file_path, "r", encoding="utf-8") as file:
                    text = file.read()
            resumes[file_name] = process_resume(text, required_skills)
    return resumes

def rank_resumes(resumes, required_skills):
    def calculate_score(resume):
        skills = set(resume.get("Skills Found", []))
        return len(skills.intersection(required_skills)) * 10 + sum(bool(resume.get(key)) for key in ["Name", "Emails", "Phone Numbers"]) * 5

    return sorted(resumes.items(), key=lambda x: calculate_score(x[1]), reverse=True)

def detect_emotions(video_file):
    emotions = []
    cap = cv2.VideoCapture(video_file)
    while True:
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            face = cv2.resize(gray[y:y+h, x:x+w], (48, 48))
            prediction = model.predict(extract_features(face))
            emotions.append(labels[np.argmax(prediction)])
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    return Counter(emotions)

# Main Flow
if __name__ == "__main__":
    video_path = input("Enter the video file path: ").strip()
    transcript = extract_text_from_video(video_path)
    if "Error" not in transcript:
        print("Text analysis:", analyze_text(transcript))

        resume_folder = input("Enter the resume folder path: ").strip()
        keywords = input("Enter skills (comma-separated): ").lower().split(", ")
        resume_analysis = process_folder(resume_folder, set(keywords))
        ranked_resumes = rank_resumes(resume_analysis, set(keywords))
        print("Ranked Resumes:", ranked_resumes)

        emotions = detect_emotions(video_path)
        print("Emotion Counts:", emotions)
    else:
        print(transcript)
