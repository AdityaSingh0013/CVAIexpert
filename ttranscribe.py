import os
import re
import moviepy as mp
import speech_recognition as sr
import spacy
import pdfplumber
import cv2
import numpy as np
from keras.models import model_from_json
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')
nlp = spacy.load("en_core_web_sm")

# Load emotion recognition model
json_file = open(r"D:\Study Stuff\Sem 5\Mini Project\spaCy\emotion recognition\emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights(r"D:\Study Stuff\Sem 5\Mini Project\spaCy\emotion recognition\emotiondetector.h5")
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)


def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0


def video_to_text(video_file, output_text_file):
    try:
        video = mp.VideoFileClip(video_file)
        audio_file = "temp_audio.wav"
        video.audio.write_audiofile(audio_file)
    except Exception as e:
        return f"Error in extracting audio: {e}"

    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
    except Exception as e:
        return f"Error in audio transcription: {e}"

    with open(output_text_file, "w") as file:
        file.write(text)

    return text


def analyze_text(text):
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    word_count = len(text.split())
    analysis_results = {
        "word_count": word_count,
        "sentiment": sentiment
    }
    return analysis_results


def find_matching_skills(skills_found, required_skills):
    normalized_skills_found = {skill.lower() for skill in skills_found}
    matching_skills = {
        req_skill for req_skill in required_skills
        if any(req_skill in skill for skill in normalized_skills_found)
    }
    return matching_skills


def extract_name(cv_text, doc):
    name_pattern = re.search(r"Name[:\-]?\s*(.+)", cv_text, re.IGNORECASE)
    if name_pattern:
        return name_pattern.group(1).strip()
    first_line = cv_text.splitlines()[0]
    if all(word.isalpha() or word == '.' for word in first_line.split()):
        return first_line.strip()
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            if all(word.isalpha() for word in ent.text.split()) and len(ent.text.split()) > 1:
                return ent.text
    return "Name not found"


def extract_email(text):
    email_regex = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    emails = re.findall(email_regex, text)
    return emails if emails else ["Email not found"]


def extract_phone(text):
    phone_regex = r'\b\d{10,13}\b'
    phones = re.findall(phone_regex, text)
    return phones if phones else ["Phone number not found"]


def extract_skills(doc, skill_set):
    found_skills = set()
    for token in doc:
        if any(skill in token.text.lower() for skill in skill_set):
            found_skills.add(token.text)
    return list(found_skills) if found_skills else ["No skills found"]


def analyze_cv(cv_text, user_skills):
    doc = nlp(cv_text)
    name = extract_name(cv_text, doc)
    emails = extract_email(cv_text)
    phones = extract_phone(cv_text)
    skills = extract_skills(doc, user_skills)
    return {
        "Name": name,
        "Emails": emails,
        "Phone Numbers": phones,
        "Skills Found": skills
    }


def extract_text_from_pdfs(folder_path):
    pdf_text_data = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.pdf'):
            pdf_path = os.path.join(folder_path, file_name)
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    text = ""
                    for page in pdf.pages:
                        extracted_text = page.extract_text()
                        if extracted_text:
                            text += extracted_text
                        else:
                            print(f"Warning: No text extracted from a page in {file_name}")
                    pdf_text_data[file_name] = text
            except Exception as e:
                print(f"Error reading {file_name}: {e}")
    return pdf_text_data


def process_folder(folder_path, user_skills):
    results = {}
    processed_files = set()
    for file_name in os.listdir(folder_path):
        base_name = os.path.splitext(file_name)[0]
        if base_name in processed_files:
            continue
        processed_files.add(base_name)
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "r", encoding="utf-8") as file:
                cv_text = file.read()
                results[file_name] = analyze_cv(cv_text, user_skills)
    pdf_text_data = extract_text_from_pdfs(folder_path)
    for pdf_name, content in pdf_text_data.items():
        txt_file_path = os.path.join(folder_path, f"{os.path.splitext(pdf_name)[0]}.txt")
        with open(txt_file_path, "w", encoding="utf-8") as f:
            f.write(content)
        results[pdf_name] = analyze_cv(content, user_skills)
    return results


def calculate_score(result, required_skills):
    score = 0
    skills_found = set(result.get("Skills Found", []))
    matching_skills = find_matching_skills(skills_found, required_skills)
    score += len(matching_skills) * 10
    if result.get("Name") and result["Name"] != "Name not found":
        score += 5
    if result.get("Emails") and result["Emails"][0] != "Email not found":
        score += 5
    if result.get("Phone Numbers") and result["Phone Numbers"][0] != "Phone number not found":
        score += 5
    return score


def rank_resumes(analysis_results, required_skills):
    resume_scores = {}
    for file_name, result in analysis_results.items():
        score = calculate_score(result, required_skills)
        resume_scores[file_name] = score
    ranked_resumes = sorted(resume_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_resumes


def recognize_emotions(video_file):
    webcam = cv2.VideoCapture(video_file)
    labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
    text = []

    while True:
        ret, im = webcam.read()
        if not ret:
            break
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(im, 1.3, 5)
        try:
            for (p, q, r, s) in faces:
                image = gray[q:q+s, p:p+r]
                cv2.rectangle(im, (p, q), (p+r, q+s), (255, 0, 0), 2)
                image = cv2.resize(image, (48, 48))
                img = extract_features(image)
                pred = model.predict(img)
                prediction_label = labels[pred.argmax()]
                cv2.putText(im, prediction_label, (p-10, q-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))
                text.append(prediction_label)
            cv2.imshow("Output", im)
            if cv2.waitKey(27) & 0xFF == ord('q'):
                break
        except cv2.error:
            pass
    webcam.release()
    cv2.destroyAllWindows()
    return text


def display_ranked_resumes(ranked_resumes):
    print("\nRanked Resumes:")
    for rank, (file_name, score) in enumerate(ranked_resumes, start=1):
        print(f"{rank}. {file_name} - Score: {score}")


def display_emotion_results(emotions):
    from collections import Counter
    emotion_counts = Counter(emotions)
    print("\nDetected Emotions:")
    for emotion, count in emotion_counts.items():
        print(f"{emotion}: {count} occurrences")


if __name__ == "__main__":
    try:
        video_file = input("Enter the path to the video file: ").strip()
        if not os.path.exists(video_file):
            print("Invalid video file path.")
            exit()

        output_text_file = "transcription.txt"
        transcribed_text = video_to_text(video_file, output_text_file)

        if "Error" not in transcribed_text:
            print("Transcription successful! Analyzing text...")
            analysis_results = analyze_text(transcribed_text)
            print("Analysis Results:", analysis_results)

            user_keywords = input("Enter the keywords to search for (comma-separated): ").strip()
            skill_set = {keyword.strip().lower() for keyword in user_keywords.split(",")}
            folder_path = input("Enter the folder path containing CVs (PDFs and/or TXT): ").strip()

            if os.path.isdir(folder_path):
                analysis_results = process_folder(folder_path, skill_set)
                ranked_resumes = rank_resumes(analysis_results, skill_set)
                display_ranked_resumes(ranked_resumes)
            else:
                print("Invalid folder path. Please try again.")

            # Recognize emotions from the video
            emotions = recognize_emotions(video_file)
            display_emotion_results(emotions)
        else:
            print(transcribed_text)
    except Exception as e:
        print(f"An error occurred: {e}")
