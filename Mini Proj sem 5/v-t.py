import moviepy.editor as mp
import speech_recognition as sr
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk


nltk.download('vader_lexicon')


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


video_file = r"C:\Users\rishi\OneDrive\Desktop\Tutorials\Mini Proj sem 5\resume.mp4"

output_text_file = "transcription.txt"

transcribed_text = video_to_text(video_file, output_text_file)
if "Error" not in transcribed_text:
    print("Transcription successful! Analyzing text...")
    analysis_results = analyze_text(transcribed_text)
    print("Analysis Results:")
    print(analysis_results)
else:
    print(transcribed_text)
