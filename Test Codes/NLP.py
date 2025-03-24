import os
import spacy
import re

nlp = spacy.load("en_core_web_sm")

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
        if token.text.lower() in skill_set:
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

def process_folder(folder_path, user_skills):
    results = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "r", encoding="utf-8") as file:
                cv_text = file.read()
                results[file_name] = analyze_cv(cv_text, user_skills)
    return results

if __name__ == "__main__":
    print("Welcome to CV Analyzer!")
    user_keywords = input("Enter the keywords to search for (comma-separated): ").strip()
    skill_set = {keyword.strip().lower() for keyword in user_keywords.split(",")}
    folder_path = input("Enter the folder path containing CVs as text files: ").strip()
    if os.path.isdir(folder_path):
        analysis_results = process_folder(folder_path, skill_set)
        print("\nCV Analysis Results:")
        for file_name, result in analysis_results.items():
            print(f"\nFile: {file_name}")
            for key, value in result.items():
                print(f"{key}: {value}")
    else:
        print("Invalid folder path. Please try again.")
