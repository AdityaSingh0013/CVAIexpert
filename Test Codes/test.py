import os
import spacy
import re
import pdfplumber

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
        "Skills Found": skills,
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
                        text += page.extract_text()
                    pdf_text_data[file_name] = text
            except Exception as e:
                print(f"Error reading {file_name}: {e}")
    return pdf_text_data


def process_folder(folder_path, user_skills):
    results = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "r", encoding="utf-8") as file:
                cv_text = file.read()
                results[file_name] = analyze_cv(cv_text, user_skills)
    pdf_text_data = extract_text_from_pdfs(folder_path)
    for file_name, cv_text in pdf_text_data.items():
        results[file_name] = analyze_cv(cv_text, user_skills)
    return results


def calculate_score(result, required_skills):
    score = 0
    skills_found = set(result.get("Skills Found", []))
    matching_skills = skills_found.intersection(required_skills)
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


if __name__ == "__main__":
    print("Welcome to CV Analyzer!")
    user_keywords = input("Enter the keywords to search for (comma-separated): ").strip()
    skill_set = {keyword.strip().lower() for keyword in user_keywords.split(",")}
    folder_path = input("Enter the folder path containing CVs as text or PDF files: ").strip()
    if os.path.isdir(folder_path):
        analysis_results = process_folder(folder_path, skill_set)
        ranked_resumes = rank_resumes(analysis_results, skill_set)
        print("\nRanked Resumes:")
        for rank, (file_name, score) in enumerate(ranked_resumes, start=1):
            print(f"{rank}. {file_name} - Score: {score}")
    else:
        print("Invalid folder path. Please try again.")