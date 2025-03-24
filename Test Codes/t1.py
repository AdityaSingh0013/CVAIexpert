import os
import spacy
import re
import pdfplumber

nlp = spacy.load("en_core_web_sm")

def find_matching_skills(skills_found, required_skills):
    # Normalize skills to lowercase for consistent comparison
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
    #print("Extracted Phones:", phones)
    return phones if phones else ["Phone number not found"]

def extract_skills(doc, skill_set):
    found_skills = set()
    for token in doc:
        if any(skill in token.text.lower() for skill in skill_set):
            found_skills.add(token.text)
    #print("Extracted Skills:", found_skills)
    return list(found_skills) if found_skills else ["No skills found"]

def analyze_cv(cv_text, user_skills):
    doc = nlp(cv_text)
    name = extract_name(cv_text, doc)
    emails = extract_email(cv_text)
    phones = extract_phone(cv_text)
    skills = extract_skills(doc, user_skills)
    #print(f"Analyzing CV: Name: {name}, Emails: {emails}, Phones: {phones}, Skills: {skills}")
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
    print(f"Calculated score for result: {result['Name']} - {score}")
    print(matching_skills)
    print("Extracted Skills:", skills_found)
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
    folder_path = input("Enter the folder path containing CVs (PDFs and/or TXT): ").strip()
    if os.path.isdir(folder_path):
        analysis_results = process_folder(folder_path, skill_set)
        ranked_resumes = rank_resumes(analysis_results, skill_set)
        print("\nRanked Resumes:")
        for rank, (file_name, score) in enumerate(ranked_resumes, start=1):
            print(f"{rank}. {file_name} - Score: {score}")
    else:
        print("Invalid folder path. Please try again.")
