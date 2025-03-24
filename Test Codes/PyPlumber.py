import os
import pdfplumber

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
                    print(f"Text extracted from: {file_name}")
            except Exception as e:
                print(f"Error reading {file_name}: {e}")
    return pdf_text_data

if __name__ == "__main__":
    folder_path = input("Enter the path to the folder containing PDFs: ")
    text_data = extract_text_from_pdfs(folder_path)
    for pdf_name, content in text_data.items():
        output_file = os.path.join(folder_path, f"{os.path.splitext(pdf_name)[0]}.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(content)
            print(f"Text saved to: {output_file}")
