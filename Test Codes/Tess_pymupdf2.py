import os
import fitz  # PyMuPDF
from pytesseract import image_to_string
from PIL import Image
import io

TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'

def extract_text_from_pdfs(folder_path):
    pdf_text_data = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.pdf'):
            pdf_path = os.path.join(folder_path, file_name)
            text = ""
            try:
                doc = fitz.open(pdf_path)
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    extracted_text = page.get_text()
                    if extracted_text.strip():
                        text += extracted_text
                    else:
                        pix = page.get_pixmap()
                        img = Image.open(io.BytesIO(pix.tobytes("png")))
                        text += image_to_string(img)
                doc.close()
                pdf_text_data[file_name] = text
                print(f"Text extracted from: {file_name}")
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
    return pdf_text_data

if __name__ == "__main__":
    from pytesseract import pytesseract
    pytesseract.tesseract_cmd = TESSERACT_PATH
    folder_path = input("Enter the path to the folder containing PDFs: ")
    text_data = extract_text_from_pdfs(folder_path)
    for pdf_name, content in text_data.items():
        output_file = os.path.join(folder_path, f"{os.path.splitext(pdf_name)[0]}.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(content)
            print(f"Text saved to: {output_file}")
sd