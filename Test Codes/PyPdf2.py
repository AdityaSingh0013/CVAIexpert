import PyPDF2

def pdf_to_text(PASSPORT application Form.pdf):
    with open(PASSPORT application Form.pdf, 'rb') as f:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text

pdf_text = pdf_to_text('your_file.pdf')
with open('output.txt', 'w') as text_file:
    text_file.write(pdf_text)
    