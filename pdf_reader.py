from pypdf import PdfReader

pdf_path = "data/certification/GrantofLicence-Guidelines-04Feb2025.pdf"

reader = PdfReader(pdf_path)

text = ""

for page in reader.pages:
    page_text = page.extract_text()
    if page_text:
        text += page_text + "\n"

print("Total pages:", len(reader.pages))
print("\n--- EXTRACTED TEXT ---\n")
print(text[:5000])