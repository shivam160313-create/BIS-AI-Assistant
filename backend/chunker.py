from pypdf import PdfReader

pdf_path = "data/certification/GrantofLicence-Guidelines-04Feb2025.pdf"

reader = PdfReader(pdf_path)

text = ""

for page in reader.pages:
    page_text = page.extract_text()
    if page_text:
        text += page_text + "\n"

# Split text into chunks
chunk_size = 1000
overlap = 200

chunks = []

start = 0

while start < len(text):
    end = start + chunk_size
    chunk = text[start:end]
    chunks.append(chunk)

    start = end - overlap

print("Total chunks:", len(chunks))

print("\n--- FIRST CHUNK ---\n")
print(chunks[0])