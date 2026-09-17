Put your medical PDF and MedQuAD-style CSV files in this folder.

For example:

data/
  medical_book.pdf
  health_guide.pdf
  medDataset_processed.csv

The ingest.py program automatically finds every file ending in .pdf or .csv here.
CSV files must contain Question and Answer columns; qtype is optional.
After adding or changing a PDF or CSV, run this command from the project folder:

python ingest.py

Use reliable, legally obtained medical documents. Scanned/image-only PDFs may
need OCR (text recognition) before this simple project can read their text.
