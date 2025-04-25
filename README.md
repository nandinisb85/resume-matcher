# 🧠 AI-Powered Resume Ranker & Skills Gap Analyzer

This project is a web-based application that helps job seekers instantly evaluate how well their resume matches a given job description. It uses both **keyword-based** and **semantic skill matching** to provide a comprehensive fit score and skill analysis.

---

## 📌 Features

- 📎 Upload your resume in **PDF or DOCX** format
- 📝 Paste any job description into the interface
- 🎯 Get a **resume-job match score**
- ✅ See both **exact matched** and **AI-inferred skills**
- ❌ Identify **missing or unmatched skills**
- 👤 Auto-extract key resume fields: Name, Location, Summary, Years of Experience
- 📊 Visualize matched vs. unmatched skills with simple charts

---

## 🛠️ Tech Stack

- **Frontend & App Framework**: [Streamlit](https://streamlit.io/)
- **NLP & Semantic Search**:
  - [Hugging Face Transformers](https://huggingface.co/)
  - [sentence-transformers](https://www.sbert.net/)
- **File Parsing**:
  - `pdfplumber` for PDFs
  - `docx2txt` for DOCX files
- **Text Processing**: Regex, Python string utilities
- **Visualization**: Matplotlib

---

## 🧠 How It Works

1. **Resume Parsing**: Extracts text from uploaded resume files using `pdfplumber` and `docx2txt`.
2. **Job Description Parsing**: Simple text input for the target job description.
3. **Skill Matching**:
   - Uses cosine similarity with `sentence-transformers` to semantically compare skills.
   - Matches keywords from a predefined skill list and computes overlaps.
4. **Scoring**:
   - Calculates a match score based on overlapping and semantically similar skills.
   - Highlights missing skills that are important for the job.

---

## 📂 Folder Structure

# resume-matcher
