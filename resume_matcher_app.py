# resume_matcher_app.py
import streamlit as st
import pdfplumber
import docx2txt
import io
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# 🔍 Master Skill Set (expanded for tech roles)
skill_set = {
    'python', 'java', 'c++', 'c', 'javascript', 'html', 'css', 'react', 'node.js',
    'angular', 'flask', 'django', 'spring boot', 'sql', 'mysql', 'postgresql',
    'mongodb', 'firebase', 'pandas', 'numpy', 'scikit-learn', 'tensorflow',
    'pytorch', 'mlops', 'docker', 'kubernetes', 'aws', 'gcp', 'azure', 'airflow',
    'bigquery', 'snowflake', 'data analysis', 'etl', 'data engineering',
    'data visualization', 'tableau', 'power bi', 'jira', 'git', 'ci/cd', 'api',
    'rest api', 'bash', 'linux', 'streamlit', 'langchain', 'hugging face',
    'prompt engineering', 'llm', 'llms', 'chatgpt', 'transformers'
} - ENGLISH_STOP_WORDS

# 📄 Extract text from uploaded file
def extract_text(file):
    if file.name.endswith(".pdf"):
        with pdfplumber.open(file) as pdf:
            return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
    elif file.name.endswith(".docx"):
        return docx2txt.process(file)
    elif file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    else:
        return ""

# 🧠 Skill extraction via regex
def extract_skills(text, skills):
    text = text.lower()
    return {skill for skill in skills if re.search(rf"\b{re.escape(skill.lower())}\b", text)}

# 🌟 Streamlit UI
st.set_page_config(page_title="Resume Ranker & Skills Gap Analyzer", layout="centered")
st.title("📄 Resume Ranker + Skills Gap Analyzer")
st.caption("Built by Nandini • Powered by NLP + Streamlit 💡")

resume_file = st.file_uploader("📎 Upload your Resume (PDF/DOCX/TXT)", type=["pdf", "docx", "txt"])
jd_text = st.text_area("📝 Paste the Job Description here")

if resume_file and jd_text:
    resume_text = extract_text(resume_file)
    
    resume_skills = extract_skills(resume_text, skill_set)
    jd_skills = extract_skills(jd_text, skill_set)
    
    matched = resume_skills & jd_skills
    missing = jd_skills - resume_skills
    match_score = round(len(matched) / max(len(jd_skills), 1) * 100, 2)

    st.metric(label="🎯 Resume Match Score", value=f"{match_score}%", delta=None)
    st.success(f"✅ Matched Skills ({len(matched)}): {', '.join(sorted(matched))}")
    st.warning(f"❌ Missing Skills ({len(missing)}): {', '.join(sorted(missing))}")

    # 📊 Visual Breakdown
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].bar(["Matched", "Missing"], [len(matched), len(missing)], color=["#4CAF50", "#F44336"])
    ax[0].set_title("Skill Match Breakdown")
    ax[1].pie([len(matched), len(missing)], labels=["Matched", "Missing"],
              colors=["#4CAF50", "#F44336"], autopct='%1.1f%%', startangle=140)
    ax[1].set_title("Skill Coverage")
    st.pyplot(fig)
