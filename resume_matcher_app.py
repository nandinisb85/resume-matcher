import streamlit as st
import pdfplumber
import docx2txt
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sentence_transformers import SentenceTransformer, util

# --- Skill Set ---
semantic_skill_list = [
    'python', 'sql', 'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch',
    'mlops', 'docker', 'kubernetes', 'airflow', 'machine learning',
    'data visualization', 'streamlit', 'transformers', 'hugging face', 'api',
    'power bi', 'tableau', 'jupyter', 'bash', 'java', 'javascript',
    'html', 'css', 'react', 'node.js', 'typescript', 'c++', 'c#', 'git', 'github',
    'aws', 'azure', 'google cloud platform (gcp)', 'rest apis', 'graphql',
    'agile', 'scrum', 'linux', 'jenkins', 'terraform', 'ci/cd', 'json',
    'nosql', 'mongodb', 'postgresql', 'mysql', 'data structures', 'algorithms',
    'object-oriented programming (oop)', 'microservices', 'unit testing',
    'integration testing', 'devops', 'artificial intelligence', 'data analysis',
    'system design', 'cybersecurity', 'networking', 'troubleshooting',
    'communication skills', 'problem solving', 'time management', 'team collaboration'
] - ENGLISH_STOP_WORDS

# --- Functions ---
def extract_text(file):
    if file.name.endswith(".pdf"):
        with pdfplumber.open(file) as pdf:
            return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
    elif file.name.endswith(".docx"):
        return docx2txt.process(file)
    elif file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    return ""

def extract_skills(text, skill_list):
    text = text.lower()
    return {skill for skill in skill_list if re.search(rf"\b{re.escape(skill.lower())}\b", text)}

# --- UI ---
st.set_page_config(page_title="Resume Ranker + Skills Gap Analyzer", layout="centered")
st.title("📄 Resume Ranker + Skills Gap Analyzer")
st.caption("Built by Nandini • Enhanced with Regex + Semantic Matching 💡")

resume_file = st.file_uploader("📎 Upload your Resume (PDF/DOCX/TXT)", type=["pdf", "docx", "txt"])
jd_text = st.text_area("📝 Paste the Job Description here")

if resume_file and jd_text:
    # Extract text from resume
    resume_text = extract_text(resume_file)

    # Regex-based match
    regex_matched = extract_skills(resume_text, semantic_skill_list)
    jd_skills = extract_skills(jd_text, semantic_skill_list)
    matched = regex_matched & jd_skills
    missing = jd_skills - regex_matched
    match_score = round(len(matched) / max(len(jd_skills), 1) * 100, 2)

    # Semantic Match
    model = SentenceTransformer('all-MiniLM-L6-v2')
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)
    skill_embeddings = model.encode(semantic_skill_list, convert_to_tensor=True)
    cos_scores = util.cos_sim(resume_embedding, skill_embeddings)[0]
    semantic_matched = [
        semantic_skill_list[i]
        for i, score in enumerate(cos_scores)
        if score > 0.4
    ]

    # Combine matches
    combined_skills = set(regex_matched) | set(semantic_matched)
    tagged_skills = {
        skill: ("✅ Exact Match" if skill in regex_matched else "🧠 AI Detected")
        for skill in combined_skills
    }

    # --- Display ---
    st.metric(label="🎯 Resume Match Score", value=f"{match_score}%")
    st.success(f"✅ Matched Skills ({len(matched)}): {', '.join(sorted(matched))}")
    st.warning(f"❌ Missing Skills ({len(missing)}): {', '.join(sorted(missing))}")

    st.markdown("### 🧠 Matched Skills (Keyword + Semantic)")
    for skill, label in tagged_skills.items():
        st.markdown(
            f"<span style='color:white;background-color:#4CAF50;padding:5px 12px;border-radius:12px;margin-right:6px;'>{skill}</span> <span style='font-size:12px;color:#888;'>({label})</span>",
            unsafe_allow_html=True
        )

    # --- Charts ---
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].bar(["Matched", "Missing"], [len(matched), len(missing)], color=["#4CAF50", "#F44336"])
    ax[0].set_title("Skill Match Breakdown")
    ax[1].pie([len(matched), len(missing)], labels=["Matched", "Missing"],
              colors=["#4CAF50", "#F44336"], autopct='%1.1f%%', startangle=140)
    ax[1].set_title("Skill Coverage")
    st.pyplot(fig)
