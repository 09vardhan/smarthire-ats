import streamlit as st
import pandas as pd
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from datetime import datetime
import urllib.parse

# ----------------- Streamlit Page Config -----------------
st.set_page_config(
    page_title="SmartHire | Standalone ATS & Talent Intelligence",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- Modern SaaS Styling -----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .hero-header {
        background: linear-gradient(135deg, #0F2027 0%, #203A43 50%, #2C5364 100%);
        padding: 24px 30px;
        border-radius: 12px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.06);
    }
    .hero-header h1 {
        margin: 0;
        font-size: 26px;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .hero-header p {
        margin: 6px 0 0 0;
        color: #B0BEC5;
        font-size: 13.5px;
    }

    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.03);
    }
    .metric-label {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #64748B;
        margin-bottom: 4px;
    }
    .metric-val {
        font-size: 24px;
        font-weight: 700;
        color: #0F172A;
    }

    .badge-shortlisted {
        background-color: #ECFDF5;
        color: #047857;
        font-weight: 600;
        padding: 3px 9px;
        border-radius: 16px;
        font-size: 12px;
        display: inline-block;
        border: 1px solid #A7F3D0;
    }
    .badge-second-look {
        background-color: #FFFBEB;
        color: #B45309;
        font-weight: 600;
        padding: 3px 9px;
        border-radius: 16px;
        font-size: 12px;
        display: inline-block;
        border: 1px solid #FDE68A;
    }
    .badge-disqualified {
        background-color: #FEF2F2;
        color: #B91C1C;
        font-weight: 600;
        padding: 3px 9px;
        border-radius: 16px;
        font-size: 12px;
        display: inline-block;
        border: 1px solid #FECACA;
    }

    .skill-tag {
        background: #F1F5F9;
        color: #334155;
        font-size: 11px;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 6px;
        display: inline-block;
        margin: 2px 4px 2px 0px;
        border: 1px solid #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- Banner -----------------
st.markdown("""
<div class="hero-header">
    <h1>💼 SmartHire ATS & Talent Intelligence Engine</h1>
    <p>Privacy-first candidate screening, bias-audited ranking algorithms, and structured talent pipeline analytics.</p>
</div>
""", unsafe_allow_html=True)

# ----------------- NLP & Parsing Helpers -----------------
STOPWORDS = set([
    "the", "and", "or", "to", "in", "a", "an", "is", "are", "with", "for", "of", "on", "at",
    "by", "from", "as", "be", "this", "that", "it", "we", "you", "our", "will", "can",
    "team", "role", "work", "looking", "candidate", "responsibilities", "responsible",
    "requirements", "qualifications", "preferred", "must", "have", "ability", "skills",
    "years", "plus", "strong", "good", "knowledge", "including", "across", "other", "etc",
    "opportunity", "equal", "status", "company", "benefits", "join", "apply", "job",
    "description", "position", "seeking", "ideal", "proficient", "familiarity", "proven"
])

def extract_pdf_text(uploaded_file):
    try:
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + " "
        return text.strip()
    except Exception:
        return ""

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def extract_candidate_email(text):
    match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    return match.group(0) if match else "Not Detected"

def redact_pii(text):
    text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[REDACTED_EMAIL]', text)
    text = re.sub(r'\+?\d[\d -]{8,12}\d', '[REDACTED_PHONE]', text)
    text = re.sub(r'http\S+|www\.\S+', '[REDACTED_URL]', text)
    return text

def extract_jd_keywords(jd_text, top_n=25):
    cleaned = clean_text(jd_text)
    words = [w for w in cleaned.split() if w not in STOPWORDS and len(w) > 2]
    if not words:
        return []
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=top_n)
    try:
        vec.fit([" ".join(words)])
        return list(vec.get_feature_names_out())
    except Exception:
        return list(set(words))[:top_n]

def parse_manual_inputs(input_string):
    if not input_string:
        return []
    items = [clean_text(item) for item in input_string.split(",") if clean_text(item)]
    return list(dict.fromkeys(items))

def score_candidate(jd_text, resume_text, mandatory_gates, target_competencies):
    cleaned_jd = clean_text(jd_text)
    cleaned_resume = clean_text(resume_text)

    if not cleaned_resume:
        return 0.0, 0.0, 0.0, [], False, ["Empty resume text"]

    # 1. Mandatory Gate Verification
    missing_gates = []
    for gate in mandatory_gates:
        pattern = r'\b' + re.escape(clean_text(gate)) + r'\b'
        if not re.search(pattern, cleaned_resume):
            missing_gates.append(gate)
    passed_mandatory = len(missing_gates) == 0

    # 2. Competency Match (60% weight)
    matched_skills = []
    for skill in target_competencies:
        pattern = r'\b' + re.escape(clean_text(skill)) + r'\b'
        if re.search(pattern, cleaned_resume):
            matched_skills.append(skill)
    skill_score = (len(matched_skills) / len(target_competencies) * 100) if target_competencies else 0.0

    # 3. Contextual Fit (40% weight - TF-IDF Cosine Similarity)
    tfidf = TfidfVectorizer(stop_words='english', sublinear_tf=True, ngram_range=(1, 2))
    matrix = tfidf.fit_transform([cleaned_jd, cleaned_resume])
    context_score = float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0]) * 100

    composite_score = round((skill_score * 0.60) + (context_score * 0.40), 1)
    return composite_score, round(skill_score, 1), round(context_score, 1), matched_skills, passed_mandatory, missing_gates

def generate_candidate_email(name, status, matched_skills, missing_gates, role_title="the position"):
    clean_name = name.split(".")[0].replace("_", " ").title()
    skills_text = ", ".join(matched_skills[:4]) if matched_skills else "your background and domain experience"

    if "SHORTLISTED" in status:
        subject = f"Interview Invitation: {role_title} with Our Team"
        body = f"""Hi {clean_name},

Thank you for your application for {role_title}! We reviewed your profile and were impressed by your demonstrated background, particularly your strengths in {skills_text}.

We would like to invite you to an introductory interview round to discuss your experience, learn more about your career goals, and share more about upcoming projects on our team.

Please reply with 2-3 date and time windows that work best for you this week.

Looking forward to speaking with you!

Best regards,
Talent Acquisition Team"""
    elif "Second-Look" in status:
        subject = f"Application Update: Next Steps for {role_title}"
        body = f"""Hi {clean_name},

Thank you for taking the time to apply for {role_title}.

Our team completed an initial review of your profile. We noted your valuable transferable skills. To help us gain a deeper understanding of your practical experience, we would like to invite you to complete a brief exploratory assessment or share a recent work sample.

Please reply to this email if you are interested, and we will share the details.

Best regards,
Talent Acquisition Team"""
    else:
        subject = f"Update regarding your application for {role_title}"
        missing_note = f"While your profile has notable strengths, we are currently prioritizing applicants with direct hands-on verification in {', '.join(missing_gates)}." if missing_gates else "At this time, we have chosen to move forward with candidates whose specific backgrounds align more closely with our immediate requirements."
        body = f"""Hi {clean_name},

Thank you for giving us the opportunity to consider your profile for {role_title}.

{missing_note}

We truly appreciate the time and effort you invested in your application. With your permission, we will keep your resume on file in our talent pool for relevant future openings.

We wish you all the best in your career pursuits!

Warm regards,
Talent Acquisition Team"""
    return subject, body

# ----------------- Sidebar Controls -----------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=42)
    st.subheader("⚙️ Screening Controls")
    
    with st.expander("🛡️ Compliance & Bias Safeguards", expanded=True):
        enable_blind = st.checkbox("Blind Screening (Anonymize PII)", value=True, help="Removes phone numbers, email addresses, and portfolio URLs during evaluation.", key="enable_blind_cb")
        enable_second_look = st.checkbox("Second-Look Leniency Buffer", value=True, help="Places candidates within 10% of the passing cutoff into a manual audit queue.", key="enable_second_look_cb")
        enable_gap_grace = st.checkbox("Career Continuity Grace", value=True, help="Prevents automated penalties for non-linear career breaks.", key="enable_gap_grace_cb")

    with st.expander("🎯 Shortlist Parameters", expanded=True):
        shortlist_cutoff = st.slider("Passing Cutoff Score (%)", min_value=30, max_value=85, value=50, step=5, key="cutoff_slider")
        role_label = st.text_input("Role Title for Communications:", value="HR Executive", key="role_label_input")

# ----------------- Navigation Tabs -----------------
tab1, tab2, tab3 = st.tabs(["📝 1. Job Brief & Resumes", "📋 2. Candidate Review Board", "📊 3. Cohort Analytics"])

with tab1:
    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        st.markdown("#### 1. Target Job Description")
        jd_input = st.text_area(
            "Paste the complete Job Description:",
            height=250,
            placeholder="Paste role responsibilities, required qualifications, and core competencies...",
            key="jd_text_area"
        )
        extracted_keywords = extract_jd_keywords(jd_input) if jd_input.strip() else []

    with c2:
        st.markdown("#### 2. Candidate Resumes")
        uploaded_files = st.file_uploader("Upload candidate PDFs:", type=["pdf"], accept_multiple_files=True, key="resume_uploader")
        if uploaded_files:
            st.success(f"📂 {len(uploaded_files)} candidate resume(s) staged.")

    st.markdown("---")
    st.markdown("#### 3. Verification Criteria & Competency Extraction")

    gc1, gc2 = st.columns([1, 1], gap="large")
    with gc1:
        st.markdown("##### 🛑 Mandatory Gates (100% presence required to pass)")
        auto_gates = st.multiselect(
            "Select mandatory keywords from JD:",
            options=extracted_keywords,
            default=[k for k in extracted_keywords if k in ["excel", "recruitment", "payroll"]][:1],
            key="mandatory_gates_multiselect"
        )
        manual_gates = st.text_input(
            "➕ Or add custom mandatory gates manually (comma-separated):",
            placeholder="e.g. mba, excel, work authorization",
            key="manual_mandatory_gates_input"
        )

    with gc2:
        st.markdown("##### ✅ Evaluated Competencies (60% Match Weight)")
        auto_skills = st.multiselect(
            "Select evaluated competencies from JD:",
            options=extracted_keywords,
            default=extracted_keywords[:8] if extracted_keywords else [],
            key="evaluated_skills_multiselect"
        )
        manual_skills = st.text_input(
            "➕ Or add custom competencies manually (comma-separated):",
            placeholder="e.g. onboarding, communication, employee relations",
            key="manual_evaluated_skills_input"
        )

    active_mandatory = list(dict.fromkeys(auto_gates + parse_manual_inputs(manual_gates)))
    active_evaluated = list(dict.fromkeys(auto_skills + parse_manual_inputs(manual_skills)))

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 Run Screening & Verification Audit", type="primary", use_container_width=True, key="run_screening_btn"):
        if not jd_input.strip():
            st.error("Please provide a Job Description.")
        elif not uploaded_files:
            st.error("Please upload at least one PDF resume.")
        else:
            candidates_data = []
            for f in uploaded_files:
                pdf_text = extract_pdf_text(f)
                email = extract_candidate_email(pdf_text)
                eval_text = redact_pii(pdf_text) if enable_blind else pdf_text

                score, skill_score, context_score, matched, passed_gate, missing_gates = score_candidate(
                    jd_input, eval_text, active_mandatory, active_evaluated
                )

                audit_log = []
                if not passed_gate:
                    audit_log.append(f"❌ Missing Mandatory: {', '.join(missing_gates).upper()}")
                elif active_mandatory:
                    audit_log.append(f"✅ Verified Mandatory: {', '.join(active_mandatory).upper()}")

                if enable_gap_grace and any(term in pdf_text.lower() for term in ["gap", "career break", "freelance"]):
                    audit_log.append("📌 Career continuity grace noted")

                if not passed_gate:
                    status = "⛔ Gate Disqualified"
                elif score >= shortlist_cutoff:
                    status = "⭐ SHORTLISTED"
                elif enable_second_look and score >= (shortlist_cutoff - 10):
                    status = "⚠️ Second-Look Audit"
                    audit_log.append("💡 Borderline score: review transferable skills")
                else:
                    status = "❌ Not Shortlisted"

                subj, draft = generate_candidate_email(f.name, status, matched, missing_gates, role_title=role_label)

                candidates_data.append({
                    "candidate_name": f.name,
                    "candidate_email": email,
                    "final_score": score,
                    "skill_score": skill_score,
                    "context_score": context_score,
                    "status": status,
                    "matched_skills": ", ".join(matched[:8]) if matched else "None",
                    "matched_skills_list": matched,
                    "audit_checks": " | ".join(audit_log) if audit_log else "Standard match",
                    "email_subject": subj,
                    "email_body": draft
                })

            candidates_data = sorted(candidates_data, key=lambda x: x["final_score"], reverse=True)
            st.session_state["cohort_results"] = candidates_data
            st.success("✅ Screening complete! Switch to Tab 2 to view the Candidate Review Board.")

with tab2:
    if "cohort_results" in st.session_state:
        res = st.session_state["cohort_results"]

        # Metric Badges
        total = len(res)
        shortlisted = sum(1 for c in res if "SHORTLISTED" in c["status"])
        second_look = sum(1 for c in res if "Second-Look" in c["status"])
        avg_score = round(sum(c["final_score"] for c in res) / total, 1) if total > 0 else 0

        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(f'<div class="metric-card"><div class="metric-label">Screened Cohort</div><div class="metric-val">{total}</div></div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="metric-card"><div class="metric-label">Shortlisted</div><div class="metric-val" style="color:#047857;">{shortlisted}</div></div>', unsafe_allow_html=True)
        m3.markdown(f'<div class="metric-card"><div class="metric-label">Second-Look Queue</div><div class="metric-val" style="color:#B45309;">{second_look}</div></div>', unsafe_allow_html=True)
        m4.markdown(f'<div class="metric-card"><div class="metric-label">Cohort Average</div><div class="metric-val">{avg_score}%</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # CSV Download Button
        export_df = pd.DataFrame([{
            "Candidate File": c["candidate_name"],
            "Detected Email": c["candidate_email"],
            "ATS Score (%)": c["final_score"],
            "Competency Score (60%)": c["skill_score"],
            "Context Fit (40%)": c["context_score"],
            "Status": c["status"],
            "Matched Competencies": c["matched_skills"],
            "Audit Notes": c["audit_checks"],
            "Email Subject": c["email_subject"],
            "Email Body": c["email_body"]
        } for c in res])

        csv_file = export_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full Cohort Report (.CSV)",
            data=csv_file,
            file_name=f"smarthire_screening_cohort_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_cohort_csv_btn"
        )

        st.markdown("### 🏆 Candidate Evaluation Cards")
        for i, cand in enumerate(res):
            status = cand["status"]
            if "SHORTLISTED" in status:
                badge = f'<span class="badge-shortlisted">{status}</span>'
            elif "Second-Look" in status:
                badge = f'<span class="badge-second-look">{status}</span>'
            else:
                badge = f'<span class="badge-disqualified">{status}</span>'

            with st.expander(f"#{i+1} {cand['candidate_name']} — Match Score: {cand['final_score']}%", expanded=(i == 0)):
                h1, h2 = st.columns([3, 1])
                with h1:
                    st.markdown(f"**Status:** {badge}", unsafe_allow_html=True)
                    st.markdown(f"**Contact Detected:** `{cand['candidate_email']}`")
                    st.markdown(f"**Audit Verification:** {cand['audit_checks']}")
                with h2:
                    st.metric("Composite Match", f"{cand['final_score']}%")

                st.progress(int(cand['final_score']))

                sc1, sc2 = st.columns(2)
                sc1.caption(f"🎯 Skill Coverage (60%): **{cand['skill_score']}%**")
                sc2.caption(f"🧠 Context Alignment (40%): **{cand['context_score']}%**")

                if cand.get("matched_skills_list"):
                    tags = "".join([f'<span class="skill-tag">{s}</span>' for s in cand["matched_skills_list"]])
                    st.markdown(f"**Matched Competencies:**<br>{tags}", unsafe_allow_html=True)

                st.markdown("---")
                st.markdown("##### ✉️ Candidate Outreach Communication")

                p_tab1, p_tab2 = st.tabs(["📧 Preview Draft", "📝 Edit Draft"])
                with p_tab1:
                    st.markdown(f"**Subject:** {cand['email_subject']}")
                    st.info(cand['email_body'])
                with p_tab2:
                    edited_body = st.text_area("Custom message:", value=cand['email_body'], height=150, key=f"edit_draft_area_{i}")

                # Mailto launch button
                subj_enc = urllib.parse.quote(cand['email_subject'])
                body_enc = urllib.parse.quote(edited_body if edited_body else cand['email_body'])
                mailto = f"mailto:{cand['candidate_email']}?subject={subj_enc}&body={body_enc}"

                st.markdown(
                    f'<a href="{mailto}" target="_blank" style="text-decoration:none;">'
                    f'<button style="background-color:#1F4E79; color:white; padding:7px 14px; border:none; '
                    f'border-radius:6px; font-weight:600; cursor:pointer;">📨 Open in Local Email Client</button></a>',
                    unsafe_allow_html=True
                )
    else:
        st.info("👈 Run screening in Tab 1 to view results.")

with tab3:
    if "cohort_results" in st.session_state:
        st.markdown("### 📊 Cohort Score Distribution")
        df_chart = pd.DataFrame([{
            "Candidate": c["candidate_name"],
            "Skill Score": c["skill_score"],
            "Context Fit": c["context_score"]
        } for c in st.session_state["cohort_results"]]).set_index("Candidate")

        st.bar_chart(df_chart)
    else:
        st.info("Run screening in Tab 1 to populate analytics.")