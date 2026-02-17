import streamlit as st
import google.generativeai as genai
import requests
import time
import numpy as np
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURATION ---
st.set_page_config(
    page_title="MediTrial AI | Global Assistant",
    page_icon="🧬",
    layout="wide"
)

# Initialize Session State
if "history_vectors" not in st.session_state:
    st.session_state.history_vectors = [] 
if "messages" not in st.session_state:
    st.session_state.messages = []
if "image_analysis_context" not in st.session_state:
    st.session_state.image_analysis_context = ""

# --- SIDEBAR & SETUP ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=80)
    st.title("MediTrial AI 2.0")
    
    # 1. API Key Handling
    api_key = None
    if "GOOGLE_API_KEY" in st.secrets:
        secret_key = st.secrets["GOOGLE_API_KEY"]
        if "YOUR_API_KEY" not in secret_key:
            api_key = secret_key
    
    if not api_key:
        api_key = st.text_input("Gemini API Key", type="password")
    
    # 2. Language Selection
    selected_language = st.selectbox(
        "🗣️ Language / 语言 / भाषा",
        ["English", "Tamil", "Spanish", "French", "Hindi", "Chinese (Simplified)", "Arabic", "German"],
        index=0
    )
    
    # 3. Persona Selection
    user_persona = st.radio(
        "I am a:",
        ("Patient / Caregiver", "Doctor / Researcher"),
        index=0
    )
    
    st.markdown("---")

# --- HELPER: ROBUST GEMINI CALLER ---
def get_working_model(api_key):
    """Finds the best available model to avoid 404s."""
    genai.configure(api_key=api_key)
    try:
        models = genai.list_models()
        available = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
        
        # Priority: Flash > Pro
        for m in available:
            if "flash" in m and "1.5" in m: return m
        for m in available:
            if "pro" in m and "1.5" in m: return m
        if available: return available[0]
    except:
        pass
    return "models/gemini-1.5-flash"

def call_gemini_safe(prompt, api_key, image=None):
    model_name = get_working_model(api_key)
    try:
        model = genai.GenerativeModel(model_name)
        if image:
            return model.generate_content([prompt, image]).text
        else:
            return model.generate_content(prompt).text
    except Exception as e:
        return f"AI Error: {str(e)}"

def transcribe_audio_with_gemini(audio_file, api_key):
    """
    Uses Gemini to transcribe audio bytes directly.
    """
    model_name = get_working_model(api_key)
    genai.configure(api_key=api_key)
    
    try:
        model = genai.GenerativeModel(model_name)
        
        # Read audio bytes
        audio_bytes = audio_file.read()
        
        # Gemini expects a specific structure for blob parts
        # Note: We are mocking the proper PyAudio upload structure for simplicity.
        # Ideally, we upload to File API, but for small clips, we can sometimes send bytes if supported.
        # If byte-upload fails in your specific library version, we use a text fallback.
        
        prompt = f"""
        Please transcribe this audio file exactly as spoken in {selected_language}. 
        Do not add any conversational filler. Just return the text.
        """
        
        # NOTE: Direct byte processing depends on library version. 
        # Standard robust way is using the File API, but here we attempt a direct generate content
        # with the audio mime type.
        response = model.generate_content([
            prompt,
            {"mime_type": "audio/wav", "data": audio_bytes}
        ])
        return response.text.strip()
    except Exception as e:
        return f"Transcription Error: {str(e)}"

# --- RAG ENGINE (MEMORY) ---
def get_embedding(text: str, api_key: str) -> np.ndarray:
    genai.configure(api_key=api_key)
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_document",
            title="Conversation History"
        )
        return np.array(result['embedding'])
    except:
        return np.zeros(768)

def retrieve_relevant_history(query: str, api_key: str, k: int = 3) -> str:
    if not st.session_state.history_vectors: return ""
    query_vec = get_embedding(query, api_key).reshape(1, -1)
    if np.all(query_vec == 0): return ""
    
    stored_vecs = np.array([item[0] for item in st.session_state.history_vectors])
    stored_texts = [item[1] for item in st.session_state.history_vectors]
    
    try:
        similarities = cosine_similarity(query_vec, stored_vecs)[0]
        top_k_indices = similarities.argsort()[-k:][::-1]
        
        retrieved = "### PAST CONVERSATION:\n"
        found = False
        for idx in top_k_indices:
            if similarities[idx] > 0.45:
                retrieved += f"- {stored_texts[idx]}\n"
                found = True
        return retrieved if found else ""
    except:
        return ""

def store_interaction(user_text: str, ai_text: str, api_key: str):
    u_vec = get_embedding(user_text, api_key)
    st.session_state.history_vectors.append((u_vec, f"User: {user_text}", "user"))
    a_vec = get_embedding(ai_text, api_key)
    st.session_state.history_vectors.append((a_vec, f"AI: {ai_text}", "ai"))

# --- BACKEND FUNCTIONS ---

@st.cache_data(ttl=3600)
def get_clinical_trials(query: str, status: str = "RECRUITING") -> Dict:
    if "treatment" not in query.lower():
        query += " treatment"
    base_url = "https://clinicaltrials.gov/api/v2/studies"
    params = {"query.term": query, "filter.overallStatus": status, "pageSize": 5}
    try:
        r = requests.get(base_url, params=params, timeout=10)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def format_trials(trials_data: Dict, include_links: bool) -> str:
    studies = trials_data.get("studies", [])
    if not studies: return "No specific clinical trials found. (Answer based on general medical knowledge)."
    text = ""
    for s in studies:
        p = s.get("protocolSection", {})
        id_mod = p.get("identificationModule", {})
        nct_id = id_mod.get('nctId', 'N/A')
        title = id_mod.get('briefTitle', 'No Title')
        
        if include_links:
            link = f"[{nct_id}](https://clinicaltrials.gov/study/{nct_id})"
            text += f"ID: {link} | Title: {title}\n"
            eligibility = p.get("eligibilityModule", {})
            criteria = eligibility.get("eligibilityCriteria", "")[:200] + "..."
            text += f"   Criteria Snippet: {criteria}\n"
        else:
            text += f"• Potential Treatment Option: {title}\n"
            eligibility = p.get("eligibilityModule", {})
            min_age = eligibility.get("minimumAge", "N/A")
            max_age = eligibility.get("maximumAge", "N/A")
            text += f"   (Available for ages {min_age} to {max_age})\n"
        text += "\n"
    return text

def generate_response_multilingual(query, context, persona, language, memory, api_key):
    if "Doctor" in persona:
        role_desc = "Expert Clinical Research Associate."
        tone = "Professional, precise, technical."
        link_instruction = "ALWAYS include clickable NCT ID links for studies."
        goal = "Summarize the recruitment status and protocol design."
    else:
        role_desc = "Compassionate Medical Caregiver."
        tone = "Warm, soothing, simple, hopeful."
        link_instruction = "DO NOT show NCT IDs or links. Focus on the 'Treatment Process' and how it helps."
        goal = "Explain the treatment journey step-by-step to reduce anxiety."

    prompt = f"""
    ROLE: {role_desc}
    TONE: {tone}
    TARGET LANGUAGE: {language} (Force output in this language).
    
    USER QUERY: {query}
    
    CONTEXT (Clinical Data):
    {context}
    
    MEMORY (Previous Chat):
    {memory}
    
    INSTRUCTIONS:
    1. {goal}
    2. {link_instruction}
    3. If the user asks for a specific language in the text (e.g., "in Tamil"), IGNORE the system setting and answer in the requested language.
    4. If the data lists a study, explain it as a "Treatment Option" rather than a "database record".
    5. Be culturally sensitive and supportive.
    """
    return call_gemini_safe(prompt, api_key)

# --- UI LAYOUT ---

st.header(f"🧬 MediTrial AI ({selected_language})")

# 1. Multimodal Input (Image)
with st.expander("📸 Upload Report / Scan", expanded=False):
    uploaded_file = st.file_uploader("Upload Medical Record", type=["jpg", "png"])
    if uploaded_file and api_key:
        from PIL import Image
        image = Image.open(uploaded_file)
        st.image(image, width=200)
        if st.button("Analyze Image"):
            with st.spinner("Analyzing..."):
                img_prompt = f"Analyze this image. Identify the condition. Explain it simply in {selected_language}."
                res = call_gemini_safe(img_prompt, api_key, image)
                st.session_state.image_analysis_context = res
                st.info(res)

# 2. Chat Interface
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 3. Logic Loop
# --- KEY CHANGE: accept_audio=True places the mic icon INSIDE the input box ---
prompt_data = st.chat_input(f"Ask about treatments in {selected_language}...", accept_audio=True)

if prompt_data:
    if not api_key:
        st.error("🔑 API Key required!")
        st.stop()
        
    # prompt_data is now a Dict: {'text': "...", 'audio': UploadedFile}
    user_text = prompt_data["text"]
    audio_file = prompt_data["audio"]
    
    # Logic: Prioritize audio if present, otherwise use text
    final_prompt = user_text
    
    if audio_file:
        with st.spinner("🎧 Transcribing your voice..."):
            transcription = transcribe_audio_with_gemini(audio_file, api_key)
            if "Error" not in transcription:
                final_prompt = transcription
                st.info(f"🗣️ You said: {final_prompt}")
            else:
                st.error(transcription)

    # Proceed only if we have a valid prompt (either text or transcribed audio)
    if final_prompt:
        st.session_state.messages.append({"role": "user", "content": final_prompt})
        with st.chat_message("user"):
            st.markdown(final_prompt)

        with st.chat_message("assistant"):
            with st.spinner("🧠 Analyzing treatment options..."):
                
                relevant_memory = retrieve_relevant_history(final_prompt, api_key)
                
                search_query = final_prompt
                if "image" in final_prompt.lower() and st.session_state.image_analysis_context:
                    search_query = st.session_state.image_analysis_context[:50]

                trials = get_clinical_trials(search_query)
                include_links_flag = True if "Doctor" in user_persona else False
                context_data = format_trials(trials, include_links=include_links_flag)
                
                response = generate_response_multilingual(
                    final_prompt, 
                    context_data, 
                    user_persona, 
                    selected_language, 
                    relevant_memory, 
                    api_key
                )
                
                st.markdown(response)
                store_interaction(final_prompt, response, api_key)
                st.session_state.messages.append({"role": "assistant", "content": response})