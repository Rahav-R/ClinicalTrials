import streamlit as st
import google.generativeai as genai
import requests
import time
from typing import List, Dict

# --- CONFIGURATION ---
st.set_page_config(
    page_title="MediTrial AI | Clinical Trials Assistant",
    page_icon="🧬",
    layout="wide"
)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR & SETUP ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=100)
    st.title("MediTrial AI")
    
    # 1. API Key Handling
    api_key = None
    if "GOOGLE_API_KEY" in st.secrets:
        secret_key = st.secrets["GOOGLE_API_KEY"]
        if "YOUR_API_KEY" not in secret_key:
            api_key = secret_key
            st.success("✅ API Key loaded from secrets")
        else:
            st.error("⚠️ Invalid Key in secrets.toml")
    
    if not api_key:
        api_key = st.text_input("Enter Google Gemini API Key", type="password")
    
    st.markdown("---")
    
    # Persona Selection
    user_persona = st.radio(
        "I am a:",
        ("Patient / Caregiver", "Doctor / Researcher"),
        index=0
    )
    
    # Advanced Filters
    st.markdown("### 🔍 Search Filters")
    status_filter = st.multiselect(
        "Recruitment Status",
        ["RECRUITING", "NOT_YET_RECRUITING", "ACTIVE_NOT_RECRUITING", "COMPLETED"],
        default=["RECRUITING"]
    )
    
    # Reduced max results to save tokens on free tier
    max_results = st.slider("Max Studies to Analyze", 3, 15, 5)
    
    st.markdown("---")
    
    # Debug Section
    with st.expander("🛠️ Debug: Available Models"):
        if st.button("Check My API Models"):
            if api_key:
                try:
                    genai.configure(api_key=api_key)
                    models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                    st.write(models)
                except Exception as e:
                    st.error(f"Error listing models: {e}")
            else:
                st.warning("Enter API Key first")

    st.warning("⚠️ **DISCLAIMER:** This tool is a student project for educational purposes. It does not provide medical advice. Always consult a healthcare professional.")

# --- HELPER: ROBUST GEMINI CALLER WITH RETRY ---
def call_gemini_safe(prompt, api_key, image=None):
    """
    1. Dynamically finds available models to avoid 404 errors.
    2. Prioritizes stable models (Flash > Pro).
    3. Implements exponential backoff for 429 (Rate Limit) errors.
    """
    genai.configure(api_key=api_key)
    
    # 1. DYNAMICALLY FIND VALID MODELS
    # We ask the API what models are actually available for this key
    available_models = []
    try:
        all_models = genai.list_models()
        for m in all_models:
            if 'generateContent' in m.supported_generation_methods:
                available_models.append(m.name)
    except Exception:
        pass # Fallback to hardcoded list if listing fails

    # Define our preference order (Fastest/Cheapest -> Most Capable)
    preferences = [
        'gemini-1.5-flash',
        'gemini-1.5-flash-latest',
        'gemini-1.5-flash-001',
        'gemini-1.5-pro',
        'gemini-1.5-pro-latest',
        'gemini-pro'
    ]
    
    models_to_try = []
    
    # Match preferences against what is actually available
    if available_models:
        for pref in preferences:
            for avail in available_models:
                if pref in avail:
                    if avail not in models_to_try:
                        models_to_try.append(avail)
        # If no preferred models found, add all available ones as backup
        if not models_to_try:
            models_to_try = available_models
    else:
        # Fallback if we couldn't list models
        models_to_try = preferences

    last_error = None
    
    # 2. ATTEMPT GENERATION
    for model_name in models_to_try:
        model = genai.GenerativeModel(model_name)
        
        # Retry loop for Rate Limiting (429 errors)
        for attempt in range(3): # Try up to 3 times per model
            try:
                if image:
                    return model.generate_content([prompt, image]).text
                else:
                    return model.generate_content(prompt).text
            
            except Exception as e:
                error_str = str(e)
                
                # If it's a Rate Limit error (429), wait and retry
                if "429" in error_str or "quota" in error_str.lower():
                    wait_time = (attempt + 1) * 2 # Wait 2s, then 4s, then 6s
                    time.sleep(wait_time)
                    last_error = f"Rate limit hit on {model_name}. Retrying..."
                    continue # Try loop again
                
                # If model not found (404), break loop to try next model in list
                if "404" in error_str:
                    last_error = f"Model {model_name} not found."
                    break 
                
                # For other errors (Auth, etc), stop immediately
                return f"AI Error: {error_str}"
                
    return f"Service Unavailable. Please wait a minute and try again. (Details: {last_error})"

# --- BACKEND FUNCTIONS ---

def extract_search_keywords(user_query: str, api_key: str) -> str:
    """
    Uses LLM to convert a natural language sentence into specific search terms.
    """
    prompt = f"""
    Extract the 2-3 most important medical search terms from this query for a clinical trial database search.
    User Query: "{user_query}"
    Return ONLY the terms separated by spaces. Do not add quotes, labels, or explanations. 
    If the query is already a keyword (e.g. "Diabetes"), just return it.
    """
    try:
        # Use our safe caller
        result = call_gemini_safe(prompt, api_key)
        
        # Fail-safe: If AI returns an error message, fallback to user query
        if "Error" in result or "Unavailable" in result:
            return user_query 
            
        return result.strip()
    except:
        return user_query

# Cache this function so we don't spam the API when the user changes filters
@st.cache_data(ttl=3600, show_spinner=False)
def get_clinical_trials(query: str, status: List[str], limit: int = 10) -> Dict:
    """
    Fetches clinical trials from ClinicalTrials.gov API v2.
    """
    base_url = "https://clinicaltrials.gov/api/v2/studies"
    status_str = ",".join(status) if status else "RECRUITING"
    
    # NOTE: We fetch the full study object to avoid 400 Bad Request errors on specific fields.
    params = {
        "query.term": query,
        "filter.overallStatus": status_str,
        "pageSize": limit
    }
    
    # Headers to prevent 403 Forbidden errors (Server rejects scripts without User-Agent)
    headers = {
        "User-Agent": "MediTrial-AI-Student-Project/1.0 (Educational Purpose)"
    }
    
    try:
        response = requests.get(base_url, params=params, headers=headers, timeout=20)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        return {"error": f"HTTP Error {e.response.status_code}: {e.response.reason}"}
    except requests.exceptions.ConnectionError:
        return {"error": "Connection Failed. Check your internet."}
    except requests.exceptions.Timeout:
        return {"error": "Request Timed Out. ClinicalTrials.gov is slow right now."}
    except requests.exceptions.RequestException as e:
        return {"error": f"API Error: {str(e)}"}

def format_trials_for_llm(trials_data: Dict) -> str:
    if "error" in trials_data:
        return f"System Error: {trials_data['error']}"
    
    studies = trials_data.get("studies", [])
    if not studies:
        return "No clinical trials found matching the criteria."
    
    context_text = "Here are the retrieved clinical trials:\n\n"
    
    for study in studies:
        protocol = study.get("protocolSection", {})
        id_module = protocol.get("identificationModule", {})
        
        nct_id = id_module.get("nctId", "N/A")
        title = id_module.get("briefTitle", "No Title")
        
        # Conditions
        conditions = protocol.get("conditionsModule", {}).get("conditions", [])
        conditions_str = ", ".join(conditions) if conditions else "Not specified"
        
        # Eligibility
        eligibility = protocol.get("eligibilityModule", {})
        gender = eligibility.get("sex", "All")
        min_age = eligibility.get("minimumAge", "N/A")
        max_age = eligibility.get("maximumAge", "N/A")
        
        criteria = eligibility.get("eligibilityCriteria", "Not specified")
        criteria_snippet = criteria[:600].replace("\n", " ") + "..." if len(criteria) > 600 else criteria
        
        context_text += f"--- STUDY ID: {nct_id} ---\n"
        context_text += f"Title: {title}\n"
        context_text += f"Conditions: {conditions_str}\n"
        context_text += f"Eligibility: Sex: {gender}, Age: {min_age}-{max_age}\n"
        context_text += f"Criteria Snippet: {criteria_snippet}\n\n"
        
    return context_text

def generate_response(messages, context, persona, api_key):
    # Enhanced Prompt Engineering with Link Instructions
    if "Doctor" in persona:
        system_instruction = (
            "You are an expert Clinical Research Assistant helping a doctor. "
            "Use precise medical terminology. Focus on inclusion/exclusion criteria. "
            "CRITICAL: When mentioning a study, ALWAYS format the NCT ID as a clickable Markdown link "
            "like this: [NCT12345](https://clinicaltrials.gov/study/NCT12345)."
        )
    else:
        system_instruction = (
            "You are a helpful, empathetic Medical Guide for a patient. "
            "Explain complex medical terms in simple language. "
            "CRITICAL: When mentioning a study, ALWAYS format the NCT ID as a clickable Markdown link "
            "like this: [NCT12345](https://clinicaltrials.gov/study/NCT12345)."
        )

    prompt = f"""
    System Instruction: {system_instruction}
    
    Context Data (Retrieved Clinical Trials):
    {context}
    
    User Query: {messages[-1]['content']}
    
    Answer the user's query strictly based on the provided Context Data.
    If the context contains no relevant trials, state that clearly.
    """
    
    return call_gemini_safe(prompt, api_key)

# --- MAIN UI LAYOUT ---

st.header("🧬 MediTrial AI: Clinical Trial Finder")
st.markdown(f"**Current Mode:** `{user_persona}`")

# 1. Image Upload Section (Multimodal)
with st.expander("📸 Upload Medical Report / Prescription (Optional)", expanded=False):
    uploaded_file = st.file_uploader("Upload an image to auto-find trials", type=["jpg", "png", "jpeg"])

    if uploaded_file and api_key:
        from PIL import Image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Medical Record", width=250)
        
        if st.button("Analyze Image & Find Trials"):
            with st.spinner("Analyzing image... (This might take a few seconds due to free tier limits)"):
                extraction_prompt = "Identify the primary medical condition and 3 key search terms from this image. Return ONLY the search terms as a comma-separated list."
                extracted_terms = call_gemini_safe(extraction_prompt, api_key, image)
                
                # Check for error in extraction before proceeding
                if "Error" in extracted_terms or "Unavailable" in extracted_terms:
                    st.error("Could not analyze image. Please try typing your query instead.")
                else:
                    st.success(f"Extracted Terms: {extracted_terms}")
                    
                    # Add to chat history and trigger search
                    st.session_state.messages.append({"role": "user", "content": f"Find trials for: {extracted_terms}"})
                    
                    with st.spinner("Fetching trials..."):
                        trials_data = get_clinical_trials(extracted_terms, status_filter, max_results)
                        context = format_trials_for_llm(trials_data)
                        ai_response = generate_response(st.session_state.messages, context, user_persona, api_key)
                        st.session_state.messages.append({"role": "assistant", "content": ai_response})
                        st.rerun()

# 2. Chat Interface
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 3. User Input Handling
if prompt := st.chat_input("Ask about a condition (e.g., 'Breast Cancer trials in Phase 3')"):
    if not api_key:
        st.error("Please enter your Gemini API Key in the sidebar.")
    else:
        # User message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Assistant Response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Connecting to ClinicalTrials.gov..."):
                
                # Step A: Keyword Extraction
                search_term = extract_search_keywords(prompt, api_key)
                if search_term and search_term.lower() not in prompt.lower() and "Unavailable" not in search_term:
                    st.caption(f"🔎 *Refined Search Term: {search_term}*")
                
                # Step B: API Search
                trials_data = get_clinical_trials(search_term, status_filter, max_results)
                
                # CHECK FOR API ERRORS
                if "error" in trials_data:
                    st.error(f"❌ Database Error: {trials_data['error']}")
                    full_response = "I couldn't access the database. Please see the error above."
                else:
                    # Step C: RAG Contextualization
                    context = format_trials_for_llm(trials_data)
                    
                    # Step D: Final Generation
                    full_response = generate_response(st.session_state.messages, context, user_persona, api_key)
                
                message_placeholder.markdown(full_response)
        
        # Save to history
        st.session_state.messages.append({"role": "assistant", "content": full_response})