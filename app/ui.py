import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Document Clustering Dashboard",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS styling (Dark theme accents with modern typography and gradients)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main title gradient */
    .title-text {
        background: linear-gradient(135deg, #FF4B4B, #FF8F8F, #8E2DE2, #4A00E0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .subtitle-text {
        font-size: 1.2rem;
        color: #7f8c8d;
        margin-bottom: 2rem;
    }
    
    /* Custom cards for results */
    .result-card {
        background-color: #f8f9fa;
        border-left: 5px solid #8E2DE2;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
    }
    
    .dark .result-card {
        background-color: #2e3136;
        border-left: 5px solid #8E2DE2;
    }
    
    .result-header {
        font-size: 0.9rem;
        text-transform: uppercase;
        color: #8E2DE2;
        font-weight: 600;
        letter-spacing: 0.1em;
        margin-bottom: 0.5rem;
    }
    
    .result-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    
    .dark .result-value {
        color: #ecf0f1;
    }
    
    /* Tag styling */
    .keyword-tag {
        display: inline-block;
        background: linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%);
        color: #2c3e50;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        margin: 0.2rem;
        font-size: 0.85rem;
        font-weight: 600;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Preprocessed text block */
    .cleaned-text-box {
        background-color: #edf2f7;
        font-family: monospace;
        padding: 1rem;
        border-radius: 6px;
        border: 1px solid #cbd5e0;
        color: #2d3748;
        font-size: 0.95rem;
        margin-top: 0.5rem;
    }
    
    .dark .cleaned-text-box {
        background-color: #1a202c;
        border-color: #4a5568;
        color: #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_URL = st.sidebar.text_input("FastAPI Base URL", value="http://localhost:8000")

st.sidebar.markdown("---")
st.sidebar.header("Model Selection")
model_choice = st.sidebar.radio(
    "Choose a dataset/model:",
    ("Wikipedia Biographies (Wiki)", "20 Newsgroups (News)"),
    index=0
)

model_type = "wiki" if "Wikipedia" in model_choice else "news"

# Setup example texts for both models
wiki_examples = {
    "Select an example...": "",
    "Sports - Michael Jordan (Basketball)": "Michael Jordan is an American former professional basketball player. He played fifteen seasons in the NBA, winning six championships with the Chicago Bulls. He is widely considered one of the greatest athletes of all time.",
    "Academic - Marie Curie (Physicist)": "Marie Curie was a Polish and naturalized-French physicist and chemist who conducted pioneering research on radioactivity. She was the first woman to win a Nobel Prize, the first person and only woman to win the Nobel Prize twice, and the only person to win a Nobel Prize in two different scientific fields.",
    "Music - Freddie Mercury (Singer)": "Freddie Mercury was a British singer, songwriter, record producer, and lead vocalist of the rock band Queen. Regarded as one of the greatest lead singers in the history of rock music, he was known for his flamboyant stage persona and four-octave vocal range.",
    "Politics - Barack Obama (Politician)": "Barack Obama is an American politician who served as the 44th president of the United States from 2009 to 2017. A member of the Democratic Party, he was the first African-American president of the United States.",
    "Literature - J.K. Rowling (Author)": "J.K. Rowling is a British author, philanthropist, film producer, and screenwriter. She is best known for writing the Harry Potter fantasy series, which has won multiple awards and sold more than 500 million copies, becoming the best-selling book series in history."
}

news_examples = {
    "Select an example...": "",
    "PC Hardware - Disk Drive Issue": "I am having trouble configuring my SCSI controller card. The system refuses to boot when the new 2GB SCSI hard drive is connected. I verified the termination resistors and SCSI ID settings, but udev fails to discover the device. Any suggestions on disk partitioning or kernel drivers?",
    "Hockey - Stanley Cup Playoffs": "The Detroit Red Wings played a spectacular game against the Colorado Avalanche last night. The defense was solid, and the goaltender made 35 saves to secure the win. With this playoff victory, they are heading to the Stanley Cup finals!",
    "Middle East - Diplomatic Relations": "The delegations from Israel and neighboring Arab states met in Geneva to negotiate a bilateral peace treaty. The discussion centered on international boundaries, Jerusalem's status, and security guarantees. Diplomatic envoys remain hopeful for a lasting ceasefire."
}

examples = wiki_examples if model_type == "wiki" else news_examples

st.sidebar.header("Quick Test Templates")
selected_example = st.sidebar.selectbox("Load example text:", list(examples.keys()))

# App layout
st.markdown('<div class="title-text">🧩 Document Clustering Hub</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">Real-time unsupervised text classification powered by FastAPI, Scikit-Learn KMeans, and TF-IDF</div>', unsafe_allow_html=True)

# Text Area
input_text = st.text_area(
    "Enter the text to analyze:",
    value=examples[selected_example] if selected_example != "Select an example..." else "",
    height=200,
    placeholder="Type or paste your text here..."
)

# Button row
col1, col2 = st.columns([1, 5])
with col1:
    submit_button = st.button("Predict Cluster", type="primary")
with col2:
    if st.button("Clear Text"):
        st.experimental_rerun()

# Run Prediction
if submit_button or (input_text and selected_example != "Select an example..."):
    if not input_text.strip():
        st.error("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing text..."):
            try:
                # Send request to FastAPI
                response = requests.post(
                    f"{API_URL}/predict",
                    json={"text": input_text, "model_type": model_type},
                    timeout=10
                )
                
                if response.status_code == 200:
                    res_data = response.json()
                    
                    # Columns for results
                    res_col1, res_col2 = st.columns([3, 2])
                    
                    with res_col1:
                        # Theme and Category Card
                        st.markdown(f"""
                        <div class="result-card">
                            <div class="result-header">Predicted Theme</div>
                            <div class="result-value">{res_data['topic']}</div>
                            <p><strong>Cluster ID:</strong> {res_data['cluster_id']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Preprocessed text
                        st.markdown("### Text Preprocessing")
                        st.markdown("This is how the text looks after stopword removal, lemmatization, and token cleaning:")
                        st.markdown(f'<div class="cleaned-text-box">{res_data["preprocessed_text"]}</div>', unsafe_allow_html=True)
                        
                    with res_col2:
                        st.markdown("### Cluster Keyword Significance")
                        st.markdown("Key terms associated with this cluster:")
                        
                        # Display keywords as tags
                        keywords_html = "".join([f'<span class="keyword-tag">{kw}</span>' for kw in res_data['keywords']])
                        st.markdown(keywords_html, unsafe_allow_html=True)
                        
                        # Draw a small keyword bar chart
                        st.markdown("---")
                        st.markdown("#### Topic Focus Pattern")
                        
                        # Let's create dummy weights for the top words just to visualize their importance relative to the cluster
                        # (A descending scale for visual representation)
                        df_keywords = pd.DataFrame({
                            "Word": res_data['keywords'],
                            "Relative Weight": [10 - i for i in range(len(res_data['keywords']))]
                        })
                        
                        # Matplotlib chart for custom sleek styling
                        fig, ax = plt.subplots(figsize=(6, 4))
                        sns.barplot(
                            x="Relative Weight", 
                            y="Word", 
                            data=df_keywords, 
                            palette="crest_r",
                            ax=ax
                        )
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.spines['left'].set_visible(False)
                        ax.spines['bottom'].set_color('#cccccc')
                        ax.tick_params(left=False)
                        plt.title("Cluster Keyword Relevance (Ranked)", fontsize=10, fontweight="bold", pad=10)
                        plt.tight_layout()
                        
                        st.pyplot(fig)
                        
                else:
                    st.error(f"Error {response.status_code}: {response.json().get('detail', 'Unknown error')}")
            except requests.exceptions.ConnectionError:
                st.error(f"Could not connect to FastAPI server at {API_URL}. Please ensure it is running and accessible.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")

# Info section at the bottom
st.markdown("---")
st.markdown("""
### How it works
1. **SpaCy Preprocessing**: The raw input text is tokenized, lemmatized, and stripped of punctuation and stopwords (common words like 'the', 'is', 'on').
2. **TF-IDF Vectorization**: The preprocessed text is transformed into a numerical vector using a trained TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer.
3. **KMeans Prediction**: The trained KMeans model calculates the closest cluster center and maps the vector to its corresponding cluster.
""")
