import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string

# --- NLTK DATA DOWNLOAD ---
# Required for deployment on Streamlit Cloud to prevent LookupErrors
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('punkt')
        nltk.download('punkt_tab')
        nltk.download('stopwords')
        nltk.download('wordnet')

download_nltk_data()

# --- FAQ DATASET ---
# Customize these questions and answers for your specific topic
FAQ_DATA = [
    {
        "question": "What is your return policy?",
        "answer": "We offer a 30-day money-back guarantee. Items must be returned in their original packaging and unused condition."
    },
    {
        "question": "How long does shipping take?",
        "answer": "Standard shipping takes 3-5 business days. International shipping can take anywhere from 7-14 business days."
    },
    {
        "question": "Do you offer international shipping?",
        "answer": "Yes! We ship to over 50 countries worldwide. Shipping fees and delivery times vary by destination."
    },
    {
        "question": "How can I track my order?",
        "answer": "Once your order ships, we will send you an email with a tracking number and a link to trace your package."
    },
    {
        "question": "Can I change or cancel my order?",
        "answer": "Orders can be changed or canceled within 1 hour of placing them. Please contact our support team immediately at support@example.com."
    }
]

# Extract questions and answers into lists
faq_questions = [faq["question"] for faq in FAQ_DATA]
faq_answers = [faq["answer"] for faq in FAQ_DATA]

# --- NLP PREPROCESSING FUNCTION ---
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords and lemmatize
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(cleaned_tokens)

# Preprocess all FAQ questions
preprocessed_faqs = [preprocess_text(q) for q in faq_questions]

# --- CHATBOT LOGIC ---
def get_best_response(user_query, threshold=0.25):
    # Preprocess user query
    processed_query = preprocess_text(user_query)
    
    # If the user input is empty after preprocessing
    if not processed_query.strip():
        return "I'm sorry, I didn't quite catch that. Could you please rephrase your question?"

    # Combine query with FAQs for vectorization
    corpus = preprocessed_faqs + [processed_query]
    
    # Vectorize text using TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    # Calculate Cosine Similarity between user query (last item) and all FAQs
    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
    
    # Find the best match
    best_match_idx = similarity_scores.argmax()
    highest_score = similarity_scores[best_match_idx]
    
    # Return answer if it passes the confidence threshold, else fallback
    if highest_score >= threshold:
        return faq_answers[best_match_idx]
    else:
        return "I'm not sure I understand. Could you try asking in a different way? Alternatively, you can reach our support team at support@example.com."

# --- STREAMLIT UI ---
st.set_page_config(page_title="FAQ Help Bot", page_icon="🤖", layout="centered")

st.title("🤖 Customer Support FAQ Bot")
st.write("Ask me anything about our shipping, returns, or order tracking!")
st.markdown("---")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! How can I help you today?"}
    ]

# Display past chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if user_input := st.chat_input("Type your question here..."):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_input)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Generate bot response
    bot_response = get_best_response(user_input)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(bot_response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
