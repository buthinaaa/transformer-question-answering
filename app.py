# app.py
import streamlit as st
from transformers import pipeline
import os

# Optional: suppress symlink warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Title
st.set_page_config(page_title="QA Model Comparator", layout="centered")
st.title("🔍 Question Answering with Transformers")
st.markdown("Enter a passage and a question. Choose a model to get an answer!")

# Model selector
MODEL_OPTIONS = {
    "BERT (csarron/bert-base-uncased-squad-v1)": "csarron/bert-base-uncased-squad-v1",
    "DistilBERT (distilbert-base-uncased-distilled-squad)": "distilbert-base-uncased-distilled-squad"
}

selected_model_name = st.selectbox("Choose a model", list(MODEL_OPTIONS.keys()))
model_id = MODEL_OPTIONS[selected_model_name]

# User inputs
context = st.text_area("Context (Passage)", height=200, placeholder="Paste a paragraph here...")
question = st.text_input("Question", placeholder="Ask a question about the passage...")

# Cache the pipeline to avoid reloading on every interaction
@st.cache_resource
def load_qa_pipeline(model_id):
    return pipeline(
        "question-answering",
        model=model_id,
        tokenizer=model_id,
        device=-1  # CPU
    )

# Run when user clicks button
if st.button("Get Answer"):
    if not context.strip() or not question.strip():
        st.warning("⚠️ Please provide both a context and a question.")
    else:
        try:
            with st.spinner("🧠 Thinking..."):
                qa_pipeline = load_qa_pipeline(model_id)
                result = qa_pipeline(question=question, context=context)
            
            # Display result
            st.success(f"**Answer**: {result['answer']}")
            st.info(f"**Confidence Score**: {result['score']:.3f}")
            
            # Optional: show model info
            st.caption(f"Model: {selected_model_name}")
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.write("Common issues: very long context, out of memory, or invalid input.")

# Footer
st.markdown("---")
st.caption("Built with Hugging Face Transformers + Streamlit | Models fine-tuned on SQuAD v1.1")