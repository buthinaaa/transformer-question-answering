# 🤖 Transformer-Based Question Answering System

This project implements a **question answering (QA) system** using pre-trained transformer models fine-tuned on the **SQuAD v1.1** dataset. It compares model performance using **Exact Match (EM)** and **F1 score**, logs results with **MLflow**, and provides an interactive **Streamlit demo** for real-time QA.
> 🚀 Try the live demo on [Streamlit]([https://buthinaaa-detecting-parkinson-s-building-a-diagnosti-app-yhscvj.streamlit.app/](https://transformer-question-answering.streamlit.app/)).
---

## 📌 Task Overview

- **Dataset**: [SQuAD v1.1]([https://rajpurkar.github.io/SQuad-explorer/](https://www.kaggle.com/datasets/stanfordu/stanford-question-answering-dataset)) (Stanford Question Answering Dataset)
- **Goal**: Given a context passage and a question, extract the correct answer span.
- **Models Evaluated**: BERT and DistilBERT (both fine-tuned on SQuAD v1.1)
- **Metrics**: Exact Match (EM) and F1 Score
- **Bonus**: Streamlit UI + MLflow model comparison

---

## 🧠 Models Compared

| Model | Hugging Face ID | Parameters | Speed | Accuracy |
|------|------------------|-----------|-------|----------|
| **BERT** | [`csarron/bert-base-uncased-squad-v1`](https://huggingface.co/csarron/bert-base-uncased-squad-v1) | ~110M | Slower | Higher (EM: ~81.6, F1: ~88.1) |
| **DistilBERT** | [`distilbert-base-uncased-distilled-squad`](https://huggingface.co/distilbert-base-uncased-distilled-squad) | ~66M | Faster | Slightly lower (EM: ~78.3, F1: ~85.5) |

> ✅ Both models are **fine-tuned on SQuAD v1.1** → valid for fair comparison.  

---

## 📂 Project Structure

```
transformer-question-answering/
├── data/
│   └── dev-v1.1.json      # SQuAD development set (used for evaluation)
├── evaluate_models.py     # Evaluate models on SQuAD dev set + log to MLflow
├── app.py                 # Streamlit web app for interactive QA
├── requirements.txt       # Minimal dependencies 
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/transformer-question-answering.git
cd transformer-question-answering
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```


### 4. Run Evaluation (Optional)
```bash
python evaluate_models.py
```
- Evaluates models on a **sample of 2000 SQuAD dev examples** (~10–15 min on CPU).
- Logs EM/F1 scores and comparison plots to **MLflow**.

### 5. Launch Streamlit App
```bash
streamlit run app.py
```
- Enter any **context + question** → get an answer instantly!

---

## 📊 Evaluation Results (Sample of 2000 Examples)

| Model | Exact Match | F1 Score | Duration (CPU) |
|-------|-------------|----------|----------------|
| BERT | 81.60 | 88.07 | ~7.2 min |
| DistilBERT | 78.25 | 85.46 | ~4.0 min |

> 📈 **Trade-off**: BERT is more accurate; DistilBERT is faster and lighter.

---

## 🛠️ Key Features

- **No training required** — uses pre-fine-tuned Hugging Face models
- **MLflow integration** — track and compare model performance
- **Lightweight Streamlit UI** — test any passage/question combo
- **Cross-platform** — works on Windows, Mac, Linux, and cloud
- **Efficient caching** — models load once per session (`@st.cache_resource`)

---

## 📚 References

- [SQuAD Dataset](https://rajpurkar.github.io/SQuad-explorer/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [MLflow](https://mlflow.org/)
- [Streamlit](https://streamlit.io/)

---

> ✨ **Try it live**: Ask *"Where is the Eiffel Tower?"* with context *"The Eiffel Tower is in Paris."* → Get *"Paris"* instantly!
