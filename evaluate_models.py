# evaluate_models.py
import os
import json
import mlflow
import matplotlib.pyplot as plt
import pandas as pd
from transformers import pipeline
import evaluate

# Disable symlink warning on Windows (optional but cleaner)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# --- Config ---
MODEL_MAP = {
    "BERT": "csarron/bert-base-uncased-squad-v1",
    "DistilBERT": "distilbert-base-uncased-distilled-squad"
}
DEV_PATH = r"data\raw\dev-v1.1.json"  
SAMPLE_SIZE = 2000  
MLFLOW_EXPERIMENT_NAME = "squad_model_comparison_SAMPLE"

# --- Setup ---
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
squad_metric = evaluate.load("squad")

# --- Load dev data ---
with open(DEV_PATH, "r", encoding="utf-8") as f:
    squad_data = json.load(f)

# --- Build list of (context, qa) pairs ---
all_examples = []
for article in squad_data["data"]:
    for para in article["paragraphs"]:
        for qa in para["qas"]:
            all_examples.append((para["context"], qa))
            if SAMPLE_SIZE and len(all_examples) >= SAMPLE_SIZE:
                break
        if SAMPLE_SIZE and len(all_examples) >= SAMPLE_SIZE:
            break
    if SAMPLE_SIZE and len(all_examples) >= SAMPLE_SIZE:
        break

print(f"Using {len(all_examples)} examples for evaluation.")

# --- Evaluate each model ---
results = []
for model_name, model_id in MODEL_MAP.items():
    print(f"\nEvaluating {model_name} ({model_id})...")

    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model_id", model_id)
        mlflow.log_param("sample_size", len(all_examples))

        # Load pipeline
        qa_pipeline = pipeline(
            "question-answering",
            model=model_id,
            tokenizer=model_id,
            device=-1  # CPU
        )

        predictions = []
        references = []

        for context, qa in all_examples:
            # Prediction
            try:
                result = qa_pipeline(question=qa["question"], context=context)
                pred_text = result["answer"]
            except Exception as e:
                pred_text = ""
                print(f"Error on {qa['id']}: {e}")

            predictions.append({
                "id": qa["id"],
                "prediction_text": pred_text
            })

            # Reference (flatten answers)
            answer_texts = [ans["text"] for ans in qa["answers"]]
            answer_starts = [ans["answer_start"] for ans in qa["answers"]]
            references.append({
                "id": qa["id"],
                "answers": {
                    "text": answer_texts,
                    "answer_start": answer_starts
                }
            })

        # Compute metrics
        metrics = squad_metric.compute(predictions=predictions, references=references)
        em = metrics["exact_match"]
        f1 = metrics["f1"]

        # Log to MLflow
        mlflow.log_metric("exact_match", em)
        mlflow.log_metric("f1", f1)

        # Save for plotting
        results.append({
            "Model": model_name,
            "Exact Match": em,
            "F1": f1
        })

        print(f"{model_name} → EM: {em:.2f}, F1: {f1:.2f}")

# --- Create and log comparison plot ---
results_df = pd.DataFrame(results)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))

results_df.plot(x="Model", y="Exact Match", kind="bar", ax=ax[0], color="skyblue", legend=False)
ax[0].set_title("Exact Match (%)")
ax[0].set_ylabel("Score")

results_df.plot(x="Model", y="F1", kind="bar", ax=ax[1], color="lightgreen", legend=False)
ax[1].set_title("F1 Score (%)")
ax[1].set_ylabel("Score")

plt.tight_layout()

# Log to MLflow
mlflow.log_figure(fig, "model_comparison.png")

# Save results
results_df.to_csv("model_results.csv", index=False)
mlflow.log_artifact("model_results.csv")

print("\n Evaluation complete! Check MLflow UI for results and graphs.")