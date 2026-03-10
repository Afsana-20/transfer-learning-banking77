import gradio as gr
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import os
import torch

model_path = os.path.join(os.path.dirname(__file__), "my_model")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    dtype=torch.float32,
    low_cpu_mem_usage=True
)

classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    top_k=3,
    device=-1
)

dataset = load_dataset("banking77", split="train[:1%]")
label_names = dataset.features["label"].names

def predict(text):
    results = classifier(text)
    output = ""
    for r in results[0]:
        label_num = int(r["label"].split("_")[-1])
        label_name = label_names[label_num]
        confidence = r["score"] * 100
        output += f"🏷️ {label_name}  →  {confidence:.2f}%\n"
    return output

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(placeholder="Type a customer query...", label="Customer Message"),
    outputs=gr.Textbox(label="Top 3 Predicted Categories"),
    title="🤖 Customer Support Ticket Classifier",
    description="Fine-tuned DistilBERT on Banking77 — 92.66% Accuracy",
    examples=[
        ["I cant login to my account"],
        ["My card was charged twice"],
        ["How do I transfer money abroad?"],
        ["I lost my card"],
        ["What is the exchange rate today?"],
    ],
    cache_examples=False
)

demo.launch()
