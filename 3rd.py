import pandas as pd
from transformers import pipeline
from sklearn.metrics import accuracy_score, f1_score

# Load the dataset and create a summarization pipeline
dataset = pd.read_csv("D:\Data_Excel\summarization_dataset.csv")
summarizer = pipeline("summarization", model="t5-small")

# Define the evaluation metrics
metrics = {
    "accuracy": accuracy_score,
    "f1": f1_score
}

# Define the prompt designs
prompt_designs = [
    {"prompt": "Please summarize the article: {article}"},
    {"prompt": "Summarize the main points of the article: {article}"},
    {"prompt": "What is the main idea of the article: {article}?"},
    {"prompt": "Can you summarize the article for me: {article}?"},
    {"prompt": "What is the article about? Summarize it for me: {article}"},
]

# Evaluate each prompt design
results = []
for prompt_design in prompt_designs:
    results.append({
        "prompt": prompt_design["prompt"],
        "accuracy": [],
        "f1": []
    })
    for i, article in enumerate(dataset["article"]):
        input_text = f"{prompt_design['prompt'].format(article=article)}"
        output = summarizer(input_text, max_length=200)
        predicted_summary = output[0]["summary_text"]
        gold_summary = dataset["summary"][i]
        accuracy = accuracy_score([gold_summary], [predicted_summary])
        f1 = f1_score([gold_summary], [predicted_summary], average="macro")
        results[-1]["accuracy"].append(accuracy)
        results[-1]["f1"].append(f1)

# Calculate the average performance for each prompt design
average_results = {}
for result in results:
    average_accuracy = sum(result["accuracy"]) / len(result["accuracy"])
    average_f1 = sum(result["f1"]) / len(result["f1"])
    average_results[result["prompt"]] = {
        "accuracy": average_accuracy,
        "f1": average_f1
    }

# Print the results
print("Prompt\tAccuracy\tF1")
for prompt, result in average_results.items():
    print(f"{prompt}\t{result['accuracy']:.4f}\t{result['f1']:.4f}")

# Select the best prompt design based on the evaluation metrics
best_prompt = max(average_results, key=lambda x: (average_results[x]["f1"], average_results[x]["accuracy"]))
print(f"Best prompt: {best_prompt}")