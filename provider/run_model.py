import evaluate
from openai import AzureOpenAI
import os
import pandas as pd
import datasets
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import evaluate


client = AzureOpenAI(azure_endpoint='https://oai-cbipm-01.openai.azure.com/',
                     api_key=os.environ["OPENAI_API_KEY"],
                     api_version="2023-12-01-preview")  # 2023-12-01-preview <- highest version number as of 1/12/23

deployment = "Deployment"

dataset = pd.read_csv("../../psych_nlp/sentiment-analysis/data/sentences_MD-labels_GPT.csv")
# replace = {"neutral": 0, "negative": 1, "positive": 2}
# dataset["labels"] = dataset["MD_label"].map(replace)
# labels = dataset["labels"]
sentences = dataset["language"].to_json()

print(sentences)

messages = [{"role": "system", "content": "You are a medical doctor."},
            {"role": "user", "content": "As a medical doctor, you write many clinical notes about patients.\n"
                                        "Your task is to analyze the sentiment of a series of sentences you wrote about patients.\n"
                                        "For each sentence, what is your attitude towards the patient you wrote about?\n"
                                        "Please assign a sentiment score of negative, neutral, or positive for each sentence.\n"
                                        "Sentences:\n"
                                        "{\"0\":\"55 yo male with XXX, h/o asthma, BIB police for threatening behavior, disorganization and paranoia in the setting of medication non-adherence\","
                                        "\"1\":\"Sister saw pt often while he was at XXX b/c she works the night shift and would visit him on her breaks; states he was always quite sweet, not agitated, but sometimes thought she was their  mother, or another sister\","
                                        "\"2\":\"She does not come regularly for her appointments and has poor compliance with tx care\"}"
             },
            {"role": "assistant", "content": "{\"0\":\"Neutral,\"1\":\"Positive,\"2\":\"Negative}"},
            {"role": "user", "content": f"Complete the same task with these sentences:\n{sentences}"}]

response = client.chat.completions.create(model=deployment, messages=messages, temperature=0, seed=42)

output = response.choices[0].message.content
# predictions = pd.read_json(output, orient="index")
# predictions.columns.values[0] = "predictions"
# predictions["predictions"] = predictions[].map(replace)
print(output)

# def compute_metrics():
#     scmetrics.add_batch(predictions=predictions, references=labels)
#     return scmetrics.compute()
#
#
# scmetrics = evaluate.load("scmetrics")

# 10k tokens/min
