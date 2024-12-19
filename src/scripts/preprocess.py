import os
import sys
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from collections import Counter
from utils.helpers import load_config

# Load config
config = load_config("./src/config/config.yaml")

def extract_top_1000_answers(annotations_file, output_file, fred=1000):
    """
    Extracts the top 1000 most frequent answers from the VQA annotations file.
    
    Args:
        annotations_file (str): Path to the VQA annotations JSON file.
        output_file (str): Path where the processed JSON file will be saved.
    """
    # Load annotations
    with open(annotations_file, "r") as file:
        annotations = json.load(file)["annotations"]
    
    # Extract answers
    answer_counter = Counter()
    for entry in annotations:
        # Collect the first annotator's answer (like in the provided format)
        for answer in entry["answers"]:
            answer_text = answer["answer"]
            answer_counter[answer_text] += 1

    # Get the top 1000 most frequent answers
    top_1000_answers = dict(answer_counter.most_common(1000))

    # Check if file already exists
    if not os.path.exists(output_file):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save the top 1000 answers to the specified output file
    with open(output_file, "w") as file:
        json.dump(top_1000_answers, file, indent=4)
    
    print(f"Top 1000 answers saved to: {output_file}")

annotations_file = os.path.join(config["dataset"]["path"], "v2_mscoco_train2014_annotations.json")
output_file = os.path.join(config["project_path"], "src/data/processed/top_1000_answers_words.json")

extract_top_1000_answers(annotations_file, output_file)