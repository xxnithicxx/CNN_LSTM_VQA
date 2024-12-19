import torch
import json
import os
import nltk

from PIL import Image
from collections import defaultdict
from torch.utils.data import Dataset
from nltk.tokenize import word_tokenize
from torch.nn.utils.rnn import pack_sequence
from utils.helpers import load_config

# Load config
config = load_config("./src/config/config.yaml")

def vqa_collate_fn(batch):
    """
    Custom collate function for VQADataset to pack questions (as one-hot) and prepare answers for cross-entropy loss.
    Args:
        batch (list): List of tuples (image, question, answer, vocab_size) from the dataset.

    Returns:
        tuple: Batched images, one-hot encoded questions, answer indices, question lengths.
    """
    images, questions, answers, vocab_size = zip(*batch)

    # Convert images to a tensor
    images = torch.stack(images)

    # One-hot encode questions
    question_tensors = [torch.zeros(len(q), vocab_size[0]) for q in questions]
    for i, q in enumerate(questions):
        for j, token_idx in enumerate(q):
            question_tensors[i][j][token_idx] = 1

    # Sort questions by length (required for pack_sequence)
    question_lengths = [len(q) for q in question_tensors]
    sorted_indices = sorted(
        range(len(question_tensors)), key=lambda i: question_lengths[i], reverse=True
    )
    question_tensors = [question_tensors[i] for i in sorted_indices]
    images = images[torch.tensor(sorted_indices)]
    answers = [answers[i] for i in sorted_indices]

    # Pack questions (as one-hot tensors)
    packed_questions = pack_sequence(
        [torch.tensor(q, dtype=torch.float32) for q in question_tensors],
        enforce_sorted=True,
    )

    return images, packed_questions, torch.tensor(answers)


class VQADataset(Dataset):
    def __init__(self, data_dir, image_dir, transform=None, max_samples=None):
        self.data_dir = data_dir
        self.image_dir = image_dir
        self.transform = transform
        self.max_samples = max_samples
        self.data = self._load_data()

        # Create a list for making word2idx and idx2word
        question_list = [question_data["question"] for question_data, _ in self.data]

        self.word2idx, self.idx2word = self._create_word2idx(question_list)
        self.ans2idx, self.idx2ans = self._create_ans2idx()

    def _load_data(self):
        questions_path = os.path.join(
            self.data_dir, "v2_OpenEnded_mscoco_train2014_questions.json"
        )
        annotations_path = os.path.join(
            self.data_dir, "v2_mscoco_train2014_annotations.json"
        )

        with open(questions_path, "r") as q_file, open(annotations_path, "r") as a_file:
            questions = json.load(q_file)["questions"]
            annotations = json.load(a_file)["annotations"]
            
        # Get the top 1000 answers
        freq_answers_path = os.path.join(config["project_path"], "src/data/processed/top_1000_answers_words.json")
        with open(freq_answers_path, "r") as file:
            top_1000_answers = json.load(file)
            
        # Convert the top 1000 answers to a set for faster lookup
        top_1000_answers_set = set(top_1000_answers.keys())
        
        # Filter out the annotations that are not in the top 1000 answers
        annotations = [
            annotation for annotation in annotations if annotation["multiple_choice_answer"] in top_1000_answers_set
        ]
        
        # Filter out the questions that have been filtered out
        question_ids = set([annotation["question_id"] for annotation in annotations])
        questions = [question for question in questions if question["question_id"] in question_ids]
        
        if self.max_samples:
            questions = questions[: self.max_samples]
            annotations = annotations[: self.max_samples]

        return list(zip(questions, annotations))

    def _create_word2idx(self, list_of_text):
        text = " ".join(list_of_text)

        # Tokenize the text
        tokens = word_tokenize(text.lower())

        # Create the word index
        word2idx = defaultdict()
        for index, word in enumerate(tokens):
            word2idx[word] = index

        # Iterate over the word2idx and modify the index from 0 to len(word2idx)
        for idx, word in enumerate(word2idx):
            word2idx[word] = idx

        # Create the index word
        idx2word = defaultdict()
        for word, idx in word2idx.items():
            idx2word[idx] = word

        return word2idx, idx2word

    def _create_ans2idx(self):
        # Get the top 1000 answers
        freq_answers_path = os.path.join(config["project_path"], "src/data/processed/top_1000_answers_words.json")
        with open(freq_answers_path, "r") as file:
            top_1000_answers = json.load(file)
            
        # Convert the top 1000 answers to a set for faster lookup
        top_1000_answers_set = set(top_1000_answers.keys())

        # Create the word index
        ans2idx = defaultdict()
        for index, word in enumerate(top_1000_answers_set):
            ans2idx[word] = index
            
        # Create the index word
        idx2ans = defaultdict()
        for word, idx in ans2idx.items():
            idx2ans[idx] = word

        return ans2idx, idx2ans

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question_data, annotation_data = self.data[idx]
        image_id = question_data["image_id"]
        image_id = str(image_id).zfill(12)
        image_path = os.path.join(self.image_dir, f"COCO_train2014_{image_id}.jpg")

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        question = word_tokenize(question_data["question"].lower())
        question = [self.word2idx[word] for word in question]

        answer = self.ans2idx[annotation_data["multiple_choice_answer"]]

        return image, question, answer, len(self.word2idx)
