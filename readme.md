## VQA With CNN and RNN

This project implements a Visual Question Answering (VQA) model to answer question in an image. The architecture combines features from both image and text inputs and uses a fusion mechanism to make predictions.

### Project Structure
- `config/`: Contains configuration files.
- `data/`: Holds raw and processed datasets.
- `models/`: Contains model definitions.
- `notebooks/`: Jupyter notebooks for experimentation.
- `scripts/`: Scripts for training and preprocessing.
- `utils/`: Utility files and helper functions.
- `tests/`: Unit tests for the project.

```bash
vqa_counting/
├── config/
│   ├── config.yaml  # Configuration file for hyperparameters, paths, etc.
├── data/
│   ├── raw/  # Raw dataset (downloaded VQAv2 files)
│   ├── processed/  # Preprocessed dataset files
├── models/
│   ├── vqa_model.py  # VQACountingModel class implementation
├── notebooks/
│   ├── experiments.ipynb  # Jupyter notebooks for experimentation
├── scripts/
│   ├── train.py  # Main training script
│   ├── preprocess.py  # Data preprocessing script
├── utils/
│   ├── dataset.py  # VQADataset class and dataset utilities
│   ├── helpers.py  # Helper functions
├── tests/
│   ├── test_model.py  # Unit tests for the model
├── .gitignore  # Git ignore file for unnecessary files
└── README.md  # Project description and instructions

```

### Setup
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd vqa_counting
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Modify `config/config.yaml` to set paths and hyperparameters.

### Running the Project
- Preprocess the dataset to get the list of most frequent answers:
  ```bash
  python src/scripts/preprocess.py
  ```
- Train the model:
  ```bash
  python src/scripts/train.py
  ```

### Configuration
Modify `config/config.yaml` to customize the following:
- Paths to the dataset.
- Hyperparameters like learning rate, batch size, and epochs.