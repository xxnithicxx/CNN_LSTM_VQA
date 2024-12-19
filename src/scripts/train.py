import sys
import os
import torch
import logging
import wandb

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from models.vqa_model import VQAModel
from utils.dataset import VQADataset, vqa_collate_fn
from utils.helpers import load_config
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torchvision import transforms
from tqdm import tqdm

# Load config
config = load_config("./src/config/config.yaml")

log_name = f'train_{config["train"]["epochs"]}_{config["dataset"]["max_samples"]}'
# Set up logger
logging.basicConfig(
    filename=os.path.join(config["output_path"], log_name + ".log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger()

# Initialize WandB
wandb.init(project="vqa-training", name=log_name, config=config)

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")
wandb.config.device = str(DEVICE)

# Dataset and DataLoader
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config["dataset"]["mean"], std=config["dataset"]["std"]
        ),
    ]
)

dataset = VQADataset(
    data_dir=config["dataset"]["path"],
    image_dir=config["dataset"]["image_path"],
    transform=transform,
    max_samples=config["dataset"]["max_samples"],
)
train_size = int(config["train"]["train_split"] * len(dataset))
val_size = int(config["train"]["val_split"] * len(dataset))
test_size = len(dataset) - train_size - val_size
train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(
    train_set,
    batch_size=config["train"]["batch_size"],
    shuffle=True,
    collate_fn=vqa_collate_fn,
)
val_loader = DataLoader(
    val_set,
    batch_size=config["train"]["batch_size"],
    shuffle=False,
    collate_fn=vqa_collate_fn,
)
logger.info(f"Dataset sizes - Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")
wandb.log({"train_size": len(train_set), "val_size": len(val_set), "test_size": len(test_set)})

# Get require parameters for init model
vocab_size = len(dataset.word2idx)

# Model
model = VQAModel(vocab_size=vocab_size).to(DEVICE)
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=config["train"]["learning_rate"])
logger.info("Model, criterion, and optimizer initialized.")
wandb.watch(model, log="all", log_freq=10)

best_val_loss = float('inf')
best_model_path = None

# Training
for epoch in range(config["train"]["epochs"]):
    model.train()
    running_loss = 0.0

    for images, questions, targets in tqdm(train_loader):
        images, questions, targets = (
            images.to(DEVICE),
            questions.to(DEVICE),
            targets.to(DEVICE),
        )

        optimizer.zero_grad()
        outputs = model(images, questions)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(
        f"Epoch [{epoch+1}/{config['train']['epochs']}], Loss: {running_loss / len(train_loader):.4f}"
    )

    epoch_loss = running_loss / len(train_loader)
    logger.info(f"Epoch {epoch+1}/{config['train']['epochs']} - Loss: {epoch_loss:.4f}")
    wandb.log({"epoch": epoch + 1, "train_loss": epoch_loss})

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, questions, targets in val_loader:
            images, questions, targets = (
                images.to(DEVICE),
                questions.to(DEVICE),
                targets.to(DEVICE),
            )

            outputs = model(images, questions)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

    print(
        f"Validation Loss: {val_loss / len(val_loader):.4f}, Accuracy: {100 * correct / total:.2f}%"
    )
    
    val_loss /= len(val_loader)
    val_accuracy = 100 * correct / total
    logger.info(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")
    wandb.log({"epoch": epoch + 1, "val_loss": val_loss, "val_accuracy": val_accuracy})
    
    if not os.path.exists(config["output_path"]):
        os.makedirs(config["output_path"], exist_ok=True)
    
    save_path = config["output_path"] + f"model_{epoch+1}.pth"
    torch.save(model.state_dict(), save_path)
    logger.info(f"Model checkpoint saved to {save_path}")

    # Save the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_path = os.path.join(config["output_path"], "best_model.pth")
        torch.save(model.state_dict(), best_model_path)
        logger.info(f"Best model updated with Validation Loss: {best_val_loss:.4f}")
        wandb.log({"best_val_loss": best_val_loss})

logger.info(f"Training complete. Best model saved to {best_model_path} with Validation Loss: {best_val_loss:.4f}")
wandb.log({"best_model_path": best_model_path, "final_best_val_loss": best_val_loss})
wandb.finish()