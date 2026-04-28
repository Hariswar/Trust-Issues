import torch
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, classification_report

# import preprocessing functions
sys.path.insert(0, str(Path(__file__).parent.parent))
from preprocessing import (
    clean_normalize_dataset,
    split_dataset,
    read_train,
    read_test,
    read_evaluation,
    read_dataset_two,
    dataset_two_train,
    dataset_two_test,
    dataset_two_validation,
    train,
    test,
    evaluation,
)


# ── Config ─────────────────────────────────────────────────────────────────
# MODEL_NAME  : Hugging Face model hub identifier for bert-base-uncased
# MAX_LEN     : Max number of tokens per input (title + text combined).
#               BERT supports 512 but 256 saves memory with minimal loss.
# BATCH_SIZE  : Number of samples processed per gradient update step.
# EPOCHS      : How many full passes over the training set.
# LR          : Learning rate for AdamW; 2e-5 is the standard BERT fine-tune.
# DEVICE      : Automatically uses GPU if available, otherwise CPU.
# NUM_LABELS  : Binary classification — real (0) vs fake (1).
CONFIG = {
    "MODEL_NAME": "bert-base-uncased",
    "MAX_LEN": 128,
    "BATCH_SIZE": 32,
    "EPOCHS": 3,
    "LR": 2e-5,
    "DEVICE": (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    ),
    "NUM_LABELS": 2,
}

OUTPUT_DIR = Path(__file__).parent.parent / "outputs" 
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def save_model(model, tokenizer, dataset_name: str) -> Path:
    """
    Save a fine-tuned model and its tokenizer to disk so they can be
    reloaded later without retraining.
 
    Saves to outputs/<dataset_name>_<timestamp>/ using Hugging Face's
    save_pretrained format
    
    Reload later with:
        AutoModelForSequenceClassification.from_pretrained(save_dir)
        AutoTokenizer.from_pretrained(save_dir)
 
    Parameters
    ----------
        model : (AutoModelForSequenceClassification)
            The fine-tuned model to persist.
        tokenizer : (AutoTokenizer)
            The tokenizer paired with the model.
        dataset_name : (str)
            Short label used in the directory name, e.g. 'dataset1'.
 
    Returns
    -------
        Path : 
            The directory the model was saved to.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = OUTPUT_DIR / f"{dataset_name}_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"  Model saved → {save_dir}")
    return save_dir


# ── Dataset class ────────────────────────────────────────────────────────
class FakeNewsDataset(Dataset):
    """
    A PyTorch Dataset that tokenizes fake-news article data for BERT.
    Each sample concatenates the article's title and body text into a single
    BERT input using the [CLS] title [SEP] text [SEP] format

    Parameters
    ----------
        df (pd.DataFrame) : A cleaned DataFrame containing at minimum the columns:
            - 'title' (str) : Title of the article.
            - 'text' (str) : Body text of the article.
            - 'label' (int) : 0=real / 1=fake.
        tokenizer (transformers.PreTrainedTokenizer) : 
            A Hugging Face tokenizer matched to the BERT model being used.
        max_len (int) : Maximum token sequence length. Sequences longer than this are
            truncated; shorter ones are padded to this length.

    Returns
    -------
        dict :
            A dictionary with keys:
            - 'input_ids' (torch.Tensor) : torch.Tensor of shape (max_len,) — token IDs.
            - 'attention_mask' (torch.Tensor) : torch.Tensor of shape (max_len,) — 1 for real
                                tokens, 0 for padding.
            - 'labels' (torch.Tensor) : torch.Tensor scalar — the ground-truth class.
    """

    def __init__(self, df: pd.DataFrame, tokenizer: AutoTokenizer, max_len: int):
        self.data = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        """
        Parameters
        ----------
            None

        Returns
        -------
            (int) : Number of rows in the underlying DataFrame.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieve and tokenize a single sample by index.

        Combines title and text as a two-segment BERT input, applies padding
        and truncation to MAX_LEN, and returns tensors ready for the model.

        Parameters
        ----------
            idx (int) : Row index of the sample to retrieve.

        Returns
        -------
            dict :
                Dictionary containing 'input_ids', 'attention_mask', and 'labels' as torch.Tensors.
        """
        title = str(self.data.loc[idx, "title"])
        text = str(self.data.loc[idx, "text"])
        label = int(self.data.loc[idx, "label"])

        encoding = self.tokenizer(
            title,
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def make_loader(df: pd.DataFrame, tokenizer, shuffle: bool) -> DataLoader:
    """
    Wrap a DataFrame in a FakeNewsDataset and return a DataLoader.

    Parameters
    ----------
        df (pd.DataFrame) :
            The split of data (train / test / validation) to wrap.
        tokenizer (transformers.PreTrainedTokenizer) :
            Tokenizer used to encode each sample inside FakeNewsDataset.
        shuffle (bool) :
            Whether to shuffle the data before each epoch.
            Should be True for training splits and False for eval splits
            to keep results reproducible.

    Returns
    -------
        torch.utils.data.DataLoader :
            A DataLoader yielding batches of tokenized samples. Uses
            num_workers=2 for parallel data loading and pin_memory=True
            to speed up CPU-to-GPU transfer.
    """
    ds = FakeNewsDataset(df, tokenizer, CONFIG["MAX_LEN"])
    use_pin = CONFIG["DEVICE"].type == "cuda"
    return DataLoader(
        ds,
        batch_size=CONFIG["BATCH_SIZE"],
        shuffle=shuffle,
        num_workers=0,
        pin_memory=use_pin,
    )

# ── Helper functions ───────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler):
    """
    Run one full training epoch over the provided DataLoader.

    For each batch: 
        - forward pass
        - computes loss
        - backward pass
        - gradient clipping
        - optimizer + scheduler step
    Accumulate predictions and ground-truth labels to compute epoch-level accuracy.

    Parameters
    ----------
        model : (transformers.AutoModelForSequenceClassification)
            The BERT classification model to train.
        loader : (torch.utils.data.DataLoader)
            DataLoader yielding batches of {'input_ids', 'attention_mask', 'labels'}.
        optimizer : (torch.optim.AdamW)
            The AdamW optimizer holding the current parameter states.
        scheduler : (transformers.get_linear_schedule_with_warmup)
            Learning rate scheduler; stepped once per batch (not per epoch).

    Returns
    -------
        float : Mean cross-entropy loss across all batches in the epoch.
        float : Fraction of correctly classified samples in the epoch (0.0-1.0).
    """
    # Set model to training mode
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []

    # Loop through each batch in loader
    for batch in loader:
        input_ids = batch["input_ids"].to(CONFIG["DEVICE"])
        attention_mask = batch["attention_mask"].to(CONFIG["DEVICE"])
        labels = batch["labels"].to(CONFIG["DEVICE"])

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = outputs.logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy

@torch.no_grad()
def evaluate(model, loader, split_name: str):
    """
    Evaluate the model on a given data split without updating weights.

    Runs inference over all batches, accumulates predictions, then prints
    a full sklearn classification report (precision, recall, F1 per class)
    alongside overall loss and accuracy.

    The @torch.no_grad() decorator disables gradient computation entirely,
    reducing memory usage and speeding up inference.

    Parameters
    ----------
    model : transformers.AutoModelForSequenceClassification
        The fine-tuned BERT model to evaluate. Should be moved to DEVICE.
    loader : torch.utils.data.DataLoader
        DataLoader for the evaluation split (test or validation).
    split_name : str
        A human-readable label printed in the output header, e.g.
        "Dataset 1 — Test" or "Dataset 2 — Validation".

    Returns
    -------
    avg_loss : float
        Mean cross-entropy loss across all batches.
    accuracy : float
        Overall classification accuracy across all samples (0.0 – 1.0).
    """
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(CONFIG["DEVICE"])
        attention_mask = batch["attention_mask"].to(CONFIG["DEVICE"])
        labels = batch["labels"].to(CONFIG["DEVICE"])

        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        total_loss += outputs.loss.item()

        preds = outputs.logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)

    print(f"\n── {split_name} Results ──────────────────────")
    print(f"  Loss:     {avg_loss:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(classification_report(all_labels, all_preds, target_names=["Real", "Fake"]))
    return avg_loss, accuracy


# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Model Configuration: {CONFIG}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["MODEL_NAME"])
    model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG["MODEL_NAME"], num_labels=CONFIG["NUM_LABELS"]
    ).to(CONFIG["DEVICE"])

    # ── Dataset 1: pre-split by source (evaluation / train / test) ──
    print("\n=== Dataset 1 ===")
    train_loader_1 = make_loader(train, tokenizer, shuffle=True)
    test_loader_1 = make_loader(test, tokenizer, shuffle=False)
    eval_loader_1 = make_loader(evaluation, tokenizer, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=CONFIG["LR"], weight_decay=0.01)
    total_steps = len(train_loader_1) * CONFIG["EPOCHS"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    for epoch in range(1, CONFIG["EPOCHS"] + 1):
        loss, acc = train_epoch(model, train_loader_1, optimizer, scheduler)
        print(
            f"Epoch {epoch}/{CONFIG['EPOCHS']}  train_loss={loss:.4f}  train_acc={acc:.4f}"
        )

    evaluate(model, test_loader_1, "Dataset 1 — Test")
    evaluate(model, eval_loader_1, "Dataset 1 — Evaluation")
    save_model(model, tokenizer, "dataset1")

    # ── Dataset 2: WELFake (already split by preprocessing.py) ──
    print("\n=== Dataset 2 (WELFake) ===")

    # Re-initialise model weights for a clean run on dataset 2
    model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG["MODEL_NAME"], num_labels=CONFIG["NUM_LABELS"]
    ).to(CONFIG["DEVICE"])

    train_loader_2 = make_loader(dataset_two_train, tokenizer, shuffle=True)
    test_loader_2 = make_loader(dataset_two_test, tokenizer, shuffle=False)
    val_loader_2 = make_loader(dataset_two_validation, tokenizer, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=CONFIG["LR"], weight_decay=0.01)
    total_steps = len(train_loader_2) * CONFIG["EPOCHS"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    for epoch in range(1, CONFIG["EPOCHS"] + 1):
        loss, acc = train_epoch(model, train_loader_2, optimizer, scheduler)
        print(
            f"Epoch {epoch}/{CONFIG['EPOCHS']}  "
            f"train_loss={loss:.4f}  train_acc={acc:.4f}"
        )

    evaluate(model, val_loader_2, "Dataset 2 — Validation")
    evaluate(model, test_loader_2, "Dataset 2 — Test")
    save_model(model, tokenizer, "dataset2")
