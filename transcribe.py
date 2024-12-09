# Tacotron Model
## Tokenization

from torch.utils.data import random_split, DataLoader
from tacotron2 import synthesize
from collections import Counter
import matplotlib.pyplot as plt
import torch
import torchaudio
from torch.utils.data import Dataset
import os
import pandas as pd
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule, Trainer
from torchaudio.models import Tacotron2


class TextTokenizer:
    def __init__(self):
        # Define a character set. You can expand this if needed.
        self.char_to_id = {char: idx for idx, char in enumerate("abcdefghijklmnopqrstuvwxyz ")}  # Include space
        self.id_to_char = {idx: char for idx, char in enumerate("abcdefghijklmnopqrstuvwxyz ")}
    
    def encode(self, text):
        # Check if the input is a string; if not, pass it as is
        if isinstance(text, torch.Tensor):
            return text  # Already encoded, return as is
        if not isinstance(text, str):
            text = self.decode(ids=text)

        # Encode text to a list of indices and convert to a tensor
        token_indices = [self.char_to_id[char] for char in text.lower() if char in self.char_to_id]
        return torch.tensor(token_indices, dtype=torch.long)

    
    def decode(self, ids):
        # Decode a tensor or list of indices back to text
        if isinstance(ids, torch.Tensor):  # If input is a tensor, convert to a list
            ids = ids.tolist()
        if isinstance(ids, float):  # If input is a tensor, convert to a list
            return f'{ids}'
        return ''.join([self.id_to_char[idx] for idx in ids])

## Mel spectrogram
class CustomDataset(Dataset):
    def __init__(self, metadata_path, audio_dir, cfg):
        super().__init__()
        self.metadata = pd.read_csv(metadata_path, sep="|", header=None, names=["file", "text"])
        self.audio_dir = audio_dir

        # Initialize the tokenizer
        self.tokenizer = TextTokenizer()

        # Create a MelSpectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=22050,  # Set your dataset's sample rate
            n_mels=cfg["model"]["num_mels"],
            n_fft=1024,
            hop_length=256,
            win_length=1024,
        )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # Get file path and text
        row = self.metadata.iloc[idx]
        audio_path = os.path.join(self.audio_dir, row["file"])
        waveform, sample_rate = torchaudio.load(audio_path)

        # Tokenize text (already returns a tensor)
        text_tensor = self.tokenizer.encode(row["text"])

        # Resample if necessary
        if sample_rate != 22050:
            resample = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=22050)
            waveform = resample(waveform)

        # Convert waveform to mel-spectrogram
        mel_spectrogram = self.mel_transform(waveform).squeeze(0)  # Squeeze only batch dim if present

        # Return tokenized text and mel-spectrogram
        return text_tensor, mel_spectrogram

def collate_fn(batch):
    # Separate texts and mel spectrograms from the batch
    texts, mel_spectrograms = zip(*batch)

    # Debugging: Print the shape of the first mel spectrogram
    print(f"First mel spectrogram shape (before padding): {mel_spectrograms[0].shape}")

    # Pad texts (assumes each text is already a 1D tensor)
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)

    # Ensure mel spectrograms are padded along the time dimension
    max_len = max(mel.shape[1] for mel in mel_spectrograms)  # Max time dimension
    padded_mels = []
    for mel in mel_spectrograms:
        pad_len = max_len - mel.shape[1]
        padded_mel = torch.nn.functional.pad(
            mel,
            (0, pad_len),  # Padding along the time dimension
            mode="constant",
            value=0.0,  # Use 0.0 for padding
        )
        padded_mels.append(padded_mel)

    # Stack into a batch tensor
    mel_batch = torch.stack(padded_mels, dim=0)  # (batch_size, num_mels, max_len)

    return padded_texts, mel_batch

## Train
class Tacotron2TTS(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.model = Tacotron2(cfg["model"])
        self.mel_channels = cfg["model"]["mel_channels"]  # Corresponds to mel_channels
        self.hidden_channels = cfg["model"]["hidden_channels"]
        self.attention_dim = cfg["model"]["attention_dim"]
        self.default_mel_length = 80
        self.cfg = cfg
        # Add the embedding layer
        self.embedding = nn.Embedding(cfg["model"]["vocab_size"], cfg["model"]["embedding_dim"])  # Define the embedding layer
    
    def forward(self, text, mel_spectrogram, token_lengths=None ,mel_specgram_lengths=None):
        # Ensure `mel_spectrogram` is a tensor, or create a placeholder if `None`
        if mel_spectrogram is None:
            mel_spectrogram = torch.zeros((text.size(0), self.mel_channels, self.default_mel_length), 
                                        dtype=torch.float32, device=self.device)
        
        mel_spectrogram = torch.tensor(mel_spectrogram, dtype=torch.float32) if not isinstance(mel_spectrogram, torch.Tensor) else mel_spectrogram
        
        # Ensure token_lengths is a tensor
        token_lengths = torch.tensor(token_lengths, dtype=torch.long) if not isinstance(token_lengths, torch.Tensor) else token_lengths
        mel_specgram_lengths = torch.tensor(mel_spectrogram, dtype=torch.long) if not isinstance(mel_specgram_lengths, torch.Tensor) else mel_specgram_lengths
        # Sort token_lengths in descending order and get the sorted indices
        sorted_lengths, sorted_idx = torch.sort(token_lengths, descending=True)
        sorted_text = text[sorted_idx]  # Sort text accordingly
        
        # Pack the padded sequences (tokens) with sorted lengths
        packed_input = nn.utils.rnn.pack_padded_sequence(sorted_text, sorted_lengths, batch_first=True, enforce_sorted=False)

        # Forward pass through the model (you will use `packed_input` now)
        embedded_inputs = self.embedding(packed_input.data).transpose(1, 2)  # Example of how to handle packed input
        
        encoder_outputs = self.encoder(embedded_inputs, sorted_lengths)
        mel_specgram, gate_outputs, alignments = self.decoder(encoder_outputs, mel_spectrogram, memory_lengths=sorted_lengths)
        
        mel_specgram_postnet = self.postnet(mel_specgram)
        
        return mel_specgram_postnet

    def training_step(self, batch, batch_idx):
        # Unpack the batch
        text, mel_spectrogram = batch

        # Ensure token lengths and spectrogram lengths are tensors
        token_lengths = torch.tensor([text.shape[1]] * text.shape[0], dtype=torch.long, device=text.device)
        mel_specgram_lengths = torch.tensor([mel_spectrogram.shape[2]] * mel_spectrogram.shape[0], dtype=torch.long, device=mel_spectrogram.device)

        # Forward pass
        mel_spectrogram_pred = self.forward(text=text, mel_spectrogram=mel_spectrogram, token_lengths=token_lengths, mel_specgram_lengths=mel_specgram_lengths)
        
        # Compute loss
        loss = torch.nn.functional.mse_loss(mel_spectrogram_pred, mel_spectrogram)
        
        # Log the loss for monitoring
        # Print debug information 
        # print(f"text type: {type(text)}, text shape: {text.shape}") 
        # print(f"mel_spectrogram type: {type(mel_spectrogram)}, mel_spectrogram shape: {mel_spectrogram.shape}")
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        text, mel_spectrogram = batch
        token_lengths = torch.tensor([text.shape[1]] * text.shape[0], dtype=torch.long, device=text.device)  # Example: length per sample
        mel_spectrogram_pred = self.forward(text=text, mel_spectrogram=mel_spectrogram, token_lengths=token_lengths)
        loss = torch.nn.functional.mse_loss(mel_spectrogram_pred, mel_spectrogram)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        text, mel_spectrogram = batch
        token_lengths = torch.tensor([text.shape[1]] * text.shape[0], dtype=torch.long, device=text.device)  # Example: length per sample
        mel_spectrogram_pred = self.forward(text=text, mel_spectrogram=mel_spectrogram, token_lengths=token_lengths)
        loss = torch.nn.functional.mse_loss(mel_spectrogram_pred, mel_spectrogram)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.cfg["trainer"]["lr"])

    def print_batch_properties(self, batch):
        """
        Prints the properties of the batch components.

        Args:
            batch: A batch of data, expected to be a tuple or a list.
        """
        if not isinstance(batch, (tuple, list)):
            print("Batch is not a tuple or list. Type:", type(batch))
            return

        print("Batch contains", len(batch), "elements.")
        for i, item in enumerate(batch):
            print(f"--- Element {i} ---")
            print(f"Type: {type(item)}")
            if isinstance(item, torch.Tensor):
                print(f"Shape: {item.shape}")
                print(f"Dtype: {item.dtype}")
            elif hasattr(item, "__len__"):
                print(f"Length: {len(item)}")
            else:
                print("No additional properties available.")
            print("-------------------")


def get_vocab_size(texts):
    # Combine all texts and count unique characters
    all_characters = ''.join(texts)
    unique_characters = set(all_characters)
    return len(unique_characters)

# Example
texts = ["hello world", "how are you"]
vocab_size = get_vocab_size(texts)
print(f"Vocabulary size: {vocab_size}")

# Configuration
cfg = {
    "model": {
        "mel_channels": 80,
        "hidden_channels": 128,
        "attention_dim": 128,
        "vocab_size": vocab_size,
        "embedding_dim":512
    },
    "trainer": {
        "max_epochs": 10,
        "lr": 1e-3,
        "batch_size": 16,
    },
}

# Paths
metadata_path = "dataset/metadata.csv"
audio_dir = "dataset/wavs"
dataset = CustomDataset(metadata_path="dataset/metadata.csv", audio_dir="dataset/wavs", cfg=cfg)

# Assume dataset is already created
dataset_size = len(dataset)
val_size = int(0.2 * dataset_size)  # 20% for validation
train_size = dataset_size - val_size

# Split dataset into training and validation sets
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(
                    train_dataset,
                    batch_size=cfg["trainer"]["batch_size"],
                    shuffle=True,
                    collate_fn=collate_fn
                )
val_loader = DataLoader(
    train_dataset,
    batch_size=cfg["trainer"]["batch_size"], 
    shuffle=False,
    collate_fn=collate_fn
    )

def fit(model, train_loader, val_loader, optimizer, criterion, device, epochs, scheduler=None):
    model.to(device)
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            # Unpack the batch
            text, mel_spectrogram = batch
            text, mel_spectrogram = text.to(device), mel_spectrogram.to(device)

            # Compute token_lengths and mel_specgram_lengths
            token_lengths = torch.tensor([text.shape[1]] * text.shape[0], dtype=torch.long, device=device)
            mel_specgram_lengths = torch.tensor([mel_spectrogram.shape[2]] * mel_spectrogram.shape[0], dtype=torch.long, device=device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(text, mel_spectrogram, token_lengths, mel_specgram_lengths)
            loss = criterion(outputs, mel_spectrogram)
            
            # Backward pass and optimisation
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        history["train_loss"].append(train_loss / len(train_loader))

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                # Unpack and move to device
                text, mel_spectrogram = batch
                text, mel_spectrogram = text.to(device), mel_spectrogram.to(device)

                token_lengths = torch.tensor([text.shape[1]] * text.shape[0], dtype=torch.long, device=device)
                mel_specgram_lengths = torch.tensor([mel_spectrogram.shape[2]] * mel_spectrogram.shape[0], dtype=torch.long, device=device)

                # Forward pass
                outputs = model(text, mel_spectrogram, token_lengths, mel_specgram_lengths)
                loss = criterion(outputs, mel_spectrogram)
                val_loss += loss.item()

        history["val_loss"].append(val_loss / len(val_loader))

        if scheduler:
            scheduler.step()

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {history['train_loss'][-1]:.4f}, Val Loss: {history['val_loss'][-1]:.4f}")

    return history

# Model
model = Tacotron2TTS(cfg)

# Move model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define criterion and optimizer
criterion = nn.MSELoss()
learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
epochs = 10
history = fit(model, train_loader, val_loader, optimizer, criterion, device, epochs)

# Plot training and validation loss

plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Validation Loss")
plt.show()

## Synthesise


audio = synthesize(
    checkpoint="checkpoints/latest_model.pth",
    text="Hello, this is a test synthesis."
)
with open("output.wav", "wb") as f:
    f.write(audio)