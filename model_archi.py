import torch
import torch.nn as nn
from torchvision import models

class SimpleImage2Text(nn.Module):
    def __init__(self, vocab_size, hidden_dim=256, max_length=512):
        super().__init__()
        # Pretrained image encoder (ResNet18)
        self.encoder = models.resnet18(pretrained=True)
        self.encoder.fc = nn.Linear(self.encoder.fc.in_features, hidden_dim)
        # Decoder
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.max_length = max_length

    def forward(self, images, targets=None, teacher_forcing_ratio=0.5):
        batch_size = images.size(0)
        features = self.encoder(images).unsqueeze(1)  # (B, 1, H)
        inputs = features
        outputs = []
        hidden = None

        if targets is not None:
            # Training with teacher forcing
            for t in range(self.max_length):
                if t == 0:
                    inp = features
                else:
                    inp = self.embedding(targets[:, t-1]).unsqueeze(1)
                out, hidden = self.rnn(inp, hidden)
                logits = self.fc(out.squeeze(1))
                outputs.append(logits)
            outputs = torch.stack(outputs, dim=1)
            return outputs
        else:
            # Inference (greedy)
            inp = features
            for t in range(self.max_length):
                out, hidden = self.rnn(inp, hidden)
                logits = self.fc(out.squeeze(1))
                outputs.append(logits)
                pred = logits.argmax(dim=1)
                inp = self.embedding(pred).unsqueeze(1)
            outputs = torch.stack(outputs, dim=1)
            return outputs