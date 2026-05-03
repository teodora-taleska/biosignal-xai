import torch.nn as nn

class BaselineCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(12, 32, kernel_size=7, padding=3), nn.ReLU(),
            nn.MaxPool1d(2),                              # (32, 500)
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool1d(2),                              # (64, 250)
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),                      # (128, 1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_classes)
            # No sigmoid here — use BCEWithLogitsLoss
        )

    def forward(self, x):
        return self.classifier(self.encoder(x))