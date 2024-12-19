import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.vgg import VGG16_Weights

class VQAModel(nn.Module):
    def __init__(self, vocab_size):
        super(VQAModel, self).__init__()

        # Save vocab size
        self.vocab_size = vocab_size

        # Image feature extractor (VGGNet)
        self.image_encoder = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.image_encoder.classifier = nn.Sequential(*list(self.image_encoder.classifier.children())[:-1])  # Output 4096
        
        # Freeze VGG16 parameters
        for param in self.image_encoder.parameters():
            param.requires_grad = False

        self.image_fc = nn.Linear(4096, 1024)

        # Text feature extractor (LSTM)
        self.text_lstm = nn.LSTM(input_size=300, hidden_size=512, num_layers=2, batch_first=True)
        self.text_fc = nn.Linear(512, 1024)  # Add this line to define text_fc
        
        # Word embedding
        self.word_embedding = nn.Sequential(
            nn.Linear(vocab_size, 300),
            nn.Tanh()
        )

        # Fusion and classifier
        self.pointwise_mul = nn.Linear(1024, 1024)
        self.classifier = nn.Sequential(
            nn.Linear(1024, 1000),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(1000, 1000)  # Predict probability of 1000 words in answers.
        )

    def forward(self, image, question):
        # Image features
        image_features = self.image_encoder(image)
        image_features = self.image_fc(image_features)
        
        # Encoding question
        embedded = self.word_embedding(question.data)
        question_embedded = question._replace(data=embedded)

        # Text features
        _, (h_n, c_n) = self.text_lstm(question_embedded)
        text_features = torch.cat((h_n[-1], c_n[-1]), dim=1)

        # Fusion
        combined_features = image_features * text_features

        # Classification
        output = self.classifier(combined_features)
        return nn.Softmax(dim=1)(output) 