# src/model.py

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, : x.size(1)]


class TransformerCaptionModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        feature_dim=2048,
        d_model=512,
        num_heads=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        max_len=20,
        dropout=0.1,
    ):
        super().__init__()

        self.d_model = d_model

        # project image features to transformer dimension
        self.feature_proj = nn.Linear(feature_dim, d_model)

        # word embedding
        self.word_embedding = nn.Embedding(vocab_size, d_model)

        # positional encodings
        self.pos_encoder = PositionalEncoding(d_model)
        self.pos_decoder = PositionalEncoding(d_model)

        # transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
        )

        # output layer
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, image_features, captions):
        """
        image_features: (batch, regions, 2048)
        captions: (batch, seq_len)
        """

        # encode image features
        img = self.feature_proj(image_features)
        img = self.pos_encoder(img)
        img = img.permute(1, 0, 2)  # (regions, batch, d_model)

        # embed captions
        cap = self.word_embedding(captions) * math.sqrt(self.d_model)
        cap = self.pos_decoder(cap)
        cap = cap.permute(1, 0, 2)  # (seq_len, batch, d_model)

        # transformer forward
        output = self.transformer(img, cap)

        # project to vocab
        output = output.permute(1, 0, 2)  # (batch, seq_len, d_model)
        logits = self.output_layer(output)

        return logits

    @torch.no_grad()
    def generate(self, image_features, sos_idx, eos_idx, max_len=20):
        """
        Greedy caption generation
        image_features: (batch, regions, 2048)
        """
        self.eval()
        device = image_features.device
        batch_size = image_features.size(0)

        # Encode image features
        img = self.feature_proj(image_features)
        img = self.pos_encoder(img)
        img = img.permute(1, 0, 2)  # (regions, batch, d_model)

        # Start with <SOS>
        generated = torch.full(
            (batch_size, 1), sos_idx, dtype=torch.long, device=device
        )

        for _ in range(max_len - 1):
            cap = self.word_embedding(generated) * math.sqrt(self.d_model)
            cap = self.pos_decoder(cap)
            cap = cap.permute(1, 0, 2)  # (seq, batch, d_model)

            out = self.transformer(img, cap)
            out = out.permute(1, 0, 2)  # (batch, seq, d_model)

            logits = self.output_layer(out[:, -1])  # last token
            next_token = logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)

            # stop if all finished
            if (next_token == eos_idx).all():
                break

        return generated
