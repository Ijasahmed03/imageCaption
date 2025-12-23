import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# -------------------------------
# Geometry computation
# -------------------------------
def compute_geometry_bias(boxes):
    """
    boxes: (B, N, 4) normalized [x1, y1, x2, y2]
    returns: (B, N, N, 4)
    """
    B, N, _ = boxes.size()

    x1, y1, x2, y2 = boxes.unbind(-1)
    w = x2 - x1
    h = y2 - y1
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    dx = cx.unsqueeze(2) - cx.unsqueeze(1)
    dy = cy.unsqueeze(2) - cy.unsqueeze(1)

    dx = dx / (w.unsqueeze(2) + 1e-6)
    dy = dy / (h.unsqueeze(2) + 1e-6)

    dw = torch.log((w.unsqueeze(2) + 1e-6) / (w.unsqueeze(1) + 1e-6))
    dh = torch.log((h.unsqueeze(2) + 1e-6) / (h.unsqueeze(1) + 1e-6))

    geom = torch.stack([dx, dy, dw, dh], dim=-1)
    return geom


# -------------------------------
# Relation bias module
# -------------------------------
class RelationAwareBias(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.linear = nn.Linear(4, hidden_dim)

    def forward(self, geom):
        """
        geom: (B, N, N, 4)
        returns: (B, N, N)
        """
        bias = self.linear(geom)
        return bias.mean(-1)


# -------------------------------
# Captioning Model
# -------------------------------
class TransformerCaptionModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        feature_dim=2048,
        hidden_dim=512,
        num_heads=8,
        num_layers=3,
        max_len=20,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.max_len = max_len

        # Feature projection
        self.feature_proj = nn.Linear(feature_dim, hidden_dim)

        # Geometry bias
        self.relation_bias = RelationAwareBias(hidden_dim)

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Decoder
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
        )

        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    # ---------------------------
    # Forward (training)
    # ---------------------------
    def forward(self, features, boxes, captions):
        """
        features: (B, N, 2048)
        boxes:    (B, N, 4)
        captions: (B, T)
        """

        # Project features
        feats = self.feature_proj(features)

        # Geometry bias
        geom = compute_geometry_bias(boxes)
        geom_bias = self.relation_bias(geom)
        geom_bias = geom_bias / math.sqrt(self.hidden_dim)
        # Inject relation bias
        feats = feats + 0.1*geom_bias.mean(dim=2).unsqueeze(-1)

        # Encode
        memory = self.encoder(feats)

        # Decode
        tgt = self.embedding(captions)
        out = self.decoder(tgt, memory)

        return self.fc_out(out)

    # ---------------------------
    # Inference
    # ---------------------------
    def generate(self, features, boxes, sos_idx, eos_idx, min_len=5):
        B = features.size(0)
        device = features.device

        feats = self.feature_proj(features)

        geom = compute_geometry_bias(boxes)
        geom_bias = self.relation_bias(geom)
        geom_bias = geom_bias / math.sqrt(self.hidden_dim)
        feats = feats + 0.1 * geom_bias.mean(dim=2).unsqueeze(-1)

        memory = self.encoder(feats)

        outputs = torch.full((B, 1), sos_idx, dtype=torch.long, device=device)

        for step in range(self.max_len):
            tgt = self.embedding(outputs)
            out = self.decoder(tgt, memory)
            logits = self.fc_out(out[:, -1])

            # 🚫 block PAD always
            pad_idx = 0
            logits[:, pad_idx] = -1e9
            logits[:, sos_idx] = -1e9
            # 🚫 block EOS for first few steps
            if step < min_len:
                logits[:, eos_idx] = -1e9

            next_token = logits.argmax(-1, keepdim=True)
            outputs = torch.cat([outputs, next_token], dim=1)

            if step >= min_len and (next_token == eos_idx).all():
                break

        return outputs
