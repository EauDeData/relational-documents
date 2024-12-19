from torch import nn

from networks import resnet
from utils_jeta import constants

from torchvision.transforms import functional as F
from PIL import Image
import torch
from utils_jeta.constants import FEATURE_DIM

class TransformerSentenceEncoder(nn.Module):
    def __init__(self, depth, width, feature_dim, dictionary_length):
        """
        Initialize the TransformerSentenceEncoder.
        :param depth: Number of transformer encoder layers.
        :param width: Number of attention heads.
        :param feature_dim: Dimensionality of the embeddings.
        :param dictionary_length: Size of the vocabulary for nn.Embedding.
        """
        super().__init__()

        # Word embedding layer
        self.embedding = nn.Embedding(dictionary_length, feature_dim)

        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(1, 1000, feature_dim))  # Supports max 1000 tokens

        # Transformer encoder layers
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=feature_dim, nhead=width, batch_first=True),
            num_layers=depth
        )

        # CLS token representation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, feature_dim))

    def forward(self, input_ids):
        """
        Forward pass through the TransformerSentenceEncoder.
        :param input_ids: Tensor of shape (batch_size, seq_length) containing token indices.
        :return: Tensor of shape (batch_size, feature_dim) representing the sentence embedding.
        """
        batch_size, seq_length = input_ids.shape

        # Embed input tokens
        embedded_tokens = self.embedding(input_ids)  # Shape: (batch_size, seq_length, feature_dim)

        # Create positional encodings
        positional_encodings = self.positional_encoding[:, :seq_length, :]  # Shape: (1, seq_length, feature_dim)

        # Add positional encodings to embeddings
        embeddings = embedded_tokens + positional_encodings  # Shape: (batch_size, seq_length, feature_dim)

        # Add the CLS token to the beginning of the sequence
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # Shape: (batch_size, 1, feature_dim)
        embeddings_with_cls = torch.cat([cls_tokens, embeddings], dim=1)  # Shape: (batch_size, seq_length + 1, feature_dim)

        # Pass through the Transformer encoder
        transformer_output = self.transformer(embeddings_with_cls)  # Shape: (batch_size, seq_length + 1, feature_dim)

        # Extract the CLS token representation (first token)
        cls_embedding = transformer_output[:, 0, :]  # Shape: (batch_size, feature_dim)

        return cls_embedding

class DisjointEncoder(nn.Module):
    def __init__(self, vocab_size, device):
        super(DisjointEncoder, self).__init__()


        # TODO: També inicialitzar encoder textual i encoder de les paraules
        self.extractor = resnet.resnet50(pretrained=True, pth_path=constants.PRETRAINED_EXTRACTOR_PATH)
        self.to_low_dim = torch.nn.Sequential(
            nn.ReLU(),
            nn.Linear(2048, FEATURE_DIM)
        )

        self.local_extractor = resnet.LatentWordEncoder(final_dim=FEATURE_DIM)

        self.query_extractor = TransformerSentenceEncoder(6, 4, FEATURE_DIM, vocab_size)
        self.DEVICE = device

    def forward(self, global_view, local_views, query):

        local_embeds = self.local_extractor(local_views)

        global_fm, global_embed_pre, _ = self.extractor(global_view.detach())
        global_embed = self.to_low_dim(global_embed_pre)

        query_embeddings = self.query_extractor(query)

        return global_embed, local_embeds, query_embeddings
