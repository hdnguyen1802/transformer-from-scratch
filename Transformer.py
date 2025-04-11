import torch
import torch.nn as nn

from Encoder import Encoder
from Decoder import Decoder

class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embed_size=512,
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout_probability=0,
        device="cpu",
        position_max_length=100,
    ):

        super(Transformer, self).__init__()

        self.encoder = Encoder(
                src_vocab_size,
                position_max_length,
                embed_size,
                heads,
                dropout_probability,
                forward_expansion,
                num_layers,
                device,
        )

        self.decoder = Decoder(
                trg_vocab_size,
                position_max_length,
                embed_size,
                heads,
                dropout_probability,
                forward_expansion,
                num_layers,
                device,
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        # src: Input tensor containing source sequence indices. Shape: (N, src_len)
        # self.src_pad_idx: The index used for padding tokens in the source vocabulary.

        # Create a boolean mask by checking which elements in src are NOT the padding index.
        # True indicates a real token, False indicates a padding token.
        # Shape after comparison: (N, src_len)
        src_mask = (src != self.src_pad_idx)

        # Add two dimensions to reshape the mask for broadcasting in multi-head attention.
        # Unsqueeze(1) adds a dimension for attention heads -> (N, 1, src_len)
        # Unsqueeze(2) adds a dimension for query positions -> (N, 1, 1, src_len)
        # This shape allows masking across all heads and query positions based on key padding.
        src_mask = src_mask.unsqueeze(1).unsqueeze(2)

        # Final intended shape: (N, 1, 1, src_len)
        # Where N = batch size, src_len = source sequence length.

        # Move the mask tensor to the specified device (e.g., 'cuda' or 'cpu').
        return src_mask.to(self.device)
    def make_trg_mask(self, trg):
        # trg: Input tensor containing target sequence indices. Shape: (N, trg_len)
        # self.trg_pad_idx: The index used for padding tokens in the target vocabulary.

        # Get batch size (N) and target sequence length (trg_len) from the input shape.
        N, trg_len = trg.shape

        # --- Create Target Padding Mask ---
        # Create a boolean mask checking which elements in trg are NOT the padding index.
        # True indicates a real token, False indicates padding.
        # Reshape to (N, 1, 1, trg_len) for broadcasting against attention scores.
        # This masks KEYS (columns in attention matrix) based on padding.
        # NOTE: Variable name 'trg_sub_mask' is potentially confusing, 'trg_pad_mask' would be clearer.
        trg_sub_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        # Shape: (N, 1, 1, trg_len)

        # --- Create Look-ahead Mask ---
        # Create a square matrix of ones with dimensions (trg_len, trg_len).
        # Apply torch.tril to keep only the lower triangular part (including diagonal).
        # This creates the look-ahead mask where True means attention is allowed (pos_key <= pos_query).
        # Shape: (trg_len, trg_len)
        look_ahead_mask_2d = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool() # Use .bool()

        # Expand the 2D look-ahead mask to 4D for broadcasting with attention scores.
        # Shape becomes (N, 1, trg_len, trg_len) by replicating the pattern.
        # NOTE: Variable name 'trg_mask_dig' is potentially confusing, 'look_ahead_mask' would be clearer.
        trg_mask_dig = look_ahead_mask_2d.expand(
            N, 1, trg_len, trg_len
        )
        # Shape: (N, 1, trg_len, trg_len)

        # --- Combine Masks ---
        # Combine the padding mask and the look-ahead mask using logical AND (&).
        # Broadcasting rules apply:
        #   trg_mask_dig: (N, 1, trg_len, trg_len)
        #   trg_sub_mask: (N, 1, 1,       trg_len) -> Broadcasts along dimension 2 (query dimension)
        # The final mask is True only if BOTH conditions are met:
        #   1. The key position is NOT padding (checked by trg_sub_mask).
        #   2. The key position is <= the query position (checked by trg_mask_dig).
        trg_mask = trg_mask_dig & trg_sub_mask
        # Final Shape: (N, 1, trg_len, trg_len)

        # Move the combined mask tensor to the specified device.
        # Note: .to(self.device) might be redundant if intermediate tensors were already on the device.
        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out