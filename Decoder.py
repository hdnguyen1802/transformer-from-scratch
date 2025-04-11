import torch
import torch.nn as nn
from SelfAttention import SelfAttention 
class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout_probability, forward_expansion):
        super(DecoderBlock, self).__init__()

        # 1. Masked Self-Attention
        self.masked_self_attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)

        # 2. Cross-Attention (Encoder-Decoder Attention)
        self.cross_attention = SelfAttention(embed_size, heads)
        self.norm2 = nn.LayerNorm(embed_size)

        # 3. Feed Forward Network
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.norm3 = nn.LayerNorm(embed_size)

        self.dropout = nn.Dropout(dropout_probability)

    def forward(self, x, encoder_output, src_mask, trg_mask):
        # x: target sequence embeddings (from previous layer or input embedding)
        # encoder_output: output from the encoder stack
        # src_mask: padding mask for the source sequence (used in cross-attention)
        # trg_mask: combined look-ahead and padding mask for the target sequence

        # --- Stage 1: Masked Self-Attention ---
        masked_attn_output = self.masked_self_attention(x, x, x, trg_mask)
        # Add & Norm 1 (Residual connection uses the input 'x')
        x_intermediate1 = self.dropout(self.norm1(masked_attn_output + x))

        # --- Stage 2: Cross-Attention ---
        # Query comes from the output of the previous stage (x_intermediate1)
        # Key and Value come from the encoder_output
        cross_attn_output = self.cross_attention(encoder_output, encoder_output, x_intermediate1, src_mask) # Note Q=x_intermediate1, K=V=encoder_output
        # Add & Norm 2 (Residual connection uses the input to this stage 'x_intermediate1')
        x_intermediate2 = self.dropout(self.norm2(cross_attn_output + x_intermediate1))

        # --- Stage 3: Feed Forward ---
        ff_output = self.feed_forward(x_intermediate2)
        # Add & Norm 3 (Residual connection uses the input to this stage 'x_intermediate2')
        output = self.dropout(self.norm3(ff_output + x_intermediate2))

        return output

class Decoder(nn.Module):
    def __init__(
        self,
        trg_vocab_size,
        position_max_length,
        embed_size,
        heads,
        dropout_probability,
        forward_expansion,
        num_layers,
        device
        ):
        super(Decoder,self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.dropout = nn.Dropout(dropout_probability)

        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(position_max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                DecoderBlock( # Use the corrected DecoderBlock now
                    embed_size,
                    heads,
                    dropout_probability=dropout_probability,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )

        self.fc = nn.Linear(embed_size, trg_vocab_size)

    def forward(self,x,encoder_output,src_mask,trg_mask):

        batch_size, sequence_length = x.shape
        position = torch.arange(0, sequence_length).expand(batch_size, sequence_length).to(self.device)

        output = self.dropout(
            (self.word_embedding(x) + self.position_embedding(position))
        )

        for layer in self.layers:
            output = layer(output, encoder_output, src_mask, trg_mask) # Pass arguments correctly

        output = self.fc(output)
        return output