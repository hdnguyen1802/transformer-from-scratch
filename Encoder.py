import torch
import torch.nn as nn

from SelfAttention import SelfAttention

class EncoderBlock(nn.Module):
    def __init__(self, embed_size,heads,dropout_probability,forward_expansion):
        
        super(EncoderBlock,self).__init__()
        
        self.attention = SelfAttention(embed_size,heads) # self attention layer
        
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        
        self.dropout=nn.Dropout(dropout_probability)
        # dropout_probability: probability of dropping out a neuron during training to prevent overfitting
    
    def forward(self, value, key, query, mask):
        
        attention = self.attention(value, key, query, mask)
         # residual connection: add the original query to the attention output
        # This helps the model learn better and prevents vanishing gradients.
        x = self.dropout(self.norm1(attention + query))
        
        forward = self.feed_forward(x)
        
        output = self.dropout(self.norm2(forward + x))
        
        return output
class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        position_max_length,
        embed_size,
        heads,
        dropout_probability,
        forward_expansion,
        num_layers,
        device
    ):
    # src_vocab_size: size of the source vocabulary (number of unique tokens in the input sequence)
    # position_max_length: maximum length of the input sequence (number of tokens)
    # embed_size: dimensionality (length) of the embedding vectors
    # heads: number of attention heads
    # dropout_probability: probability of dropping out a neuron during training to prevent overfitting
    # forward_expansion: expansion factor for the feed-forward network (number of neurons in the hidden layer)
    # num_layers: number of Encoder blocks in the encoder
    # device: device to run the model on (CPU or GPU)
    
        super(Encoder,self).__init__()
        
        self.embed_size = embed_size
        self.device = device
        self.dropout = nn.Dropout(dropout_probability)
        
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(position_max_length, embed_size)
        
        # word_embedding: converts input tokens to embedding vectors
        # position_embedding: adds positional information to the embedding vectors
        
        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    embed_size,
                    heads,
                    dropout_probability=dropout_probability,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )
    
    def forward(self, input,mask):
        
        batch_size, sequence_length = input.shape
        
        position = torch.arange(0, sequence_length).expand(batch_size, sequence_length).to(self.device)
        # position: tensor of shape (batch_size, sequence_length) containing the positional indices for each token in the input sequence
        # For example, if batch_size=2 and sequence_length=4, position will be [[0, 1, 2, 3], [0, 1, 2, 3]]
        
        output = self.dropout(
            self.word_embedding(input) + self.position_embedding(position)
        )
        
        for layer in self.layers:
            output = layer(output, output, output, mask)
            # This is a unique for the Encoder-only architecture, where the input sequence (x) is used for Q,K,V.
            
        return output