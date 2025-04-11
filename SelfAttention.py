import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self,embed_size,heads):
        
    # embed_size: specifies the dimensionality (length) of these embedding vectors
    # For example,embed_size=4, and the input sequence is ' Love AI', Token 1: "Love" -> Embedding Vector (x₁): [0.1, -0.5, 0.8, 0.2]
    
    # heads: number of attention heads 
    
        super(SelfAttention,self).__init__() #inheriting from nn.Module
        
        self.embed_size = embed_size
        self.heads = heads
        
        self.head_dim = embed_size // heads
        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"
        # head_dim: dimensionality of each attention head
        # For example, if embed_size=4 and heads=2, then head_dim=2.
        # Token 1: "Love" -> Embedding Vector (x1): [a_1, b_1, c_1, d_1]
        # Token 1: "AI" -> Embedding Vector (x2): [a_2, b_2, c_2, d_2]
        # input sequence: i=[x₁, x₂] has shape (2,4) (2 tokens, embed_side) 
        # Q=i*W_q_initial (2,embed_size)
        # K=i*W_k_initial (2,embed_size)
        # V=i*W_v_initial (2,embed_size)
        # then we reshape Q,K,V to (2,2,2) (2 tokens, 2 heads, 2 head_dim) => 2 heads of 2 dimensions each. 
        
        self.queries = nn.Linear(embed_size, embed_size, bias=False) # learned weight matrix for Q
        self.keys = nn.Linear(embed_size, embed_size, bias=False) # learned weight matrix for K
        self.values = nn.Linear(embed_size, embed_size, bias=False) # learned weight matrix for V
        
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size) # learned weight matrix for the output of the attention layer
        # heads * head_dim = embed_size : Concatenation of all the attention heads (2*2=4)
        
    def forward(self, value,key,query,mask):
        # mask here is a matrix of 0s and 1s, where 1s indicate the positions of the padding tokens in the input sequence.
        # The mask is used to prevent the model from attending to these padding tokens during the attention calculation.
        # This approach provides flexiblity in choosing masking strategies, such as padding masking or look-ahead masking.
        
        batch_size = query.shape[0]
        
        query_len, key_len, value_len = query.shape[1], key.shape[1], value.shape[1]
        
        # In in the Encoder-only, or Decoder-only architectures, the input sequence (len) is the same for Q, K, and V.
        # In the Encoder-decoder (cross-attention) architecture, the input sequence (len) is different for Q vs K,V.
        # + query_len is from the length of the target sequence (decoder input)
        # + key_len and value_len are from the length of the source sequence (encoder output)
        
        # Now we apply the learned weight matrices to the input sequence (x) to get Q,K,V:
        query = self.queries(query) # (batch_size, query_len, embed_size)
        key = self.keys(key) # (batch_size, key_len, embed_size)
        value = self.values(value) # (batch_size, value_len, embed_size)
        
        # We can reshape the Q,K,V matrices to separate the heads:
        query = query.reshape(batch_size,query_len,self.heads,self.head_dim)
        key = key.reshape(batch_size,key_len,self.heads,self.head_dim)
        value = value.reshape(batch_size,value_len,self.heads,self.head_dim)
        
        # Calculate the energy scores (attention scores) using the dot product of Q and K
        # Method 1: using torch.matmul (@)
        # query shape (batch_size,query_len,heads,head_dim) -> (batch_size,heads,query_len,head_dim)
        # query = query.permute(0,2,1,3)
        # key shape Tranpose (batch_size,key_len,heads,head_dim) -> (batch_size,heads,head_dim,key_len)
        # key = key.permute(0,2,3,1)
        # energy = torch.matmul(query,key) # (batch_size,heads,query_len,key_len)
        
        # Method 2: using einsum (Einstein summation convention)
        
        energy = torch.einsum("bqhd,bkhd->bhqk", [query, key]) # (batch_size,heads,query_len,key_len)
        
        # Mask for padding tokens (if any) in Decoder input (target sequence)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        
        attention = torch.softmax(energy / (self.head_dim ** (1 / 2)), dim=3) 
        # (batch_size,heads,query_len,key_len) dim=3 => softmax across the key_len dimension (columns) for each head
        
        # Calculate attention_score output
        
        output = torch.einsum("bhql,blhd->bqhd", [attention, value]) # (batch_size,query_len,heads,head_dim)
        output = output.reshape(batch_size, query_len, self.heads * self.head_dim) # (batch_size,query_len,embed_size)
        
        out =self.fc_out(output) # (batch_size,query_len,embed_size)
        
        return out 