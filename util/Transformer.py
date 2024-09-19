import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, N=6, d_model=512, d_ff=2048, head=8, dropout=0.1):
        super(Transformer, self).__init__()

        # Input
        position = PositionalEncoding(d_model, dropout)
        src_embed = nn.Sequential(Embeddings(src_vocab_size, d_model), position)
        tgt_embed = nn.Sequential(Embeddings(tgt_vocab_size, d_model), position)
        # Process
        atten = MultiHeadAttention(d_model, head, dropout)
        ff = FeedForward(d_model, d_ff, dropout)
        encoder = Encoder(EncoderLayer(d_model, atten, ff, dropout), N)
        decoder = Decoder(DecoderLayer(d_model, atten, atten, ff, dropout), N)
        # Output
        generator = Generator(d_model, tgt_vocab_size)
        
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.d_model = d_model
        
        # Model
        self.model = EncoderDecoder(src_embed, tgt_embed, encoder, decoder, generator)
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        最终输出目标序列概率分布
        (batch_size, tgt_seq_len, tgt_vocab_size)
        (每次处理句子数量，句子长度，词汇表大小即每个词概率分布)
        Args:
            src (batch_size, seq_len): 编码器文本数字索引输入
            tgt (batch_size, seq_len): 解码器文本数字索引输入
            src_mask (batch_size, 1, seq_len): 屏蔽padding
            tgt_mask (batch_size, seq_len, seq_len): 屏蔽未生成的注意力分数 QK
        """
        return self.model(src, tgt, src_mask, tgt_mask)

# EncoderDecoder
class EncoderDecoder(nn.Module):
    def __init__(self, src_embed, tgt_embed, encoder, decoder, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
    
    # Ugly but Coding-friendly
    def input_embedding(self, src):
        """
        Source Embedding = Input Embedding
        """
        return self.src_embed(src)
        
    def output_embedding(self, tgt):
        """
        Target Embedding = Output Embedding
        """
        return self.tgt_embed(tgt)
    
    def encode(self, src, src_mask):
        return self.encoder(self.input_embedding(src), src_mask)
    
    def decode(self, tgt, tgt_mask, memory, src_mask):
        """
        self.encode(src, src_mask): memory from Encoder, which means K & V
        """
        return self.decoder(self.output_embedding(tgt), tgt_mask, memory, src_mask)
        
    def generate(self, src, src_mask, tgt, tgt_mask):
        return self.generator(self.decode(tgt, tgt_mask, self.encode(src, src_mask), src_mask))
        
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        EncoderDecoder的forward是由五个模块的forward头尾拼接的
        """
        return self.generate(src, src_mask, tgt, tgt_mask)
    
## Encoder
class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.dim_size)
        
    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)
    
class EncoderLayer(nn.Module):
    def __init__(self, dim_size, atten, ff, dropout):
        super(EncoderLayer, self).__init__()
        self.dim_size = dim_size
        self.atten = atten
        self.ff = ff
        self.sublayerconnection = clones(SublayerConnection(dim_size, dropout), 2)
        
    def forward(self, x, src_mask):
        x = self.sublayerconnection[0](x, lambda x: self.atten(x, x, x, src_mask))
        x = self.sublayerconnection[1](x, self.ff)
        return x

## Decoder
class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.dim_size)
        
    def forward(self, x, tgt_mask, memory, src_mask):
        for layer in self.layers:
            x = layer(x, tgt_mask, memory, src_mask)
        return self.norm(x)
      
class DecoderLayer(nn.Module):
    def __init__(self, dim_size, atten, msk_atten, ff, dropout):
        super(DecoderLayer, self).__init__()
        self.dim_size = dim_size
        self.atten = atten
        self.msk_atten = msk_atten
        self.ff = ff
        self.sublayerconnection = clones(SublayerConnection(dim_size, dropout), 3)
        
    def forward(self, x, tgt_mask, memory, src_mask):
        m = memory
        x = self.sublayerconnection[0](x, lambda x: self.msk_atten(x, x, x, tgt_mask))
        x = self.sublayerconnection[1](x, lambda x: self.atten(x, m, m, src_mask))
        x = self.sublayerconnection[2](x, self.ff)
        return x

## Generator
class Generator(nn.Module):
    def __init__(self, d_model, tgt_vocab_size):
        super(Generator, self).__init__()
        self.linear = nn.Linear(d_model, tgt_vocab_size)
        
    def forward(self, x):
        return F.log_softmax(self.linear(x), dim=-1)

### MultiHeadAttention & FeedForward & SublayerConnection
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, head, dropout):
        super(MultiHeadAttention, self).__init__()
        assert(d_model % head == 0)
        self.d_k = d_model // head
        self.head = head
        self.dropout = nn.Dropout(dropout)
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        
    def forward(self, query, key, value, mask):
        if mask is not None:
            mask = mask.unsqueeze(1)
        
        batch_size = query.size(0)
        
        Q, K, V = [L(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
                             for L, x in zip(self.linears, (query, key, value))]
        
        x, _ = attention(Q, K, V, mask, self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)
        
        return self.linears[-1](x)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.norm = LayerNorm(d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)  
            
    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.norm(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x

class SublayerConnection(nn.Module):
    def __init__(self, dim_size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(dim_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))
        # 原始论文架构
        # return x + self.dropout(sublayer(self.norm(x)))

### clones
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

### attention
def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

### Normlization
class LayerNorm(nn.Module):
    def __init__(self, dim_size, eps=1e-9):
        super(LayerNorm, self).__init__()
        
        self.a = nn.Parameter(torch.ones(dim_size))
        self.b = nn.Parameter(torch.zeros(dim_size))
        self.eps = eps
        
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a * (x - mean) / (std + self.eps) + self.b

## Input/Output Embedding
class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(Embeddings, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        
    def forward(self, x):
        """
        x = src / tgt sequence
        """
        x = self.embedding(x) * math.sqrt(self.d_model)
        return x
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
         
        pe = torch.zeros(max_len,d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
                
    def forward(self, x):
        pe = self.pe[:, :x.size(1)].to(x.device)
        x = x + pe
        return self.dropout(x)