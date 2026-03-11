import torch
import torch.nn.functional as F

vocab_size = 100
embed_dim = 32
batch_size = 4
seq_len = 8

weight = torch.randn(vocab_size, embed_dim)
indices = torch.randint(0, vocab_size, (batch_size, seq_len))

out_f = F.embedding(indices, weight)
out_s = weight[indices]

print(f"out_f shape: {out_f.shape}")
print(f"out_s shape: {out_s.shape}")
print(f"Equal: {torch.allclose(out_f, out_s)}")
