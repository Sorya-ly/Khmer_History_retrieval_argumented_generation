# -*- coding: utf-8 -*-
"""
TINY fine-tune DEMO (NumPy only, no torch / no downloads).

It mimics MultipleNegativesRankingLoss from the real notebook:
  - each question should sit CLOSE to its correct chunk
  - and FAR from every other chunk (the in-batch negatives)
We start from random vectors and nudge them with gradient descent,
then show the question->correct-chunk similarity rising.
"""
import numpy as np
np.random.seed(42)

# 4 toy (question, correct-chunk) pairs. Pretend these are Khmer Q/passage pairs.
pairs = [
    ("when did the Khmer Rouge take power",      "Khmer Rouge ruled Cambodia 1975 to 1979"),
    ("who built Angkor Wat",                     "Suryavarman II built Angkor Wat"),
    ("what religion in the Angkor period",       "Angkor kings followed Hindu and Buddhist faiths"),
    ("when was Longvek the capital",             "Longvek served as the royal capital in the 16th century"),
]
Q = [q for q, _ in pairs]
P = [p for _, p in pairs]
N = len(pairs)
DIM = 8  # tiny embedding size (real e5-small = 384)

# random starting vectors (this stands in for the pretrained model's output)
q_vec = np.random.randn(N, DIM)
p_vec = np.random.randn(N, DIM)

def unit(x):                       # normalize to length 1 -> dot product = cosine
    return x / np.linalg.norm(x, axis=-1, keepdims=True)

def sim_matrix(a, b):              # cosine similarity of every q vs every p
    return unit(a) @ unit(b).T

def report(tag):
    S = sim_matrix(q_vec, p_vec)
    diag = np.diag(S).mean()       # avg similarity to the CORRECT chunk
    off  = (S.sum() - np.trace(S)) / (N * N - N)  # avg similarity to WRONG chunks
    acc  = (S.argmax(1) == np.arange(N)).mean()   # does top match = correct chunk?
    print(f"{tag:18} correct-sim={diag:+.3f}  wrong-sim={off:+.3f}  retrieval-acc={acc:.0%}")

print("Toy pairs:")
for i, (q, p) in enumerate(pairs):
    print(f"  Q{i}: {q!r}  ->  P{i}: {p!r}")
print()

report("BEFORE training")

# ---- training loop: this IS MultipleNegativesRankingLoss, by hand ----
# For each question, softmax over all chunks; label = its own index.
# Loss = cross-entropy. We push the correct pair up, the negatives down.
lr = 0.5
for step in range(300):
    qn, pn = unit(q_vec), unit(p_vec)
    logits = qn @ pn.T                      # NxN similarity scores
    # softmax rows
    e = np.exp(logits - logits.max(1, keepdims=True))
    probs = e / e.sum(1, keepdims=True)
    # cross-entropy gradient on logits: prob - onehot(correct)
    grad = probs.copy()
    grad[np.arange(N), np.arange(N)] -= 1
    grad /= N
    # backprop through the dot product (approx, on normalized vecs)
    q_vec -= lr * (grad @ pn)
    p_vec -= lr * (grad.T @ qn)

report("AFTER training")

print("\nInterpretation:")
print("  correct-sim went UP  -> question now sits close to its right chunk")
print("  wrong-sim went DOWN  -> question sits far from the other chunks")
print("  retrieval-acc 100%   -> searching by similarity now returns the right chunk")
print("\nThis is exactly what the notebook's model.fit(...) does, but on")
print("real 384-dim Khmer embeddings instead of these 8-dim toy vectors.")
