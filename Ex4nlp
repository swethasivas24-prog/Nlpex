from collections import Counter
from nltk.util import ngrams

# Sample corpus
text = """
Speech recognition system helps users type faster.
speech recognition models predict the next word.
predictive text systems use language modes.
speech recognition and predictive text are important.
"""

# Tokenization
tokens = text.lower().split()

# N-gram counts
ug = Counter(tokens)
bg = Counter(ngrams(tokens, 2))
tg = Counter(ngrams(tokens, 3))

# Vocabulary size and smoothing parameter
V = len(ug)
k = 1

# Context words
context = ["speech", "recognition"]

# Candidate next words
nextwords = set(tokens)

scores = {}

for word in nextwords:
    # Laplace smoothing
    laplace = (bg[(context[1], word)] + 1) / (ug[context[1]] + V)

    # Add-k smoothing
    addk = (bg[(context[1], word)] + k) / (ug[context[1]] + k * V)

    # Backoff model
    if bg[(context[0], context[1])] > 0 and tg[(context[0], context[1], word)] > 0:
        backoff = tg[(context[0], context[1], word)] / bg[(context[0], context[1])]
    elif bg[(context[1], word)] > 0:
        backoff = bg[(context[1], word)] / ug[context[1]]
    else:
        backoff = ug[word] / sum(ug.values())

    # Interpolation weights
    i1, i2, i3 = 0.5, 0.3, 0.2

    # Trigram probability
    if bg[(context[0], context[1])] > 0:
        tri = tg[(context[0], context[1], word)] / bg[(context[0], context[1])]
    else:
        tri = 0

    # Bigram probability
    if ug[context[1]] > 0:
        bi = bg[(context[1], word)] / ug[context[1]]
    else:
        bi = 0

    # Unigram probability
    uni = ug[word] / sum(ug.values())

    # Interpolated probability
    interp = i1 * tri + i2 * bi + i3 * uni

    # Final score (average of methods)
    scores[word] = (laplace + addk + backoff + interp) / 4

# Top 3 predictions
suggestions = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]

print("Input:", " ".join(context))
print("Next word suggestions:")
for w, p in suggestions:
    print(w)
