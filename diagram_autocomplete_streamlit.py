import streamlit as st
import torch

st.title("Bigram Name Autocomplete")
st.write("Type a name and see the most likely next characters based on the bigram model.")

# Load dataset
words = open('names.txt').read().splitlines()

# Build vocabulary
chars = sorted(list(set(''.join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}

# Build bigram count matrix
N = torch.zeros((27, 27), dtype=torch.int32)
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        N[stoi[ch1], stoi[ch2]] += 1

# Convert counts to probabilities
P = N.float()
P = P / P.sum(dim=1, keepdim=True)

# User input
user_input = st.text_input("Start typing a name:", value="a")

if user_input:
    # Use last character as context
    last_char = user_input[-1].lower()
    if last_char not in stoi:
        st.warning(f"Character '{last_char}' is not in the training vocabulary.")
    else:
        ix = stoi[last_char]
        probs = P[ix]
        top_k = torch.topk(probs, 5)

        st.subheader(f"Top 5 likely next characters after '{last_char}':")
        for i, (prob, idx) in enumerate(zip(top_k.values, top_k.indices)):
            char = itos[idx.item()]
            if char == '.':
                char = "(end of name)"
            st.write(f"{i + 1}. **{char}** — probability {prob.item():.4f}")
