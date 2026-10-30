"""Twin scoring for the MVP: the probability a language model gives each answer to a prompt from prompts.py.

For each prompt:

1. `render` puts the system and user messages into the Llama 3 chat format, ending where the assistant's reply
   begins. Meta's chat template also writes a "Cutting Knowledge Date" and a "Today Date" line into the system message
   (Llama 3.2's template uses the date of the run); `render` leaves both out, so no prompt carries a date.
2. The tokenizer turns the text into token IDs, and the model runs one forward pass over them.
3. At the last position, the model gives a probability to each of its 128,256 tokens as the next one. The probabilities
   of the option numbers ("1", "2", "3", one token each) are read off.
4. Their sum is the answer mass, the probability that the reply starts with a valid option. Dividing each option's
   probability by it gives the option probabilities, which sum to 1.
5. `to_classes` maps options to classes. DUR's web form has no middle option, so the twin's middle probability is the
   baseline shares' middle share (`mid_share`), and its two options share the rest in their own ratio.

Prompts are scored in batches. A batch pads shorter prompts on the left to a common length, and the position of each
token is counted from its prompt's first real token, so padding cannot change a prompt's probabilities.

The weights are released in bf16 and computed with in fp32. Converting bf16 to fp32 is exact, so the weights are the
released ones. bf16 arithmetic keeps about 3 significant digits, and its rounding depends on the shape of the batch:
with the Llama 3.2 1B model, the same prompt scored alone and in a batch differed by up to 0.03 in probability in
bf16, and by 0.00001 in fp32.

The intercept-shifted twin (`fit_shift`, `apply_shift`) moves the log of each class probability by a constant per
class, b = (b_fav, 0, b_unf), and takes the softmax again: softmax(log p + b). The same b applies to every respondent,
so the shift changes the twin's levels but not how its answers differ across respondents and states. b is fitted
once per model, prompt, feature set and item, by weighted maximum likelihood on the training window's web interviews.
"""
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from transformers import AutoModelForCausalLM, AutoTokenizer

from sdt.mvp.clean import CLASSES
from sdt.mvp.prompts import OPTIONS

# Hugging Face Hub revision of each model. Llama 3.1 8B: its weights are unchanged since release on 23 July 2024;
# later commits fixed the tokenizer and chat template. Llama 3.2 1B is for local tests only: released on
# 25 September 2024, it falls after the knowledge horizon.
MODELS = {
    'meta-llama/Llama-3.1-8B-Instruct': '0e9e39f249a16976918f6564b8830bc894c89659',
    'meta-llama/Llama-3.2-1B-Instruct': '9213176726f574b556790deb65791e0c5aa438b6',
}
# Llama 3 sets no padding token. Meta reserved this one for padding in fine-tuning; it never occurs in text, and the
# attention mask hides it from every real token, so which token pads makes no difference to the probabilities.
PAD = '<|finetune_right_pad_id|>'
RAW = ['p1', 'p2', 'p3']           # raw probability of option number 1, 2 and 3


def render(chat):
    """The Llama 3 chat format of a list of messages, ending with the header of the assistant's reply."""
    turns = ''.join(f"<|start_header_id|>{m['role']}<|end_header_id|>\n\n{m['content']}<|eot_id|>" for m in chat)
    return f'<|begin_of_text|>{turns}<|start_header_id|>assistant<|end_header_id|>\n\n'


def device():
    if torch.cuda.is_available():
        return 'cuda'
    return 'mps' if torch.backends.mps.is_available() else 'cpu'


def load(name, dtype=torch.float32):
    """Tokenizer and model at the pinned revision. The released bf16 weights are converted to fp32 unless `dtype` is
    torch.bfloat16, which keeps them as released (used only to measure what fp32 changes)."""
    tok = AutoTokenizer.from_pretrained(name, revision=MODELS[name])
    tok.pad_token, tok.padding_side = PAD, 'left'
    model = AutoModelForCausalLM.from_pretrained(name, revision=MODELS[name], dtype=dtype)
    return tok, model.to(device()).eval()


def option_tokens(tok, n):
    """Token IDs of the option numbers 1 to n. Raise unless each number is a single token."""
    ids = [tok.encode(str(k), add_special_tokens=False) for k in range(1, n + 1)]
    if any(len(i) != 1 for i in ids):
        raise ValueError(f'option numbers are not single tokens: {ids}')
    return [i[0] for i in ids]


@torch.inference_mode()
def next_token_probs(tok, model, texts, token_ids, batch_size=16):
    """For each text, the probability of each of `token_ids` as the next token (softmax over the full vocabulary)."""
    out = []
    for i in range(0, len(texts), batch_size):
        # render() already starts with <|begin_of_text|>, so the tokenizer must not add another
        enc = tok(texts[i:i + batch_size], add_special_tokens=False, padding=True, return_tensors='pt').to(model.device)
        positions = (enc.attention_mask.cumsum(-1) - 1).clamp(min=0)    # 0 at each prompt's first real token
        logits = model(**enc, position_ids=positions, logits_to_keep=1, use_cache=False).logits[:, -1].float()
        out.append(logits.softmax(-1)[:, token_ids].cpu().numpy())
    return np.concatenate(out)


def to_classes(p, item, mid_share=None):
    """Option probabilities (rows summing to 1, columns in the web form's order) -> a DataFrame of CLASSES."""
    options = OPTIONS[item]
    out = pd.DataFrame(p, columns=options)
    if 'mid' not in options:
        if mid_share is None:
            raise ValueError(f'{item} has no middle option: pass mid_share')
        out = (out * (1 - mid_share)).assign(mid=mid_share)
    return out[CLASSES]


def option_probs(tok, model, chats, item, batch_size=16):
    """Run each chat of `item` (a Series from prompts.messages): the raw probability of each option number, before
    dividing by the answer mass, in columns RAW (p3 is NaN for an item with two options). Indexed like `chats`."""
    n = len(OPTIONS[item])
    p = next_token_probs(tok, model, [render(c) for c in chats], option_tokens(tok, n), batch_size)
    return pd.DataFrame(p, columns=RAW[:n], index=chats.index).reindex(columns=RAW)


def class_probs(raw, item, mid_share=None):
    """Raw option probabilities of `item` (columns RAW) -> CLASSES probabilities and the answer mass, indexed like
    `raw`: divide by the mass (steps 4 and 5 of the module docstring)."""
    p = raw[RAW[:len(OPTIONS[item])]].to_numpy(float)
    mass = p.sum(axis=1)
    return to_classes(p / mass[:, None], item, mid_share).assign(mass=mass).set_axis(raw.index)


def _shifted(logp, b):
    """softmax(logp + b) row by row, with b = (b_fav, 0, b_unf)."""
    z = logp + np.array([b[0], 0.0, b[1]])
    z -= z.max(axis=1, keepdims=True)
    q = np.exp(z)
    return q / q.sum(axis=1, keepdims=True)


def fit_shift(p, y, w):
    """The shift b = (b_fav, 0, b_unf) that maximises the weighted log-likelihood of answers `y` (class labels) under
    softmax(log p + b), where `p` holds CLASSES probabilities. Returned as a Series indexed by CLASSES."""
    logp = np.log(np.clip(np.asarray(p[CLASSES], float), 1e-12, None))   # clip: guard against log(0)
    onehot = (np.asarray(y)[:, None] == np.array(CLASSES)).astype(float)
    missing = [c for c, n in zip(CLASSES, onehot.sum(axis=0)) if n == 0 and c != 'mid']
    if missing:     # the likelihood then keeps rising as that class's b falls, so no finite b maximises it
        raise ValueError(f'no answers in {missing}: the shift has no maximum')
    w = np.asarray(w, float) / np.sum(w)

    def loss(b):
        q = _shifted(logp, b)
        nll = -(w * np.log((q * onehot).sum(axis=1))).sum()
        grad = (w[:, None] * (q - onehot)).sum(axis=0)[[0, 2]]          # d nll / d (b_fav, b_unf)
        return nll, grad

    fit = minimize(loss, np.zeros(2), jac=True, method='L-BFGS-B', options={'gtol': 1e-8})
    if not fit.success:
        raise RuntimeError(f'intercept shift did not converge: {fit.message}')
    return pd.Series([fit.x[0], 0.0, fit.x[1]], index=CLASSES)


def apply_shift(p, b):
    """softmax(log p + b) for a DataFrame `p` of CLASSES probabilities and a shift `b` from fit_shift."""
    logp = np.log(np.clip(np.asarray(p[CLASSES], float), 1e-12, None))
    return pd.DataFrame(_shifted(logp, b[['fav', 'unf']].to_numpy()), columns=CLASSES, index=p.index)
