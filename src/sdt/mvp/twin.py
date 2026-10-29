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
"""
import numpy as np
import pandas as pd
import torch
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


def render(chat):
    """The Llama 3 chat format of a list of messages, ending with the header of the assistant's reply."""
    turns = ''.join(f"<|start_header_id|>{m['role']}<|end_header_id|>\n\n{m['content']}<|eot_id|>" for m in chat)
    return f'<|begin_of_text|>{turns}<|start_header_id|>assistant<|end_header_id|>\n\n'


def device():
    if torch.cuda.is_available():
        return 'cuda'
    return 'mps' if torch.backends.mps.is_available() else 'cpu'


def load(name):
    """Tokenizer and model at the pinned revision, with the released bf16 weights converted to fp32."""
    tok = AutoTokenizer.from_pretrained(name, revision=MODELS[name])
    tok.pad_token, tok.padding_side = PAD, 'left'
    model = AutoModelForCausalLM.from_pretrained(name, revision=MODELS[name], dtype=torch.float32)
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


def answer_probs(tok, model, chats, item, mid_share=None, batch_size=16):
    """Class probabilities and answer mass for each chat of `item` (a Series from prompts.messages), indexed like it."""
    p = next_token_probs(tok, model, [render(c) for c in chats], option_tokens(tok, len(OPTIONS[item])), batch_size)
    mass = p.sum(axis=1)
    return to_classes(p / mass[:, None], item, mid_share).assign(mass=mass).set_axis(chats.index)
