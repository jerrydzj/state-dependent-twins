import re

import numpy as np
import pandas as pd
import pytest

from raw_rows import BASE
from sdt.data.clean import clean
from sdt.twin import readout
from sdt.twin.prompts import RAW, messages
from sdt.twin.readout import MODELS, option_probs, option_tokens, render

SMALL = 'meta-llama/Llama-3.2-1B-Instruct'
CHAT = [{'role': 'system', 'content': 'You are a survey respondent.'}, {'role': 'user', 'content': 'Pick 1, 2 or 3.'}]


def cached_tokenizer(name):
    """The model's tokenizer from the local Hugging Face cache; skip if it was never downloaded."""
    from huggingface_hub import try_to_load_from_cache
    from transformers import AutoTokenizer
    if not all(isinstance(try_to_load_from_cache(name, f, revision=MODELS[name]), str)
               for f in ['tokenizer.json', 'tokenizer_config.json']):
        pytest.skip(f'{name} tokenizer not in the local cache')
    return AutoTokenizer.from_pretrained(name, revision=MODELS[name], local_files_only=True)


@pytest.mark.parametrize('name', MODELS)
def test_render_is_the_chat_template_without_dates(name):
    official = cached_tokenizer(name).apply_chat_template(CHAT, add_generation_prompt=True, tokenize=False)
    assert 'Today Date' in official
    assert render(CHAT) == re.sub(r'Cutting Knowledge Date: [^\n]*\nToday Date: [^\n]*\n\n', '', official)


@pytest.mark.parametrize('name', MODELS)
def test_render_tokenizes_with_one_begin_of_text(name):
    tok = cached_tokenizer(name)
    ids = tok(render(CHAT), add_special_tokens=False).input_ids
    assert ids.count(tok.convert_tokens_to_ids('<|begin_of_text|>')) == 1 and ids[0] == tok.bos_token_id


@pytest.mark.parametrize('name', MODELS)
def test_option_numbers_are_single_distinct_tokens(name):
    ids = option_tokens(cached_tokenizer(name), 3)
    assert len(set(ids)) == 3


@pytest.fixture(scope='module')
def small():
    cached_tokenizer(SMALL)
    try:
        return readout.load(SMALL)
    except OSError:
        pytest.skip(f'{SMALL} weights not in the local cache')


def test_padding_does_not_change_probabilities(small):
    # A short prompt scored alone and in a batch with a longer one, which pads it on the left
    df = clean(pd.DataFrame([BASE, {**BASE, 'AGE': float('nan'), 'SEX': float('nan'), 'MARRY': float('nan')}]))
    chats = messages(df, 'demographics', 'BUS12')
    alone = option_probs(*small, chats.iloc[[1]], 'BUS12')
    batched = option_probs(*small, chats, 'BUS12', batch_size=2)
    assert np.allclose(alone.to_numpy(), batched.iloc[[1]].to_numpy(), atol=1e-3)
    assert batched.index.equals(chats.index) and batched[RAW].sum(axis=1).between(0, 1).all()
