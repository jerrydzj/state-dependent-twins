import re

import numpy as np
import pandas as pd
import pytest

from sdt.mvp import twin
from sdt.mvp.clean import CLASSES, clean
from sdt.mvp.prompts import messages
from sdt.mvp.twin import MODELS, answer_probs, option_tokens, render, to_classes
from test_clean import BASE

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


def test_to_classes_maps_options_in_form_order():
    out = to_classes(np.array([[0.5, 0.3, 0.2]]), 'BUS12')
    assert out.columns.tolist() == CLASSES and out.iloc[0].tolist() == [0.5, 0.3, 0.2]


def test_to_classes_gives_dur_the_baseline_middle_share():
    out = to_classes(np.array([[0.6, 0.4]]), 'DUR', mid_share=0.01)
    assert np.allclose(out.iloc[0].tolist(), [0.594, 0.01, 0.396]) and np.isclose(out.sum(axis=1)[0], 1)
    with pytest.raises(ValueError):
        to_classes(np.array([[0.6, 0.4]]), 'DUR')


@pytest.fixture(scope='module')
def small():
    cached_tokenizer(SMALL)
    try:
        return twin.load(SMALL)
    except OSError:
        pytest.skip(f'{SMALL} weights not in the local cache')


def test_padding_does_not_change_probabilities(small):
    # A short prompt scored alone and in a batch with a longer one, which pads it on the left
    df = clean(pd.DataFrame([BASE, {**BASE, 'AGE': float('nan'), 'SEX': float('nan'), 'MARRY': float('nan')}]))
    chats = messages(df, 'demographics', 'BUS12')
    alone = answer_probs(*small, chats.iloc[[1]], 'BUS12')
    batched = answer_probs(*small, chats, 'BUS12', batch_size=2)
    assert np.allclose(alone.to_numpy(), batched.iloc[[1]].to_numpy(), atol=1e-3)
    assert np.allclose(batched[CLASSES].sum(axis=1), 1) and batched['mass'].between(0, 1).all()
