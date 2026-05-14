from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.config import CFG

SUPERCLASSES = CFG['data']['superclasses']

SYSTEM_PROMPT = (
    'You are a clinical AI assistant helping cardiologists interpret ECG predictions '
    'from a deep learning model. Be concise (3-5 sentences), clinically grounded, '
    'and always remind the reader that AI results require clinical correlation. '
    'Do not diagnose -- interpret what the model found.'
)


def safe_print(text: str) -> None:
    """Print text safely on Windows cp1252 terminals by replacing non-ASCII."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('ascii', errors='replace').decode('ascii'))


def load_qwen3(
    model_id: str = 'Qwen/Qwen3-0.6B',
    device_map: str = 'auto',
):
    """
    Load Qwen3 tokenizer and causal LM.

    Call AFTER setting os.environ['TRANSFORMERS_OFFLINE'] = '0'.
    First run downloads the model (~400 MB for 0.6B); subsequent runs
    use the local HuggingFace cache.

    Args:
        model_id:   HuggingFace model ID (swap to Qwen3-1.7B for richer output)
        device_map: passed to from_pretrained -- 'auto' uses GPU if available

    Returns:
        (model, tokenizer) tuple, model in eval mode
    """
    print(f'Loading {model_id} ...')
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    model.eval()
    dev = next(model.parameters()).device
    print(f'Qwen3 loaded  |  device: {dev}')
    return model, tokenizer


def build_ecg_prompt(
    result_dict:  dict,
    saliency,
    lead_names:   list[str],
    superclasses: list[str] | None = None,
    true_classes: list[str] | None = None,
) -> str:
    """
    Format a structured ECG inference result into a Qwen3 user prompt.

    Args:
        result_dict:  output of ECGInferencePipeline.predict()
        saliency:     (12, 1000) numpy array or None
        lead_names:   list of 12 lead name strings
        superclasses: class name list (defaults to CFG value)
        true_classes: ground-truth class list (optional, included for context)

    Returns:
        Formatted prompt string ready for apply_chat_template.
    """
    import numpy as np

    if superclasses is None:
        superclasses = SUPERCLASSES

    probs    = result_dict['class_probabilities']
    prob_str = '  '.join(f'{c}:{probs[c]:.2f}' for c in superclasses)

    if saliency is not None:
        scores    = saliency.mean(axis=1)
        top_leads = ', '.join(
            [lead_names[i] for i in np.argsort(scores)[::-1][:3]]
        )
    else:
        top_leads = 'not computed'

    true_str = ', '.join(true_classes) if true_classes else 'unknown'

    return (
        f'ECG ANALYSIS RESULT\n'
        f'True diagnosis (if known): {true_str}\n'
        f'Predicted class:           {", ".join(result_dict["predicted_classes"])}\n'
        f'Confidence score:          {result_dict["confidence_score"]:.2f}\n'
        f'Class probabilities:       {prob_str}\n'
        f'Signal uncertainty:        {result_dict["uncertainty"]:.4f}'
        f' ({result_dict["uncertainty_level"]})\n'
        f'Top salient leads:         {top_leads}\n\n'
        f'Please provide a 3-5 sentence clinical interpretation of these results.'
    )


def generate_explanation(
    prompt_text:    str,
    qwen_model,
    qwen_tokenizer,
    system_prompt:  str  = SYSTEM_PROMPT,
    max_new_tokens: int  = 220,
    temperature:    float = 0.3,
) -> str:
    """
    Call Qwen3 to generate a clinical narrative from a structured prompt.

    Uses non-thinking mode (enable_thinking=False) for concise output.
    Strips any leaked <think>...</think> blocks automatically.

    Args:
        prompt_text:    user prompt from build_ecg_prompt()
        qwen_model:     loaded Qwen3 model (from load_qwen3)
        qwen_tokenizer: loaded Qwen3 tokenizer (from load_qwen3)
        system_prompt:  system message (override with custom instructions)
        max_new_tokens: generation limit
        temperature:    sampling temperature (lower = more deterministic)

    Returns:
        Stripped explanation string.
    """
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user',   'content': prompt_text},
    ]

    try:
        text = qwen_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        text = qwen_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    dev    = next(qwen_model.parameters()).device
    inputs = qwen_tokenizer(text, return_tensors='pt').to(dev)

    with torch.no_grad():
        out = qwen_model.generate(
            **inputs,
            max_new_tokens = max_new_tokens,
            temperature    = temperature,
            do_sample      = True,
            pad_token_id   = qwen_tokenizer.eos_token_id,
        )

    new_tokens = out[0][inputs.input_ids.shape[1]:]
    raw        = qwen_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Strip thinking block if it leaked through
    if '<think>' in raw and '</think>' in raw:
        raw = raw[raw.index('</think>') + len('</think>'):].strip()

    return raw
