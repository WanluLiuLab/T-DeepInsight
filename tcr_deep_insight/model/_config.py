# Hugginface Transformers
from transformers import BertConfig

BERT_TINY = {
    "hidden_size": 96,
    "intermediate_size": 192,
}

BERT_SMALL = {
    "hidden_size": 384,
    "intermediate_size": 768,
}

BERT_BASE = {
    "hidden_size": 768,
    "intermediate_size": 1536,
}

def get_config(
    hidden_size=768, 
    intermediate_size=1536,
    extra_vocab_size=183
):
    return {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": hidden_size,
        "initializer_range": 0.02,
        "intermediate_size": intermediate_size,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 516,
        "model_type": "bert",
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "pad_token_id": 0,
        "position_embedding_type": "absolute",
        "type_vocab_size": 2,
        "use_cache": 1,
        "vocab_size": 27+extra_vocab_size # Amino Acids + V-J genes
    }

def get_human_config(bert_type="base"):
    if bert_type == "tiny":
        return BertConfig.from_dict(get_config(**BERT_TINY))
    elif bert_type == "small":
        return BertConfig.from_dict(get_config(**BERT_SMALL))
    elif bert_type == "base":
        return BertConfig.from_dict(get_config(**BERT_BASE))
    else:
        raise ValueError("bert_type must be one of 'tiny', 'small', or 'base'")

def get_mouse_config(bert_type):
    if bert_type == "tiny":
        return BertConfig.from_dict(get_config(**BERT_TINY, extra_vocab_size=213))
    elif bert_type == "small":
        return BertConfig.from_dict(get_config(**BERT_SMALL, extra_vocab_size=213))
    elif bert_type == "base":
        return BertConfig.from_dict(get_config(**BERT_BASE, extra_vocab_size=213))
    else:
        raise ValueError("bert_type must be one of 'tiny', 'small', or 'base'")