BERT_CONFIG = lambda hidden_size: {
    "attention_probs_dropout_prob": 0.1,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "hidden_size": hidden_size,
    "initializer_range": 0.02,
    "intermediate_size": 1536,
    "layer_norm_eps": 1e-12,
    "max_position_embeddings": 516,
    "model_type": "bert",
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "pad_token_id": 0,
    "position_embedding_type": "absolute",
    "type_vocab_size": 2,
    "use_cache": 1,
    "vocab_size": 27+182 # Amino Acids + V-J genes
}