from typing import Literal
import torch 
from ._tokenizer import TRABTokenizer as TCRabTokenizer
from ._collator import TRABCollator as TCRabCollator

def default_optimizer(model, *args, **kwargs):
    optimizer = torch.optim.AdamW(model.parameters(), *args, **kwargs), 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    return (optimizer, scheduler)

def default_tokenizer(species:Literal['human','mouse']):
    return TCRabTokenizer(
        tra_max_length=48, 
        trb_max_length=48,
        species=species
    )

def default_collator(species:Literal['human','mouse']):
    return TCRabCollator(
        tra_max_length=48, 
        trb_max_length=48,
        species=species
    )

