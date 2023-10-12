from typing import Literal
import torch 
from ._tokenizer import TCRabTokenizerForVJCDR3 as TCRabTokenizerForVJCDR3
from ._collator import TCRabCollatorForVJCDR3 as TCRabCollatorForVJCDR3

def default_optimizer(model, *args, **kwargs):
    optimizer = torch.optim.AdamW(model.parameters(), *args, **kwargs), 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    return (optimizer, scheduler)

def default_tokenizer(
    species:Literal['human','mouse'],
    tra_max_length:int=48,
    trb_max_length:int=48,
):
    return TCRabTokenizerForVJCDR3(
        tra_max_length=tra_max_length, 
        trb_max_length=trb_max_length,
        species=species
    )

def default_collator(
    species:Literal['human','mouse'],
    tra_max_length:int=48,
    trb_max_length:int=48,
):
    return TCRabCollatorForVJCDR3(
        tra_max_length=tra_max_length, 
        trb_max_length=trb_max_length,
        species=species
    )

