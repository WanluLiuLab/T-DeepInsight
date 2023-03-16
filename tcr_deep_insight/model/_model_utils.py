from ._model import *

def to_embedding_tcr_only(model, eval_dataset, k, device='cuda', n_per_batch=64, progress=False, mask_tr='none'):
    model.eval()
    all_embedding = []
    with torch.no_grad():
        if progress:
            for_range = tqdm.trange(0,len(eval_dataset),n_per_batch)
        else:
            for_range = range(0,len(eval_dataset),n_per_batch)
        for j in for_range:
           
            tcr_input_ids = torch.tensor(
                eval_dataset[j:j+n_per_batch]['input_ids'] if 'input_ids' in eval_dataset.features.keys() else  eval_dataset[j:j+n_per_batch]['tcr_input_ids']
            ).to(device)
            tcr_attention_mask = torch.tensor(
                eval_dataset[j:j+n_per_batch]['attention_mask'] if 'attention_mask' in eval_dataset.features.keys() else  eval_dataset[j:j+n_per_batch]['tcr_attention_mask']
            ).to(device)
            indices_length = int(tcr_attention_mask.shape[0]/2)
            if mask_tr == 'tra':
                #tcr_input_ids[:,2:indices_length] = _AMINO_ACIDS_INDEX[_AMINO_ACIDS_ADDITIONALS['MASK']]
                tcr_attention_mask[:,2:indices_length] = False
            elif mask_tr == 'trb':
                #tcr_input_ids[:,indices_length+2:indices_length*2] = _AMINO_ACIDS_INDEX[_AMINO_ACIDS_ADDITIONALS['MASK']]
                tcr_attention_mask[:,indices_length+2:indices_length*2] = False
            tcr_token_type_ids = torch.tensor(
                eval_dataset[j:j+n_per_batch]['token_type_ids'] if 'token_type_ids' in eval_dataset.features.keys() else  eval_dataset[j:j+n_per_batch]['tcr_token_type_ids']
            ).to(device)
      
            output = model(
                input_ids = tcr_input_ids,
                attention_mask = tcr_attention_mask,
                labels = tcr_input_ids,
                token_type_ids = tcr_token_type_ids,
            ) 
            all_embedding.append(output[k].detach().cpu().numpy())
    all_embedding = np.vstack(all_embedding)
    model.train()
    return all_embedding


def to_embedding_tcr_only_from_pandas_v2(
    model, 
    df, 
    tokenizer, 
    device,  
    n_per_batch=64, 
    mask_tr='none'
):
    all_embedding = []
    for i in tqdm.trange(0,len(df),4096):
        ds = tokenizer.to_dataset(
            ids=list(map(str, range(i,i+len(df.iloc[i:i+4096,:])))), 
            alpha_v_genes=list(df.iloc[i:i+4096,:]['TRAV']), 
            alpha_j_genes=list(df.iloc[i:i+4096,:]['TRAJ']), 
            beta_v_genes=list(df.iloc[i:i+4096,:]['TRBV']), 
            beta_j_genes=list(df.iloc[i:i+4096,:]['TRBJ']), 
            alpha_chains=list(df.iloc[i:i+4096,:]['CDR3a']), 
            beta_chains=list(df.iloc[i:i+4096,:]['CDR3b']), 
            check_full_length_sequence=False
        )['train']
        all_embedding.append(to_embedding_tcr_only(model, ds, 'hidden_states', device, mask_tr=mask_tr))
    return np.vstack(all_embedding)