import torch
from torch.nn import functional as F
from typing import Union, Iterable, List, Tuple, Optional
from einops import rearrange, repeat
from sklearn.model_selection import train_test_split
import datasets
import numpy as np

from transformers import (
    PreTrainedTokenizerBase,
)

from ..utils._tcr import (
    HumanTCRAnnotations,
    MouseTCRAnnotations
)

from ..utils._tensor_utils import get_k_elements

from ..utils._compat import Literal

from ..utils._amino_acids import (
    _AMINO_ACIDS_ADDITIONALS,
    _AMINO_ACIDS_INDEX_REVERSE,
    _AMINO_ACIDS_INDEX
)

class AminoAcidTokenizer(PreTrainedTokenizerBase):
    """Tokenizer for amino acids. The amino acid to token index follows the same layout as tcr-bert"""
    def __init__(self, 
        *,
        model_max_length: int,
        append_cls_token: bool = True,
        append_eos_token: bool = True,
        **kwargs
    ) -> None:
        # A special token representing an out-of-vocabulary token.
        if "pad_token" not in kwargs.keys() or not kwargs.get("pad_token", None):
            kwargs["pad_token"] = _AMINO_ACIDS_ADDITIONALS["PAD"] 
        # A special token representing the class of the input
        if "unk_token" not in kwargs.keys()  or not kwargs.get("unk_token", None):
            kwargs["unk_token"] = _AMINO_ACIDS_ADDITIONALS["UNK"]
        # A special token representing a masked token (used by masked-language modeling pretraining objectives, like BERT). 
        if "mask_token" not in kwargs.keys()  or not kwargs.get("mask_token", None):
            kwargs["mask_token"] = _AMINO_ACIDS_ADDITIONALS["PAD"] 
        if "sep_token" not in kwargs.keys()  or not kwargs.get("sep_token", None):
            kwargs["sep_token"] = _AMINO_ACIDS_ADDITIONALS["SEP"] 
        if "cls_token" not in kwargs.keys()  or not kwargs.get("cls_token", None):
            kwargs["cls_token"] = _AMINO_ACIDS_ADDITIONALS["CLS"] 

        kwargs["model_max_length"] = model_max_length
        super(AminoAcidTokenizer, self).__init__(**kwargs)
        self._vocab_size = len(_AMINO_ACIDS_INDEX)
        self.append_cls_token = append_cls_token
        self.append_eos_token = append_eos_token
        self._pad_token_id = _AMINO_ACIDS_INDEX[kwargs["pad_token"]]
        self._unk_token_id = _AMINO_ACIDS_INDEX[kwargs["unk_token"]]
        self._mask_token_id = _AMINO_ACIDS_INDEX[kwargs["mask_token"]]
        self._sep_token_id = _AMINO_ACIDS_INDEX[kwargs["sep_token"]]
        self._cls_token_id = _AMINO_ACIDS_INDEX[kwargs["cls_token"]]

    @property
    def is_fast(self) -> bool:
        return False

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def _encode(self, aa: str, max_length:int = None) -> torch.Tensor:
        if self.append_cls_token and not aa.startswith(
            _AMINO_ACIDS_ADDITIONALS["CLS"]
        ):
            aa = _AMINO_ACIDS_ADDITIONALS["CLS"] + aa

        if self.append_eos_token and not aa.startswith(
            _AMINO_ACIDS_ADDITIONALS["SEP"]
        ):
            aa += _AMINO_ACIDS_ADDITIONALS["SEP"]

        max_length = max_length or self.model_max_length
        if len(aa) < max_length:
            aa = aa + self._pad_token * (max_length - len(aa))
        return torch.Tensor(list(map(lambda a: _AMINO_ACIDS_INDEX[a], aa)))
    
    def _unpad(self, s) -> str:
        return ''.join(list(filter(lambda x: x != self.pad_token, s)))

    def _decode(self, ids) -> str:
        return self._unpad(list(map(lambda t: _AMINO_ACIDS_INDEX_REVERSE[t], ids)))

    def convert_tokens_to_ids(self, sequence: Union[Iterable[str], str]) -> torch.Tensor:
        if isinstance(sequence, str):
            ids = rearrange(self._encode(sequence), '(n h) -> n h', n = 1).type(torch.LongTensor)
        elif isinstance(sequence, Iterable):
            ids = torch.vstack(list(map(lambda x: self._encode(x), sequence))).type(torch.LongTensor)
        mask = self.convert_ids_to_mask(ids)
        return {"indices": ids, "mask": mask}

    def convert_ids_to_tokens(
        self, 
        ids: Union[torch.Tensor, np.ndarray, Iterable[int]]
    ) -> Iterable[str]:
        ids = ids.detach().cpu().numpy().astype(np.int64)
        if len(ids.shape) == 1:
            return self._decode(ids)
        else:
            return list(map(self._decode, ids))
    
    def convert_ids_to_mask(self, ids: torch.Tensor) -> torch.Tensor:
        return (ids != self._pad_token_id) & (ids != self._cls_token_id) & (ids != self._sep_token_id)

    def to_dataset(
        self, 
        ids: Iterable[str],
        chains: Iterable[str], 
        
        split: bool = False
    ) -> datasets.DatasetDict:
        # sourcery skip: remove-unnecessary-else, swap-if-else-branches
        assert(len(ids) == len(chains))
        tokenized = self.convert_tokens_to_ids(chains)
        if split:
            train_idx, test_idx = train_test_split(list(range(len(ids))))
            return datasets.DatasetDict({
                "train": datasets.Dataset.from_dict({
                    "id": np.array(ids)[train_idx],
                    "chains": np.array(chains)[train_idx],
                    "input_ids": tokenized["indices"][train_idx],
                    "attention_mask": tokenized["mask"][train_idx],
                }),
                "test": datasets.Dataset.from_dict({
                    "id": np.array(ids)[test_idx],
                    "chains": np.array(chains)[test_idx],
                    "input_ids": tokenized["indices"][test_idx],
                    "attention_mask": tokenized["mask"][test_idx],
                })
            })
        else: 
            return datasets.DatasetDict({
                "train": datasets.Dataset.from_dict({
                    "id": np.array(ids),
                    "chains": np.array(chains),
                    "input_ids": tokenized["indices"],
                    "attention_mask": tokenized["mask"],
                }),
            })


class TRABTokenizer(AminoAcidTokenizer):
    """Tokenizer for TRA and TRB sequence"""
    def __init__(self,
        *,
        tra_max_length:int,
        trb_max_length:int,
        pad_token: Optional[str] = None,
        unk_token: Optional[str] = None,
        mask_token: Optional[str] = None,
        cls_token: Optional[str] = None,
        sep_token: Optional[str] = None,
        species: Literal['human', 'mouse'] = 'human',
        **kwargs
    ) -> None:
        # A special token representing an out-of-vocabulary token.
        kwargs["pad_token"] = pad_token or _AMINO_ACIDS_ADDITIONALS["PAD"]
        # A special token representing the class of the input
        kwargs["unk_token"] = unk_token or _AMINO_ACIDS_ADDITIONALS["UNK"]
        # A special token representing a masked token (used by masked-language modeling pretraining objectives, like BERT). 
        kwargs["mask_token"] = mask_token or _AMINO_ACIDS_ADDITIONALS["MASK"]

        kwargs["cls_token"] = cls_token or _AMINO_ACIDS_ADDITIONALS["CLS"]
        kwargs["sep_token"] = sep_token or _AMINO_ACIDS_ADDITIONALS["SEP"]

        super(TRABTokenizer, self).__init__(model_max_length = tra_max_length + trb_max_length, **kwargs)
        self.tra_max_length = tra_max_length
        self.trb_max_length = trb_max_length
        if species == 'human':
            self.VJ_GENES2INDEX = HumanTCRAnnotations.VJ_GENES2INDEX
            self.VJ_GENES2INDEX_REVERSE = HumanTCRAnnotations.VJ_GENES2INDEX_REVERSE
        elif species == 'mouse':
            self.VJ_GENES2INDEX = MouseTCRAnnotations.VJ_GENES2INDEX
            self.VJ_GENES2INDEX_REVERSE = MouseTCRAnnotations.VJ_GENES2INDEX_REVERSE


    def _encode(self, aa: str, v_gene: str = None, j_gene: str = None, max_length: int = None) -> torch.Tensor:
        aa = list(aa)
        if self.append_cls_token:
            if aa[0] != _AMINO_ACIDS_ADDITIONALS["CLS"]:
                aa = [_AMINO_ACIDS_ADDITIONALS["CLS"]] + aa
            if aa[-1] != _AMINO_ACIDS_ADDITIONALS["SEP"]:
                aa += [_AMINO_ACIDS_ADDITIONALS["SEP"]]
        max_length = max_length or self.model_max_length
        if v_gene and j_gene:
            aa = [v_gene] + aa + [j_gene]
        if len(aa) < max_length:
            aa += list(self._pad_token * (max_length - len(aa)))
        return torch.Tensor(list(map(lambda a: _AMINO_ACIDS_INDEX.get(a) if a in _AMINO_ACIDS_INDEX.keys() else self.VJ_GENES2INDEX.get(a, 0) + len(_AMINO_ACIDS_INDEX), aa)))

    def convert_tokens_to_ids(self, sequence: Union[List[Tuple[str]], Tuple[str]], alpha_vj: Optional[Union[List[Tuple[str]], Tuple[str]]] = None, beta_vj: Optional[Union[List[Tuple[str]], Tuple[str]]] = None):
        # sourcery skip: none-compare, swap-if-else-branches
        if isinstance(sequence, list):
            if alpha_vj != None and beta_vj != None:
                ids = torch.hstack([
                    torch.vstack(
                        list(map(lambda x: self._encode(x[0], x[1][0], x[1][1], max_length=self.tra_max_length), 
                        zip(get_k_elements(sequence, 0), alpha_vj)))
                    ), 
                    torch.vstack(
                        list(map(lambda x: self._encode(x[0], x[1][0], x[1][1], max_length=self.trb_max_length), 
                        zip(get_k_elements(sequence, 1), beta_vj)))
                    )]).type(torch.LongTensor) 
                    
            else:
                ids = torch.hstack([
                    torch.vstack(list(map(lambda x: self._encode(x[0], max_length=self.tra_max_length), sequence))), 
                    torch.vstack(list(map(lambda x: self._encode(x[1], max_length=self.trb_max_length), sequence)))
                ]).type(torch.LongTensor)

        elif alpha_vj != None and beta_vj != None:
            ids = rearrange(torch.hstack([self._encode(sequence[0], alpha_vj[0], alpha_vj[1], max_length=self.tra_max_length), self._encode(sequence[1], beta_vj[0], beta_vj[1], max_length=self.tra_max_length)]), '(n h) -> n h', n=1).type(torch.LongTensor)

        else:
            ids = rearrange(torch.hstack([self._encode(sequence[0], max_length=self.tra_max_length), self._encode(sequence[1], max_length=self.tra_max_length)]), '(n h) -> n h', n=1).type(torch.LongTensor)

        mask = self.convert_ids_to_mask(ids)
        token_type_ids = torch.hstack([torch.zeros(ids.shape[0], self.tra_max_length), torch.ones(ids.shape[0], self.trb_max_length)]).type(torch.LongTensor)

        return {"indices": ids, "mask": mask, "token_type_ids": token_type_ids}

    def _decode(self, ids):
        return list(map(lambda t: _AMINO_ACIDS_INDEX_REVERSE[t] if t in _AMINO_ACIDS_INDEX_REVERSE.keys() else self.VJ_GENES2INDEX_REVERSE[t - len(_AMINO_ACIDS_INDEX)], ids))

    def _trab_decode(self, ids):
        dec = self._decode(ids)
        return (self._unpad(dec[:self.tra_max_length]), self._unpad(dec[self.tra_max_length:]))

    def convert_ids_to_tokens(self, ids: torch.Tensor):
        ids = ids.detach().cpu().numpy().astype(np.int64)
        if len(ids.shape) == 1:
            return self._trab_decode(ids)
        else:
            return list(map(lambda x: self._trab_decode(x), ids))

    def to_dataset(
            self, 
            ids: Iterable[str],
            alpha_chains: Iterable[str], 
            beta_chains: Iterable[str], 
            alpha_v_genes: Iterable[str] = None,
            alpha_j_genes: Iterable[str] = None,
            beta_v_genes: Iterable[str] = None,
            beta_j_genes: Iterable[str] = None,
            pairing: Iterable[int] = None,
            split: bool = False
        ) -> datasets.DatasetDict:
        if not len(ids) == len(alpha_chains) == len(beta_chains):
            raise ValueError("Length of ids(%d), alpha_chains(%d) and beta_chains(%d) do not match" % (len(ids), len(alpha_chains), len(beta_chains)))
        alpha_chains = list(alpha_chains)
        beta_chains = list(beta_chains)
        if all(map(lambda x: x is not None, [alpha_v_genes, alpha_j_genes, beta_v_genes, beta_j_genes])):
            tokenized = self.convert_tokens_to_ids(
                list(zip(alpha_chains, beta_chains)), 
                list(zip(alpha_v_genes, alpha_j_genes)),
                list(zip(beta_v_genes, beta_j_genes)),
            )
        else:
            tokenized = self.convert_tokens_to_ids(list(zip(alpha_chains, beta_chains)))
        if split:
            train_idx, test_idx = train_test_split(list(range(len(ids))))
            return datasets.DatasetDict({
                "train": datasets.Dataset.from_dict({
                    "id": np.array(ids)[train_idx],
                    "alpha_chains": np.array(alpha_chains)[train_idx],
                    "beta_chains": np.array(beta_chains)[train_idx],
                    "input_ids": tokenized["indices"][train_idx],
                    "token_type_ids": tokenized["token_type_ids"][train_idx],
                    "attention_mask": tokenized["mask"][train_idx],
                    "pairing": np.array(pairing)[train_idx] if pairing is not None else np.ones(len(train_idx), dtype=np.uint8)
                }),
                "test": datasets.Dataset.from_dict({
                    "id": np.array(ids)[test_idx],
                    "alpha_chains": np.array(alpha_chains)[test_idx],
                    "beta_chains": np.array(beta_chains)[test_idx],
                    "input_ids": tokenized["indices"][test_idx],
                    "token_type_ids": tokenized["token_type_ids"][test_idx],
                    "attention_mask": tokenized["mask"][test_idx],
                    "pairing": np.array(pairing)[test_idx] if pairing is not None else np.ones(len(test_idx), dtype=np.uint8)
                })
            })
        else:
            return datasets.DatasetDict({
                "train": datasets.Dataset.from_dict({
                    "id": ids,
                    "alpha_chains": alpha_chains,
                    "beta_chains": beta_chains,
                    "input_ids": tokenized["indices"],
                    "token_type_ids": tokenized["token_type_ids"],
                    "attention_mask": tokenized["mask"],
                    "pairing": np.array(pairing) if pairing is not None else np.ones(len(alpha_chains), dtype=np.uint8)
                })
            })
    
    def __call__(self):
        raise NotImplementedError


