import torch
from torch import nn
from torch.nn import functional as F
import json
import tqdm
from abc import ABC
import datasets
from typing import Optional, Union, Tuple, Mapping
from transformers import PreTrainedModel
import numpy as np
from pathlib import Path


from ._collator import TRABCollator, TRABMutator

class TrainerBase(ABC):

    def attach_train_dataset(self, train_dataset: Optional[datasets.Dataset]):
        self.train_dataset = train_dataset

    def attach_test_dataset(self, test_dataset: Optional[datasets.Dataset]):
        self.test_dataset = test_dataset

class TrainerMixin(TrainerBase):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module],
        collator: TRABCollator,
        train_dataset: Optional[datasets.Dataset] = None,
        test_dataset: Optional[datasets.Dataset] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        loss_weight: Mapping[str, float] = {"reconstruction_loss": 1, "triplet_loss": 1},
        device: str = 'cuda'
    ) -> None:
        self.model = model
        self.optimizer, self.scheduler = optimizers
        self.device = device
        self.collator = collator 
        self.loss_weight = loss_weight
        self.attach_train_dataset(train_dataset)
        self.attach_test_dataset(test_dataset)

    def train(
        self,
        *,
        max_epoch: int,
        tra_max_length: int = 36,
        trb_max_length: int = 36,
        max_train_sequence: int,
        n_per_batch: int = 10,
        focus_trb = True,
        shuffle: bool = False,
        show_progress_bar: bool = False
    ):
        """Main training entry point"""
        loss = []
        self.model.train()
        for i in range(1, max_epoch + 1):
            epoch_total_loss = 0
            epoch_batch_loss = 0
            epoch_triplet_loss = 0
            if show_progress_bar:
                pbar = tqdm.tqdm(total=max_train_sequence // n_per_batch)

            for j in range(0, max_train_sequence, n_per_batch):
                if shuffle: self.train_dataset.shuffle()
                self.optimizer.zero_grad()
                epoch_indices = np.array(
                    self.train_dataset[j:j+n_per_batch]['input_ids']
                )

                epoch_attention_mask = np.array(
                    self.train_dataset[j:j+n_per_batch]['attention_mask']
                )

                epoch_token_type_ids = torch.tensor(
                    self.train_dataset[j:j+n_per_batch]['token_type_ids'], dtype=torch.int64
                ).to(self.device)

                epoch_indices_mlm, epoch_attention_mask_mlm = self.collator(
                    epoch_indices, epoch_attention_mask
                )

                epoch_indices_mlm = epoch_indices_mlm.to(self.device)
                epoch_attention_mask_mlm = epoch_attention_mask_mlm.to(self.device)
                epoch_batch_ids = None
                if 'tcr_batch' in self.train_dataset.features.keys():
                    epoch_batch_ids = torch.tensor(
                        self.train_dataset[j:j+n_per_batch]['tcr_batch'], dtype=torch.int64
                    ).to(self.device)


                self.optimizer.zero_grad()
                output = self.model.forward(
                        input_ids = epoch_indices_mlm,
                        attention_mask = epoch_attention_mask_mlm,
                        token_type_ids =  epoch_token_type_ids,
                        tcr_batch_ids = epoch_batch_ids,
                        labels = torch.tensor(epoch_indices, 
                        dtype=torch.int64).to(self.device)
                    )

                triplet_loss = torch.tensor(0.).to(self.device)

                if focus_trb:
                    mutator_tra_for_tra, mutator_trb_for_tra = (
                        TRABMutator(tra_max_length,trb_max_length,is_full_length=False,mutate_trb_probability=0,max_mutation_aa=1),
                        TRABMutator(tra_max_length,trb_max_length,is_full_length=False,mutate_trb_probability=1,max_mutation_aa=4)
                    )
                    mutator_tra_for_trb, mutator_trb_for_trb = (
                        TRABMutator(tra_max_length,trb_max_length,is_full_length=False,mutate_trb_probability=0,max_mutation_aa=4),
                        TRABMutator(tra_max_length,trb_max_length,is_full_length=False,mutate_trb_probability=1,max_mutation_aa=1)
                    )
                    criterion = nn.TripletMarginWithDistanceLoss(distance_function=nn.PairwiseDistance(), reduction='none')
                    # We start metrics learning here for similar motif


                    epoch_indices_mt_trb, epoch_attention_mask_mt_trb = mutator_tra_for_trb(
                        input_ids = epoch_indices,
                        attention_mask = epoch_attention_mask,
                    )
                    epoch_indices_mt_trb, epoch_attention_mask_mt_trb = mutator_trb_for_trb(
                        input_ids = epoch_indices_mt_trb,
                        attention_mask = epoch_attention_mask_mt_trb
                    )

                    epoch_indices_mt_tra, epoch_attention_mask_mt_tra = mutator_tra_for_tra(
                        input_ids = epoch_indices,  
                        attention_mask = epoch_attention_mask
                    )
                    epoch_indices_mt_tra, epoch_attention_mask_mt_tra = mutator_trb_for_tra(
                        input_ids = epoch_indices_mt_tra, 
                        attention_mask = epoch_attention_mask_mt_tra
                    )
                    epoch_indices_mt_tra = epoch_indices_mt_tra.to(self.device)
                    epoch_attention_mask_mt_tra = epoch_attention_mask_mt_tra.to(self.device)         

                    epoch_indices_mt_trb = epoch_indices_mt_trb.to(self.device)
                    epoch_attention_mask_mt_trb = epoch_attention_mask_mt_trb.to(self.device)
                    epoch_indices = torch.tensor(epoch_indices).to(self.device)
                    epoch_attention_mask = torch.tensor(epoch_attention_mask).to(self.device)

                    step = int(np.ceil(epoch_indices_mlm.shape[0]/3))

                    for i in range(3):
                        stacked_input_ids = torch.vstack([
                            epoch_indices[i*step:(i+1)*step],
                            epoch_indices_mt_trb[i*step:(i+1)*step],
                            epoch_indices_mt_tra[i*step:(i+1)*step],
                        ])
                        stacked_attention_mask = torch.vstack([
                            epoch_attention_mask[i*step:(i+1)*step],
                            epoch_attention_mask_mt_trb[i*step:(i+1)*step],
                            epoch_attention_mask_mt_tra[i*step:(i+1)*step],
                        ])

                        stacked_token_type_ids = epoch_token_type_ids[0].repeat(
                            stacked_input_ids.shape[0],1
                        ).to(stacked_input_ids.device)

                        stacked_output = self.model(
                            input_ids = stacked_input_ids,
                            attention_mask = stacked_attention_mask,
                            labels = stacked_input_ids,
                            token_type_ids = stacked_token_type_ids,
                        )
                        stacked_hidden_states = stacked_output["hidden_states"]
                        curstep = int(stacked_hidden_states.shape[0]/3)
                        triplet_loss += criterion(
                            stacked_hidden_states[:curstep], 
                            stacked_hidden_states[1*curstep:2*curstep],
                            stacked_hidden_states[2*curstep:3*curstep]
                        ).mean()

                total_loss = output["output"].loss * self.loss_weight['reconstruction_loss'] + \
                             output["batch_loss"] + \
                             triplet_loss * self.loss_weight['triplet_loss']
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                epoch_total_loss += total_loss.item()
                epoch_batch_loss += output["batch_loss"].item() 
                epoch_triplet_loss += triplet_loss.item()  * self.loss_weight['triplet_loss']
                if show_progress_bar:
                    pbar.update(1)
            if self.scheduler is not None:
                self.scheduler(epoch_total_loss)
                self.scheduler.step()
            if show_progress_bar:
                pbar.close()

            print("epoch {} | total loss {:.2e}, triplet loss {:.2e}".format(i, epoch_total_loss, epoch_triplet_loss))
            loss.append((epoch_total_loss, epoch_batch_loss))
        self.model.train(False)
        return loss


