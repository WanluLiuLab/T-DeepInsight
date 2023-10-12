from collections import Counter
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


from ._collator import TCRabCollatorForVJCDR3
from ..utils._tensor_utils import one_hot

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True
                
class TrainerBase(ABC):
    def attach_train_dataset(self, train_dataset: Optional[datasets.Dataset]):
        self.train_dataset = train_dataset

    def attach_test_dataset(self, test_dataset: Optional[datasets.Dataset]):
        self.test_dataset = test_dataset

class TCRabTrainer(TrainerBase):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module],
        collator: TCRabCollatorForVJCDR3,
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

    def fit(
        self,
        *,
        max_epoch: int,
        max_train_sequence: int = 0,
        n_per_batch: int = 10,
        shuffle: bool = False,
        balance_label: bool = False,
        label_weight: Optional[torch.Tensor] = None,
        show_progress: bool = False,
        early_stopping: bool = False
    ):
        """Main training entry point"""
        loss = []
        self.model.train()
        if early_stopping:
            early_stopper = EarlyStopping()
        if max_train_sequence == 0:
            max_train_sequence = len(self.train_dataset)
        for i in range(1, max_epoch + 1):
            epoch_total_loss = []
            if show_progress:
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

                if 'cdr3_mr_mask' in self.train_dataset.features.keys():
                    cdr3_mr_mask = np.array(
                        self.train_dataset[j:j+n_per_batch]['cdr3_mr_mask']
                    )
                else:
                    cdr3_mr_mask = self.train_dataset[j:j+n_per_batch]['attention_mask']
                    
                epoch_token_type_ids = torch.tensor(
                    self.train_dataset[j:j+n_per_batch]['token_type_ids'], dtype=torch.int64
                ).to(self.device)

                epoch_indices_mlm, epoch_attention_mask_mlm = self.collator(
                    epoch_indices, epoch_attention_mask, cdr3_mr_mask
                )
                # print(tokenizer.convert_ids_to_tokens(torch.tensor(epoch_indices_mlm)))
                epoch_indices_mlm = epoch_indices_mlm.to(self.device)
                epoch_attention_mask_mlm = epoch_attention_mask_mlm.to(self.device)
                epoch_label_ids = None

                if 'tcr_label' in self.train_dataset.features.keys():
                    epoch_label_ids = torch.tensor(
                        self.train_dataset[j:j+n_per_batch]['tcr_label'], dtype=torch.int64
                    ).to(self.device)
                    if balance_label:
                        label_counter = Counter(epoch_label_ids.cpu().numpy())
                        min_label_number = np.min(list(label_counter.values()))
                        min_label = {v:k for k,v in label_counter.items()}[min_label_number]
                        min_indices = np.argwhere(epoch_label_ids.cpu().numpy() == min_label).squeeze()
                        for k in label_counter.keys():
                            if k != min_label:
                                indices = np.argwhere(epoch_label_ids.cpu().numpy() == k).squeeze()
                                np.random.shuffle(indices)
                                indices = indices[:min_label_number]
                                min_indices = np.concatenate([min_indices, indices])
                        epoch_indices_mlm = epoch_indices_mlm[min_indices]
                        epoch_attention_mask_mlm = epoch_attention_mask_mlm[min_indices]
                        epoch_label_ids = epoch_label_ids[min_indices]
                        epoch_token_type_ids = epoch_token_type_ids[min_indices]
                        epoch_indices = epoch_indices[min_indices]


                self.optimizer.zero_grad()
                output = self.model.forward(
                    input_ids = epoch_indices_mlm,
                    attention_mask = epoch_attention_mask_mlm,
                    token_type_ids =  epoch_token_type_ids,
                    labels = torch.tensor(epoch_indices, 
                    dtype=torch.int64).to(self.device)
                )

                if epoch_label_ids is not None:
                    prediction_loss = nn.BCEWithLogitsLoss(
                        weight=label_weight.to(self.device)
                    )(
                        output["prediction_out"], 
                        one_hot(epoch_label_ids.unsqueeze(1), 4)
                    )
                else: 
                    prediction_loss = torch.tensor(0.)

                total_loss = output["output"].loss * self.loss_weight['reconstruction_loss'] + prediction_loss
                
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                epoch_total_loss.append( total_loss.item() )

                if show_progress:
                    pbar.update(1)
                    pbar.set_postfix({
                        f'Learning rate': self.optimizer.param_groups[0]['lr'],
                        f'Loss': np.mean(epoch_total_loss[-4:]),
                        f'Prediction': prediction_loss.item()
                    })

            if self.scheduler is not None:
                self.scheduler.step(np.mean(epoch_total_loss))

            if early_stopping:
                early_stopper(np.mean(epoch_total_loss))
                if early_stopper.early_stop:
                    print("Early stopping at epoch {}".format(i))
                    break
            if show_progress:
                pbar.close()

            print("epoch {} | total loss {:.2e}".format(i, np.mean(epoch_total_loss)))
            
            loss.append(np.mean(epoch_total_loss))
        self.model.train(False)
        return loss



    def evaluate(
        self,
        *,
        n_per_batch: int = 10,
        show_progress: bool = False
    ):
        """Main training entry point"""
        self.model.eval()
        max_train_sequence = len(self.train_dataset)
        max_test_sequence = len(self.test_dataset)


        ### Train
        if show_progress:
            pbar = tqdm.tqdm(total=max_train_sequence // n_per_batch)
        all_train_result = {
            "aa": [],
            "aa_pred": [],
            "aa_gt": [],
            "av": [],
            "bv": []
        }
        for j in range(0, max_train_sequence, n_per_batch):
            self.optimizer.zero_grad()
            epoch_indices = np.array(
                    self.train_dataset[j:j+n_per_batch]['input_ids']
            )

            epoch_attention_mask = np.array(
                    self.train_dataset[j:j+n_per_batch]['attention_mask']
            )

            if 'cdr3_mr_mask' in self.train_dataset.features.keys():
                cdr3_mr_mask = np.array(
                    self.train_dataset[j:j+n_per_batch]['cdr3_mr_mask']
                )
            else:
                cdr3_mr_mask = self.train_dataset[j:j+n_per_batch]['attention_mask']
                    
            epoch_token_type_ids = torch.tensor(
                    self.train_dataset[j:j+n_per_batch]['token_type_ids'], dtype=torch.int64
            ).to(self.device)

            epoch_indices_mlm, epoch_attention_mask_mlm = self.collator(
                    epoch_indices, epoch_attention_mask, cdr3_mr_mask
            )
            # print(tokenizer.convert_ids_to_tokens(torch.tensor(epoch_indices_mlm)))
            epoch_indices_mlm = epoch_indices_mlm.to(self.device)
            epoch_attention_mask_mlm = epoch_attention_mask_mlm.to(self.device)
            epoch_label_ids = None
            if 'tcr_label' in self.train_dataset.features.keys():
                epoch_label_ids = torch.tensor(
                            self.train_dataset[j:j+n_per_batch]['tcr_label'], dtype=torch.int64
                ).to(self.device)


            self.optimizer.zero_grad()
            output = self.model.forward(
                        input_ids = epoch_indices_mlm,
                        attention_mask = epoch_attention_mask_mlm,
                        token_type_ids =  epoch_token_type_ids,
                        # tcr_label_ids = epoch_label_ids,
                        labels = torch.tensor(epoch_indices, 
                        dtype=torch.int64).to(self.device)
            )

            predictions = output['output']['logits'].topk(1)[1].squeeze()
            evaluate_mask = epoch_indices_mlm == 22
            ground_truth = torch.tensor(epoch_indices, dtype=torch.int64).to(self.device)
            evaluate_mask_trav = torch.zeros_like(evaluate_mask, dtype=torch.bool)
            evaluate_mask_trav[:, 0] = True
            evaluate_mask_trbv = torch.zeros_like(evaluate_mask, dtype=torch.bool)
            evaluate_mask_trbv[:, 48] = True
            evaluate_mask[:, [0, 48]] = False
            result = (predictions == ground_truth)[evaluate_mask].detach().cpu().numpy()
            all_train_result['aa'].append(result)
            all_train_result['aa_pred'].append(predictions[evaluate_mask].detach().cpu().numpy())
            all_train_result['aa_gt'].append(ground_truth[evaluate_mask].detach().cpu().numpy())
            result = (predictions == ground_truth)[evaluate_mask_trav].detach().cpu().numpy()
            all_train_result['av'].append(result)
            result = (predictions == ground_truth)[evaluate_mask_trbv].detach().cpu().numpy()
            all_train_result['bv'].append(result)
            if show_progress:
                pbar.update(1)

        ### Test
        if show_progress:
            pbar = tqdm.tqdm(total=max_test_sequence // n_per_batch)
        all_test_result = {
            "aa": [],
            "av": [],
            "bv": []
        }
        for j in range(0, max_test_sequence, n_per_batch):
            self.optimizer.zero_grad()
            epoch_indices = np.array(
                    self.test_dataset[j:j+n_per_batch]['input_ids']
            )

            epoch_attention_mask = np.array(
                    self.test_dataset[j:j+n_per_batch]['attention_mask']
            )

            if 'cdr3_mr_mask' in self.test_dataset.features.keys():
                cdr3_mr_mask = np.array(
                    self.test_dataset[j:j+n_per_batch]['cdr3_mr_mask']
                )
            else:
                cdr3_mr_mask = self.test_dataset[j:j+n_per_batch]['attention_mask']
                    
            epoch_token_type_ids = torch.tensor(
                    self.test_dataset[j:j+n_per_batch]['token_type_ids'], dtype=torch.int64
            ).to(self.device)

            epoch_indices_mlm, epoch_attention_mask_mlm = self.collator(
                    epoch_indices, epoch_attention_mask, cdr3_mr_mask
            )
            # print(tokenizer.convert_ids_to_tokens(torch.tensor(epoch_indices_mlm)))
            epoch_indices_mlm = epoch_indices_mlm.to(self.device)
            epoch_attention_mask_mlm = epoch_attention_mask_mlm.to(self.device)
            epoch_label_ids = None
            if 'tcr_label' in self.test_dataset.features.keys():
                epoch_label_ids = torch.tensor(
                            self.test_dataset[j:j+n_per_batch]['tcr_label'], dtype=torch.int64
                ).to(self.device)


            self.optimizer.zero_grad()
            output = self.model.forward(
                        input_ids = epoch_indices_mlm,
                        attention_mask = epoch_attention_mask_mlm,
                        token_type_ids =  epoch_token_type_ids,
                        # tcr_label_ids = epoch_label_ids,
                        labels = torch.tensor(epoch_indices, 
                        dtype=torch.int64).to(self.device)
            )

            predictions = output['output']['logits'].topk(1)[1].squeeze()
            evaluate_mask = epoch_indices_mlm == 22
            ground_truth = torch.tensor(epoch_indices, dtype=torch.int64).to(self.device)
            evaluate_mask_trav = torch.zeros_like(evaluate_mask, dtype=torch.bool)
            evaluate_mask_trav[:, 0] = True
            evaluate_mask_trbv = torch.zeros_like(evaluate_mask, dtype=torch.bool)
            evaluate_mask_trbv[:, 48] = True
            evaluate_mask[:, [0, 48]] = False
            result = (predictions == ground_truth)[evaluate_mask].detach().cpu().numpy()
            all_test_result['aa'].append(result)
            result = (predictions == ground_truth)[evaluate_mask_trav].detach().cpu().numpy()
            all_test_result['av'].append(result)
            result = (predictions == ground_truth)[evaluate_mask_trbv].detach().cpu().numpy()
            all_test_result['bv'].append(result)
            if show_progress:
                pbar.update(1)
        
        if show_progress:
            pbar.close()
        return all_train_result, all_test_result