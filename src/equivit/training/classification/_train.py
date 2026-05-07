import torch
import torch.nn as nn
from typing import Callable, Any, Collection, Tuple, Optional
from tqdm import tqdm


# Note: this file contains old code from hexvit. These functions are meant only for reference.


def train_one_epoch(
    model: nn.Module, 
    optimizer: torch.optim.Optimizer, 
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    trainloader: Collection[Tuple[torch.Tensor, torch.Tensor]],
    device: str = 'cuda',
    display_update_frequency: int=5,
    binary: bool = False
) -> float:
    r"""
    Train a classification model for one epoch, and return the mean training loss over the epoch.

    Args:
        model: the classification model. It should take in a tensor of shape :math:`(B, *)` and output a tensor of shape :math:`(B, N_{classes})` for multi-class classification, or :math:`(B, 1)` for binary classification.
        optimizer: the optimizer to use for training
        loss_fn: the loss function to use for training. It should take in the model's output and the labels (of shape :math:`(B,)`), and return a scalar loss.
        trainloader: a dataloader that provides the training data, where each batch is a tuple of two tensors of shapes :math:`(B, *)` and :math:`(B,)` respectively (the images and the labels)
        device: the device to use for training (e.g., 'cuda' or 'cpu')
        display_update_frequency: how many iterations before updating the current loss in the progress bar
        binary: whether it is a binary classification task (if True, squeeze the last dimension of logits and convert labels to float32 before computing loss)

        
    Returns:
        The mean training loss over the epoch.
    """

    model.train()

    prog = tqdm(enumerate(trainloader), total=len(trainloader))

    tot_loss = 0.
    total_images = 0

    for itr, (img, lbl) in prog:
        # lbl has shape (B,)
        # img hs shape (B, *)

        optimizer.zero_grad()

        # (B, N_classes) or (B, 1) for binary classification
        logits = model(img.to(device))

        if binary:
            loss = loss_fn(logits.squeeze(-1), lbl.to(torch.float32).to(device))
        else:
            loss = loss_fn(logits, lbl.to(device))

        loss.backward()
        optimizer.step()

        tot_loss += loss.detach() * img.shape[0]
        total_images += img.shape[0]

        if itr % display_update_frequency == 0:
            mean_loss = tot_loss / total_images
            prog.set_description(f'mean loss = {mean_loss:.4f}, curr loss = {loss:.4f}')

    return float(tot_loss / total_images)



def evaluate(
    model: nn.Module,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    dataloader: Collection[Tuple[torch.Tensor, torch.Tensor]],
    display_update_frequency: int = 5,
    binary: bool = False,
    device: str = 'cuda', 
):    
    """
    Evaluate a classification model on a dataset, and return the mean validation loss.
    """
    model.eval()
    prog = tqdm(enumerate(dataloader), total=len(dataloader))
    tot_loss = 0.
    total_images = 0
    for itr, (img, lbl) in prog:
        logits = model(img.to(device))

        if binary:
            loss = loss_fn(logits.squeeze(-1), lbl.to(torch.float32).to(device))
        else:
            loss = loss_fn(logits, lbl.to(device))

        tot_loss += loss.detach() * img.shape[0] # batch size
        total_images += img.shape[0]

        if itr % display_update_frequency == 0:
            mean_loss = tot_loss / total_images
            prog.set_description(f'mean loss = {mean_loss:.4f}, curr loss = {loss:.4f}')

    return tot_loss / total_images

