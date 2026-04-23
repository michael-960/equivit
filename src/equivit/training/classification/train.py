import torch
import torch.nn as nn
from typing import Callable, Any, Collection, Tuple, Optional
from tqdm import tqdm
from pathlib import Path
import h5py
import os

# from ..metrics import ConfusionMatrixCalculator

from ...registry import MODULE, OPTIMIZER, LOSS, DATASET

from ...version import __version__

# from .hooks import EvaluationHook, TrainingHook, _dummy_training_hook, _dummy_eval_hook




def train(
    savedir: str,
    model: nn.Module,
    optimizer,
    loss_fn: nn.Module,
    trainloader: torch.utils.data.DataLoader,
    valloader: torch.utils.data.DataLoader,

    N_epochs: int,

    device: str = 'cuda',
    load_epoch: Optional[int] = None,
    # deterministic: bool = False,
    # seed = None,
    checkpoint_frequency=5,
    display_update_frequency=5,
    keep_best_checkpoints=5,
    binary: bool = False,

    training_hook: TrainingHook = _dummy_training_hook,
):
    """
    Train a classification model for multiple epochs.

    Args:

        checkpoint_frequency: how many epochs per validation & save
        display_update_frequency: in one epoch, how many iterations before updating the current loss
        keep_best_checkpoints: how many checkpoints with the best validation losses to keep (if more checkpoints are saved, the checkpoint with the worst validation loss among the saved checkpoints will be removed)

    """

    epoch = 0

    model.train();

    model.to(device)

    if load_epoch is None:
        # if deterministic:
        #     torch.backends.cudnn.benchmark = False
        #     torch.backends.cudnn.deterministic = True
        #     torch.manual_seed(seed)
        #     np.random.seed(seed)
        record_file = h5py.File(f'{savedir}/record.h5', 'w-')
        record_file.create_dataset('epoch_train', shape=(0,), maxshape=(None,), chunks=(1,), dtype='int32')
        record_file.create_dataset('loss_train', shape=(0,), maxshape=(None,), chunks=(1,), dtype='float32')
        record_file.create_dataset('epoch_val', shape=(0,), maxshape=(None,), chunks=(1,), dtype='int32')
        record_file.create_dataset('loss_val', shape=(0,), maxshape=(None,), chunks=(1,), dtype='float32')
        record_file.create_dataset('confmat_val', shape=(0,), maxshape=(None,), chunks=(1,), dtype='int64')

    else:
        assert load_epoch < N_epochs
        state = torch.load(f'{savedir}/checkpoints/epoch_{load_epoch:04d}')
        epoch = load_epoch
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])

        record_file = h5py.File(f'{savedir}/record.h5', 'r+')


    # keep track of the best epochs and their validation losses, so that we only save checkpoints for the best epochs
    best_epochs = []
    best_val_losses = []
    
    while epoch < N_epochs:
        training_hook.on_epoch_start(epoch)

        print(f'Training: epoch {epoch}')

        train_loss = train_one_epoch(
                        device=device,
                        model=model, 
                        optimizer=optimizer, loss_fn=loss_fn, trainloader=trainloader, 
                        display_update_frequency=display_update_frequency,
                        binary=binary
                        )

        training_hook.on_train_epoch_end(epoch, train_loss.detach().cpu())
        
        record_file['epoch_train'].resize((epoch+1,))
        record_file['epoch_train'][-1] = epoch
        record_file['loss_train'].resize((epoch+1,))
        record_file['loss_train'][-1] = train_loss.detach().cpu()


        if epoch % checkpoint_frequency == 0:
            print('---- Validating ----')
            val_loss = evaluate(
                        device=device,
                        model=model, loss_fn=loss_fn, dataloader=valloader, 
                        display_update_frequency=display_update_frequency,
                        binary=binary,
                        eval_hook=training_hook
                        )

            training_hook.on_val_epoch_end(epoch, val_loss.detach().cpu())


            _N = record_file['epoch_val'].shape[0]

            record_file['epoch_val'].resize((_N+1,))
            record_file['epoch_val'][-1] = epoch
            record_file['loss_val'].resize((_N+1,))
            record_file['loss_val'][-1] = val_loss.detach().cpu()


            worst_val_loss_in_best_epochs = max(best_val_losses, default=float('inf'))

            if val_loss < worst_val_loss_in_best_epochs:
                best_epochs.append(epoch)
                best_val_losses.append(val_loss.detach().cpu())

                # only save checkpoint when the current val loss is better than the worst val loss in the best epochs
                torch.save({
                    'hexvit_version': __version__,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    # 'rng_state': torch.get_rng_state()
                }, f'{savedir}/checkpoints/epoch_{epoch:04d}.pt')


                # if we have more best epochs than the number of checkpoints we want to keep, remove the checkpoint with the worst val loss in the best epochs
                if len(best_epochs) > keep_best_checkpoints:
                    worst_idx = best_val_losses.index(worst_val_loss_in_best_epochs)
                    worst_epoch = best_epochs[worst_idx]

                    try:
                        os.remove(f'{savedir}/checkpoints/epoch_{worst_epoch:04d}.pt')
                    except FileNotFoundError:
                        print(f'Warning: checkpoint for epoch {worst_epoch} not found when trying to remove it.')

                    best_epochs.pop(worst_idx)
                    best_val_losses.pop(worst_idx)

            model.train()

        epoch += 1

    record_file.close()



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
    eval_hook: EvaluationHook = _dummy_eval_hook
):    
    """
    Evaluate a classification model on a dataset, and return the mean validation loss.
    """
    eval_hook.on_eval_start()

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

        eval_hook.eval_update(logits.detach().cpu(), lbl.detach().cpu())   

        tot_loss += loss.detach() * img.shape[0] # batch size
        total_images += img.shape[0]

        if itr % display_update_frequency == 0:
            mean_loss = tot_loss / total_images
            prog.set_description(f'mean loss = {mean_loss:.4f}, curr loss = {loss:.4f}')

    eval_hook.on_eval_end()
    
    return tot_loss / total_images

