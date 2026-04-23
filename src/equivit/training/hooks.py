


class EvaluationHook:
    def on_eval_start(self): ...
    def eval_update(self, logits, labels): ...
    def on_eval_end(self): ...


class TrainingHook(EvaluationHook):
    """
    To be passed as an argument into the train() function.
    This allows custom code to be executed at specific points during training and evaluation. 
    """
    def on_epoch_start(self, epoch): ...
    def on_train_epoch_end(self, epoch, loss): ...
    def on_val_epoch_end(self, epoch, loss): ...



_dummy_training_hook = TrainingHook()

_dummy_eval_hook = EvaluationHook()