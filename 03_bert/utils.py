from mxnet.gluon.contrib.estimator import TrainBegin, BatchBegin

class MyLearningRateHandler(TrainBegin, BatchBegin):
    """Warm-up learning rate handler.

    Parameters
    ----------
    trainer: gluon.Trainer
        Trainer object to adjust the learning rate on.
    num_warmup_steps: int
        Number of initial steps during which the learning rate is linearly
        increased to it's target.
    num_train_steps: int
        Total number of steps to be taken during training. Should be equal to
        the number of batches * number of epochs.
    lr: float
        Base learning rate to reach after warmup.
    """

    def __init__(self, trainer, num_warmup_steps, num_train_steps, lr):
        self.trainer = trainer
        self.num_warmup_steps = num_warmup_steps
        self.num_train_steps = num_train_steps
        self.lr = lr

        self.step_num = 0

    def train_begin(self, estimator, *args, **kwargs):
        self.step_num = 0

    def batch_begin(self, estimator, *args, **kwargs):
        self.step_num += 1
        if self.step_num < self.num_warmup_steps:
            new_lr = self.lr * self.step_num / self.num_warmup_steps
        else:
            non_warmup_steps = self.step_num - self.num_warmup_steps
            offset = non_warmup_steps / (self.num_train_steps - self.num_warmup_steps)
            new_lr = self.lr - offset * self.lr
        self.trainer.set_learning_rate(new_lr)
