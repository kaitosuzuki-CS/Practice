import copy


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0

        self.best_loss = float("inf")
        self.best_model = None
        self.stop = False

    def get_state_dict(self):
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "counter": self.counter,
            "best_loss": self.best_loss,
            "best_model": (
                self.best_model.state_dict() if self.best_model is not None else None
            ),
            "stop": self.stop,
        }

    def __call__(self, model, loss):
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
            self.best_model = copy.deepcopy(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
