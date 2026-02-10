from dataclasses import dataclass

@dataclass
class EarlyStopping:
    patience: int
    min_delta: float = 0.0

    def __post_init__(self):
        self.best = None
        self.num_bad = 0
        self.should_stop = False

    def step(self, current: float) -> bool:
        #lower is better
        if self.best is None or current < (self.best-self.min_delta):
            self.best = current
            self.num_bad = 0
            return True
        self.num_bad += 1
        if self.num_bad >= self.patience:
            self.should_stop = True
        return False