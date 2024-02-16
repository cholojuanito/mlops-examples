from torch.utils.data import (
    DataLoader,
    Dataset,
)

class CancerDataLoader(DataLoader):
    def __init__(
        self,
        dataset: Dataset,
        is_train: bool,
        batch_size: int | None = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__(
            dataset,
            shuffle=True if is_train else False,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
