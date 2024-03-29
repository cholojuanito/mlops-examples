import torch
from tqdm import tqdm

# A basic class that encapsulates the training process through multiple epochs
# It also displays progress in the CLI

class SegmentationTrainer:
    def __init__(self, network, optimizer, loss_func, train_loader, val_loader, tracker_func, n_tracks_per_epoch, device='cuda'):
        self.network = network
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.train_dataloader = train_loader 
        self.val_dataloader = val_loader

        self.device = device
        self.tracker_func =tracker_func
        self.n_tracks_per_epoch = n_tracks_per_epoch
        self._loop = tqdm()


    def train(self, num_epochs):
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        for i in range(1, num_epochs+1):
            self._loop = tqdm(total=len(self.train_dataloader), position=0, leave=False)
            # Train
            print(f'\nTraining for epoch {i} of {num_epochs}')
            train_loss, train_acc = self._epoch_(self.train_dataloader, training=True)
            self._loop.close()

            self._loop = tqdm(total=len(self.val_dataloader), position=0, leave=False)
            # Validate
            print(f"\nValidation for epoch {i} of {num_epochs}")
            val_loss, val_acc = self._epoch_(self.val_dataloader, training=False)
            self._loop.close()
            
            # Track final metrics at end of epoch
            self.tracker_func({
                'train/train_loss': train_loss,
                'train/train_acc': train_acc,
                'val/val_loss': val_loss,
                'val/val_acc': val_acc
            })

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)

        return  train_losses, val_losses, train_accuracies, val_accuracies

    def _epoch_(self, data_loader, training=True):
        if training:
            self.network.train() 
            torch.set_grad_enabled(True)
        else:
            self.network.eval()
            torch.set_grad_enabled(False)

        loss_sum = 0
        acc_sum = 0
        for batch_idx, (imgs, labels) in enumerate(data_loader):
            imgs, labels = imgs.to(self.device, non_blocking=True), labels.to(self.device,non_blocking=True) # non_blocking is a speed up (async)
            
            if training:
                self.optimizer.zero_grad() # Set gradient to zero

            y_hat = self.network(imgs)
            loss = self.loss_func(y_hat, labels.long())
            loss_value = loss.detach().sum().item()
            loss_sum += loss_value

            
            probs = y_hat.argmax(1)
            accuracy = (probs == labels).float().mean()
            acc_value = accuracy.detach().item()
            acc_sum += acc_value

            mem_allocated = torch.cuda.memory_allocated(0) / 1e9

            self._loop.set_description('loss: {:.4f}, accuracy: {:.4f}, mem: {:.2f}'.format(loss_value, acc_value, mem_allocated))
            self._loop.update(1)

            # track metrics of the first couple of batches in epoch
            if batch_idx + 1 < self.n_tracks_per_epoch:
                if training:
                    metrics = {
                        'train/train_loss': loss_value,
                        'train/train_acc': acc_value
                    }
                else:
                    metrics = {
                        'val/val_loss': loss_value,
                        'val/val_acc': acc_value
                    }
                self.tracker_func(metrics)

            if training:
                loss.backward() # Compute gradient, for weight with respect to loss
                self.optimizer.step() # Take step in the direction of the negative gradient

        return loss_sum/batch_idx, acc_sum/batch_idx