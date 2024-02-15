import torch
from tqdm import tqdm

# A basic class that encapsulates the training process through multiple epochs
# It also displays progress in the CLI

class SegmentationTrainer:
    def __init__(self):
        self.network = None
        self.optimizer = None
        self.loss_func = None
        self.train_dataloader = None 
        self.val_dataloader = None

        self.device = 'cuda'
        self.epoch_count = 0
        self.loop = None

    def increment_epoch_count(self):
        self.epoch_count += 1

    def train(self, num_epochs):
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        for i in range(1, num_epochs+1):
            self.loop = tqdm(total=len(self.train_dataloader), position=0, leave=False)
            # Train
            print(f'\nTraining for epoch {i} of {num_epochs}')
            train_loss, train_acc = self.train_epoch_(self.train_dataloader, training=True)
            self.loop.close()

            self.loop = tqdm(total=len(self.val_dataloader), position=0, leave=False)
            # Validate
            print(f"\nValidation for epoch {i} of {num_epochs}")
            val_loss, val_acc = self.train_epoch_(self.val_dataloader, training=False)
            self.loop.close()
            

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)

        return  train_losses, val_losses, train_accuracies, val_accuracies

    def train_epoch_(self, data_loader, training=True):
        if training:
            self.network.train() 
            torch.set_grad_enabled(True)
        else:
            self.network.eval()
            torch.set_grad_enabled(False)

        loss_sum = 0
        acc_sum = 0
        for i, (imgs, labels) in enumerate(data_loader):
            imgs, labels = imgs.to(self.device, non_blocking=True), labels.to(self.device,non_blocking=True) # non_blocking is a speed up (async)
            
            if training:
                self.optimizer.zero_grad() # Set gradient to zero

            y_hat = self.network(imgs)
            loss = self.loss_func(y_hat, labels.long())
            loss_sum += loss.detach().sum().item()

            
            probs = y_hat.argmax(1)
            accuracy = (probs == labels).float().mean()
            acc_sum += accuracy.detach().item()

            mem_allocated = torch.cuda.memory_allocated(0) / 1e9

            self.loop.set_description('loss: {:.4f}, accuracy: {:.4f}, mem: {:.2f}'.format(loss.detach().sum().item(), accuracy, mem_allocated))
            self.loop.update(1)

            if training:
                loss.backward() # Compute gradient, for weight with respect to loss
                self.optimizer.step() # Take step in the direction of the negative gradient

        return loss_sum/i, acc_sum/i