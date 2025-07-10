import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error
from model import HeadPoseMobileNetV2
from dataloader import HeadPoseDataset, X_train, Y_train, X_test_1, Y_test_1, X_test_2, Y_test_2
import os
from tqdm import tqdm

# === Hyperparameters ===
batch_size = 16
epochs = 300
start_decay_epoch = [30, 60]
learning_rate = 1e-3
save_freq = 5
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# === Data ===
train_dataset = HeadPoseDataset(X_train, Y_train, augment=True)
val_dataset = HeadPoseDataset(X_test_2, Y_test_2, augment=False)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# === Model, Loss, Optimizer ===
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = HeadPoseMobileNetV2(alpha=0.6).to(device)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# === Learning Rate Scheduler ===
class ManualDecayLR:
    def __init__(self, optimizer, decay_epochs):
        self.optimizer = optimizer
        self.decay_epochs = decay_epochs

    def step(self, epoch):
        if epoch in self.decay_epochs:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 0.1

lr_scheduler = ManualDecayLR(optimizer, decay_epochs=start_decay_epoch)

# === Training ===
def evaluate(model, X_test, Y_test):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_test).permute(0, 3, 1, 2).float() / 255.0
        X_tensor = (X_tensor - 0.5) / 0.5  # match Normalize
        X_tensor = X_tensor.to(device)
        preds = model(X_tensor).cpu().numpy()
    return preds

def benchmark(model):
    print("\nBenchmarking:")
    for name, (X_test, Y_test) in zip(["BIWI", "AFLW"], [(X_test_1, Y_test_1), (X_test_2, Y_test_2)]):
        preds = evaluate(model, X_test, Y_test)
        yaw, pitch, roll = Y_test[:, 0], Y_test[:, 1], Y_test[:, 2]
        pred_yaw, pred_pitch, pred_roll = preds[:, 0], preds[:, 1], preds[:, 2]

        yaw_mae = mean_absolute_error(yaw, pred_yaw)
        pitch_mae = mean_absolute_error(pitch, pred_pitch)
        roll_mae = mean_absolute_error(roll, pred_roll)

        print(f'{name}')
        print(f'YAW: {yaw_mae:.3f}\nPITCH: {pitch_mae:.3f}\nROLL: {roll_mae:.3f}')
        print(f'MAE: {(yaw_mae + pitch_mae + roll_mae)/3:.3f}\n')

best_val_loss = float('inf')
patience = 5
no_improve_epochs = 0

for epoch in range(epochs):
    model.train()
    total_loss = 0
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Train", leave=False)

    for imgs, targets in train_bar:
        imgs, targets = imgs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)

        train_bar.set_postfix(batch_loss=loss.item())

    avg_loss = total_loss / len(train_loader.dataset)
    print(f"[{epoch+1:03d}] Train MAE: {avg_loss:.4f}", end='  ')

    # === Validation ===
    model.eval()
    val_loss = 0
    val_bar = tqdm(val_loader, desc=f"Val", leave=False)
    with torch.no_grad():
        for imgs, targets in val_bar:
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * imgs.size(0)
            val_bar.set_postfix(batch_loss=loss.item())

    val_loss /= len(val_loader.dataset)
    print(f"Val MAE (AFLW): {val_loss:.4f}")

    # === Save Checkpoints ===
    if (epoch + 1) % save_freq == 0:
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"epoch_{epoch+1:03d}.pt"))

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve_epochs = 0
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"best.pt"))
    else:
        no_improve_epochs += 1
        if no_improve_epochs >= patience:
            print(f"Early stopping at epoch {epoch+1} due to no improvement in {patience} epochs.")
            break

    lr_scheduler.step(epoch)

# === Final Benchmark ===
model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "best.pt")))
benchmark(model)
