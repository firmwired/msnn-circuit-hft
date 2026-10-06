==============================
FILE: notebooks\notebooke8dae9d3db (1).ipynb
==============================
--- CELL 0 ---
%pip install snntorch
import torch
import torch.nn as nn
import glob
from torch.nn.functional import cross_entropy
import torch.nn.functional as F
import snntorch as snn
from snntorch import spikegen
from snntorch import surrogate
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import gc
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


--- CELL 1 ---

class LOBDayDataset(Dataset):
    """Loads a single day file from the FI-2010 dataset without cross-day leakage"""
    def __init__(self, filepath, window_size=30, stride=5, delta_threshold=0.005):
        
        data = np.loadtxt(filepath)
        features = data[:144, :].T  # features
        labels = data[146, :].T - 1 # labels (we map them from {1, 2, 3...} to {0, 1, 2, 3.....})
        del data

        features_tensor = torch.from_numpy(features).float()
        labels_tensor = torch.from_numpy(labels).long()
        del features, labels

        # spike encoding using delta modulation see https://snntorch.readthedocs.io/en/latest/tutorials/legacy/tutorial_1_old.html#delta-modulation
        spike = spikegen.delta(features_tensor, threshold=delta_threshold, padding=True, off_spike=True)
        spikes_pos = torch.where(spike > 0, spike, torch.zeros_like(spike))
        spikes_neg = torch.where(spike < 0, torch.abs(spike), torch.zeros_like(spike))
        del spike, features_tensor

        # inputs <= spike_pos + spike_neg
        inputs = torch.cat((spikes_pos, spikes_neg), dim=1).half() 
        del spikes_pos, spikes_neg

        
        
        # we slice the inputs into sequence windows 
        self.X, self.Y = self._create_sequences(inputs, labels_tensor, window_size, stride)
        del inputs, labels_tensor
        gc.collect()
    
    # create sequence of the inputs based on window_size and stride
    def _create_sequences(self, inputs, labels, window_size, stride):
        X, Y = [], []
        for i in range(0, len(inputs) - window_size, stride):
            X.append(inputs[i : i + window_size])
            Y.append(labels[i + window_size - 1])
        return torch.stack(X), torch.stack(Y)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx].float(), self.Y[idx]

# this function implements the paper exact anchred cross-validation protocol
# example: for a fold of 2 (fold_idx = 2)
# we train the MSNN on day 1 + day 2 and test the MSNN on day 3 data  
def get_anchored_fold_loaders(fold_idx, train_files, test_files, batch_size=256):
    print(f"\n***  fold == {fold_idx + 1} ***")
    
    # identify the days for the training 
    active_train_files = train_files[:fold_idx + 1]
    # The test target is always the next sequential day window
    active_test_file = test_files[fold_idx + 1]
    
    print(f"training files are up to: {active_train_files[-1].split('/')[-1]}")
    print(f"testing file: {active_test_file.split('/')[-1]}")
    
    train_datasets = [LOBDayDataset(f) for f in active_train_files]
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)

    test_dataset = LOBDayDataset(active_test_file)
    
    
    
    # set shuffle=False to preserve internal temporal sequence blocks
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    return train_loader, test_loader, train_dataset

--- CELL 2 ---

def weights(device, labels, smoothing=.15):
    class_counts = torch.bincount(labels.long())
    total_samples = len(labels)
    
    smoothed_c = class_counts.float() + (smoothing * total_samples / len(class_counts))
    dynamic_weights = total_samples / (len(class_counts) * smoothed_c)
    print(dynamic_weights)
    return dynamic_weights.to(device=device)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # weights
        self.gamma = gamma 
        self.reduction = reduction
    
    def forward(self, logits, targets):
        ce_loss = cross_entropy(logits, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma  
        loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        return loss.sum()

class STEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, num_levels):
        if num_levels is None or num_levels <= 1:
            return x
        scaled = x * (num_levels - 1)
        quantized = torch.round(scaled)
        return quantized / (num_levels - 1)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

# Hardware-aware RRAM crossbar with: 
# Differential conductance mapping 
# Stuck at fauls (SAF)
# Cycle-to-cyle variation
# Device-to-Device variation
# Retention Drift    
class MemristorCrossbar(nn.Module):
    def __init__(self, in_features, out_features, num_levels=16, saf_rate=0.03, noise_std=0.05, 
                d2d_std=0.05,
                drift_rate=0.002,
                force_ideal=False,
                force_noise_eval=False
                ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        if force_ideal:
            num_levels = 0
            saf_rate = 0.0
            noise_std = 0.0
            d2d_std = 0.0
            drift_rate = 0.0
            print(f"params memristor: {num_levels }, {saf_rate}, {noise_std}, {d2d_std}, {drift_rate}")
            
        self.num_levels = num_levels
        self.noise_std = noise_std
        self.d2d_std = d2d_std
        self.force_ideal = force_ideal
        self.drift_rate = drift_rate
        self.force_noise_eval = force_noise_eval
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        nn.init.normal_(self.weight, mean=0.0, std=0.05)
        nn.init.zeros_(self.bias)

      
        self.register_buffer("saf_pos_lrs", torch.rand(out_features, in_features) < (saf_rate / 2))
        self.register_buffer("saf_pos_hrs", torch.rand(out_features, in_features) < (saf_rate / 2))
        self.register_buffer("saf_neg_lrs", torch.rand(out_features, in_features) < (saf_rate / 2))
        self.register_buffer("saf_neg_hrs",torch.rand(out_features, in_features) < (saf_rate / 2))

       # Device-to-Device variation D2D
        self.register_buffer("d2d_pos", torch.clamp( 1.0 + torch.randn(out_features, in_features) * d2d_std, 0.7, 1.3))
        self.register_buffer("d2d_neg", torch.clamp( 1.0 + torch.randn(out_features, in_features) * d2d_std, 0.7, 1.3))

    def get_hardware_weights(self):

        # weights normalized
        weight_scale = torch.max(torch.abs(self.weight)).detach().clamp(min=1e-5)
        w_norm = torch.clamp(self.weight / weight_scale, -1.0, 1.0)
        g_pos = (w_norm + 1.0) / 2.0
        g_neg = (1.0 - w_norm) / 2.0
        
        g_pos *= self.d2d_pos
        g_neg *= self.d2d_neg
        
        g_pos = torch.clamp(g_pos, 0.0, 1.0)
        g_neg = torch.clamp(g_neg, 0.0, 1.0)

        # stuch at faults (SAF)
        g_pos = torch.where(self.saf_pos_lrs, torch.ones_like(g_pos), g_pos)
        g_pos = torch.where(self.saf_pos_hrs, torch.zeros_like(g_pos), g_pos)
        g_neg = torch.where(self.saf_neg_lrs, torch.ones_like(g_neg), g_neg)
        g_neg = torch.where(self.saf_neg_hrs, torch.zeros_like(g_neg), g_neg)
        # qunatization
        g_pos_q = STEQuantize.apply(g_pos, self.num_levels)
        g_neg_q = STEQuantize.apply(g_neg, self.num_levels)
       
        #C2C cycle to cycle
        if (self.training or self.force_noise_eval) and self.noise_std > 0:
            g_pos_q = torch.clamp(g_pos_q + torch.randn_like(g_pos_q) * self.noise_std, 0.0, 1.0)
            g_neg_q = torch.clamp(g_neg_q + torch.randn_like(g_neg_q) * self.noise_std, 0.0, 1.0)

        # retention drift
        if not self.training:
            drift = torch.exp(torch.full_like(g_pos_q, -self.drift_rate))
            g_pos_q *= drift
            g_neg_q *= drift
        w_hardware = (g_pos_q - g_neg_q) * weight_scale
        
        return w_hardware

    def forward(self, x):
        w_hardware = self.get_hardware_weights()
        return F.linear(x, w_hardware, self.bias)
    
    
class MSNN(nn.Module):
    """hardware-constrained MSNN and tracking spike metrics"""
    def __init__(self, num_inputs=288, num_hidden=128, num_outputs=3, beta=0.9, num_levels=16, saf_rate=0.03, noise_std=0.05, force_ideal=False, force_noise_eval=False):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=25)
        
        self.fc1 = MemristorCrossbar(num_inputs, num_hidden, num_levels, saf_rate, noise_std, force_ideal=force_ideal, force_noise_eval=force_noise_eval)
        self.lif1 = snn.Leaky(threshold=0.5, beta=beta, spike_grad=spike_grad, reset_mechanism="subtract")
        self.drop = nn.Dropout(0.3) 
        
        self.fc2 = MemristorCrossbar(num_hidden, num_outputs, num_levels, saf_rate, noise_std, force_ideal=force_ideal, force_noise_eval=force_noise_eval)
        self.lif2 = snn.Leaky(threshold=1.0, beta=beta, spike_grad=spike_grad, reset_mechanism="subtract")
        
        # keep track of spike counts
        self.spike_counts = {"layer1": 0.0, "layer2": 0.0, "total_steps": 0}

    def reset_spike_metrics(self):
        self.spike_counts = {"layer1": 0.0, "layer2": 0.0, "total_steps": 0}

    def forward(self, x):
        # adjust the input shame from (batch, time, featurres) ->> (time, batch, features)
        x = x.permute(1, 0, 2)
        batch_size = x.shape[1]
        time_steps = x.shape[0]
        
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        # input current is calculated in a single vectorized operation
        cur1_all = self.fc1(x)
        
        # Cache Layer 2 hardware matrix weights for this specific forward timeline execution block
        w_phys2 = self.fc2.get_hardware_weights()
        bias2 = self.fc2.bias
        
        for step in range(time_steps):
            spk1, mem1 = self.lif1(cur1_all[step], mem1)
            spk1_dropped = self.drop(spk1)
            
            # record layer 1 spike track
            if not self.training:
                self.spike_counts["layer1"] += spk1.detach().sum().item()
            
            cur2 = F.linear(spk1_dropped, w_phys2, bias2)
            spk2, mem2 = self.lif2(cur2, mem2)
            
            # record layer 2 spike track
            if not self.training:
                self.spike_counts["layer2"] += spk2.detach().sum().item()
                
        if not self.training:
            self.spike_counts["total_steps"] += (batch_size * time_steps)
            
        return mem2



--- CELL 3 ---
def compute_hardware_metrics(data_loader, net, device, window_size=30):
    """evaluates the model and computes hardware metrics etc...
        window_size is a must, unless if its 30
    """
    net.eval()
    net.reset_spike_metrics()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, targets in data_loader:
            data = data.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).long()
            logits = net(data)
            preds = logits.argmax(dim=1)
            
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
    # calculates sparsity metrics
    total_samples_processed = net.spike_counts["total_steps"] / window_size  
    total_neuron_steps_l1 = net.spike_counts["total_steps"] * net.fc1.out_features
    total_neuron_steps_l2 = net.spike_counts["total_steps"] * net.fc2.out_features
    
    l1_sparsity = net.spike_counts["layer1"] / max(1, total_neuron_steps_l1)
    l2_sparsity = net.spike_counts["layer2"] / max(1, total_neuron_steps_l2)
    
    accuracy = correct / total
    
    print("\n*** extracted hardware metrics ***")
    print(f"accuracy: {accuracy*100:.2f}%")
    print(f"layer 1 firing density : {l1_sparsity*100:.3f}% spikes/neuron/step")
    print(f"layer 2 firing density : {l2_sparsity*100:.3f}% spikes/neuron/step")
    print(f"total spikes emitted per sequence window ({window_size}): {(net.spike_counts['layer1'] + net.spike_counts['layer2']) / max(1, total_samples_processed):.2f}")
    
    return {
        "accuracy": accuracy,
        "l1_sparsity": l1_sparsity,
        "l2_sparsity": l2_sparsity,
    }


--- CELL 4 ---
import torch
import numpy as np
import os

num_epochs = 20  #significantly lower as anchored dataset scale up easily
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

# update with ur own paths 
train_files = sorted(glob.glob(
    "/kaggle/input/datasets/firmwired/fi-2010/data/published/BenchmarkDatasets/BenchmarkDatasets/BenchmarkDatasets/NoAuction/NoAuction_MinMax/NoAuction_MinMax_Training/Train_Dst_NoAuction_MinMax_CF_*.txt"
))
test_files = sorted(glob.glob(
    "/kaggle/input/datasets/firmwired/fi-2010/data/published/BenchmarkDatasets/BenchmarkDatasets/BenchmarkDatasets/NoAuction/NoAuction_MinMax/NoAuction_MinMax_Testing/Test_Dst_NoAuction_MinMax_CF_*.txt"
))

output_dir = '/kaggle/working/MSNN/noisy'
os.makedirs(output_dir, exist_ok=True)
torch.manual_seed(123)
fold_summary_metrics = []

net = MSNN(
    num_inputs=288, 
    num_hidden=128, 
    num_outputs=3, 
    beta=0.9, 
    num_levels=16, 
    saf_rate=0.03, 
    noise_std=0.05,
    force_ideal=False,
    force_noise_eval=True
).to(device)


# loop across all 9 files
for fold_idx in range(8):
    print("\n" + "*"*70)
    print(f"* anchord fold  {fold_idx + 1} / 9 *")
    print("*"*70)
    
    # get the needed fold time (based on fold_idx)
    train_loader, test_loader, fold_train_dataset = get_anchored_fold_loaders(
        fold_idx=fold_idx, 
        train_files=train_files, 
        test_files=test_files, 
        batch_size=256
    )
    
    if isinstance(fold_train_dataset, torch.utils.data.ConcatDataset):
        active_labels = torch.cat([ds.Y for ds in fold_train_dataset.datasets])
    else:
        active_labels = fold_train_dataset.Y
        
    w = weights(device=device, labels=active_labels)
    loss_fn = FocalLoss(alpha=w, gamma=2)
    
   
    num_epochs_per_fold = 10  
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs_per_fold, eta_min=1e-6)
    
    best_fold_acc = 0.0
    patience_counter = 0
    patience = 4
    
    # training loop
    for epoch in range(num_epochs_per_fold):
        net.train()
        epoch_loss = 0.0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).long()
            
            optimizer.zero_grad(set_to_none=True)
            logits = net(data)
            loss_val = loss_fn(logits, targets)
            
            loss_val.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss_val.item()
            
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        
        net.eval()
        with torch.no_grad():
            val_correct = 0
            val_total = 0
            for val_data, val_targets in test_loader:
                val_data = val_data.to(device, non_blocking=True)
                val_targets = val_targets.to(device, non_blocking=True).long()
                
                val_preds = net(val_data).argmax(dim=1)
                val_correct += (val_preds == val_targets).sum().item()
                val_total += val_targets.size(0)
            current_val_acc = val_correct / max(1, val_total)
            
        print(f"Fold {fold_idx+1} | Epoch [{epoch+1}/{num_epochs_per_fold}] | Loss: {avg_loss:.4f} | Current Test Acc: {current_val_acc*100:.2f}%")
        
        if current_val_acc > best_fold_acc:
            best_fold_acc = current_val_acc
            patience_counter = 0
            torch.save(net.state_dict(), os.path.join(output_dir, f'model_{fold_idx+1}.pt'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("↳ Early stopping fold fine-tuning profile.")
                break
                
    # 4. Extract Final Hardware Profiling numbers for this fold's configuration
    print(f"\n--- Compiling Hardware Metrics for Fold {fold_idx + 1} ---") 
    checkpoint_file = os.path.join(output_dir, f'model_{fold_idx+1}.pt')
    net.load_state_dict(torch.load(checkpoint_file, map_location=device, weights_only=True))
    
    total_input_sparsity = 0.0
    num_batches = 0
    with torch.no_grad():
        for val_data, _ in test_loader:
            val_data = val_data.to(device, non_blocking=True)
            batch_density = val_data.float().mean().item()             
            total_input_sparsity += batch_density
            num_batches += 1
            
    fold_input_sparsity = total_input_sparsity / num_batches
    
    fold_profile = compute_hardware_metrics(test_loader, net, device)
    fold_summary_metrics.append({
        "fold": fold_idx + 1,
        "accuracy": fold_profile["accuracy"],
        "l1_sparsity": fold_profile["l1_sparsity"],
        "l2_sparsity": fold_profile["l2_sparsity"],
        "input_sparsity": fold_input_sparsity
    })

# 5. Output Summary Results Block for Paper Generation
print("\n" + "="*60)
print("final report:::")
print("="*60)
print(f"{'Fold':<6} | {'Test Accuracy':<15} | {'L1 Spike Density':<18} | {'L2 Spike Density':<18}")
print("-"*60)
for entry in fold_summary_metrics:
  print(f"{entry['fold']:<6} | "
          f"{entry['accuracy']*100:<14.2f}% | "
          f"{entry['input_sparsity']*100:<14.4f}% | "
          f"{entry['l1_sparsity']*100:<17.4f}% | "
          f"{entry['l2_sparsity']*100:<17.4f}%")
print("-"*60)

--- CELL 5 ---
output_dir_ideal = '/kaggle/working/MSNN/ideal'
os.makedirs(output_dir_ideal, exist_ok=True)

torch.manual_seed(123)
net_ideal = MSNN(
    num_inputs=288, 
    num_hidden=128, 
    num_outputs=3, 
    beta=0.9, 
    force_ideal=True
).to(device)
# net_ideal.force_noise_eval = True
fold_summary_metrics_ideal = []
# loop across all 9 files
for fold_idx in range(8):
    print("\n" + "*"*70)
    print(f"* anchord fold  {fold_idx + 1} / 9 *")
    print("*"*70)
    
    # get the needed fold time (based on fold_idx)
    train_loader, test_loader, fold_train_dataset = get_anchored_fold_loaders(
        fold_idx=fold_idx, 
        train_files=train_files, 
        test_files=test_files, 
        batch_size=256
    )
    
    if isinstance(fold_train_dataset, torch.utils.data.ConcatDataset):
        active_labels = torch.cat([ds.Y for ds in fold_train_dataset.datasets])
    else:
        active_labels = fold_train_dataset.Y
        
    w = weights(device=device, labels=active_labels)
    loss_fn = FocalLoss(alpha=w, gamma=2)
    
   
    num_epochs_per_fold = 10  
    optimizer = torch.optim.Adam(net_ideal.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs_per_fold, eta_min=1e-6)
    
    best_fold_acc = 0.0
    patience_counter = 0
    patience = 4
    
    # training loop
    for epoch in range(num_epochs_per_fold):
        net_ideal.train()
        epoch_loss = 0.0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).long()
            
            optimizer.zero_grad(set_to_none=True)
            logits = net_ideal(data)
            loss_val = loss_fn(logits, targets)
            
            loss_val.backward()
            torch.nn.utils.clip_grad_norm_(net_ideal.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss_val.item()
            
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        
        net_ideal.eval()
        with torch.no_grad():
            val_correct = 0
            val_total = 0
            for val_data, val_targets in test_loader:
                val_data = val_data.to(device, non_blocking=True)
                val_targets = val_targets.to(device, non_blocking=True).long()
                
                val_preds = net_ideal(val_data).argmax(dim=1)
                val_correct += (val_preds == val_targets).sum().item()
                val_total += val_targets.size(0)
            current_val_acc = val_correct / max(1, val_total)
            
        print(f"Fold {fold_idx+1} | Epoch [{epoch+1}/{num_epochs_per_fold}] | Loss: {avg_loss:.4f} | Current Test Acc: {current_val_acc*100:.2f}%")
        
        if current_val_acc > best_fold_acc:
            best_fold_acc = current_val_acc
            patience_counter = 0
            torch.save(net.state_dict(), os.path.join(output_dir_ideal, f'model_{fold_idx+1}.pt'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("↳ Early stopping fold fine-tuning profile.")
                break
                
    # 4. Extract Final Hardware Profiling numbers for this fold's configuration
    print(f"\n--- Compiling Hardware Metrics for Fold {fold_idx + 1} ---")
    checkpoint_file = os.path.join(output_dir_ideal, f'model_{fold_idx+1}.pt')
    net.load_state_dict(torch.load(checkpoint_file, map_location=device, weights_only=True))
    
    total_input_sparsity_ideal = 0.0
    num_batches_ideal = 0
    with torch.no_grad():
        for val_data, _ in test_loader:
            val_data = val_data.to(device, non_blocking=True)
            
            # NOTE: If val_data is already binary spikes, this works as-is. 
            # If your MSNN encodes data into spikes internally, apply that 
            # specific encoding step to val_data here first.
            batch_density_ideal = val_data.float().mean().item() 
            
            total_input_sparsity_ideal += batch_density_ideal
            num_batches_ideal += 1
            
    fold_input_sparsity_ideal = total_input_sparsity_ideal / num_batches_ideal
    
    
    fold_profile_ideal = compute_hardware_metrics(test_loader, net_ideal, device)
    fold_summary_metrics_ideal.append({
        "fold": fold_idx + 1,
        "accuracy": fold_profile_ideal["accuracy"],
        "l1_sparsity": fold_profile_ideal["l1_sparsity"],
        "l2_sparsity": fold_profile_ideal["l2_sparsity"],
        "input_sparsity": fold_input_sparsity_ideal
    })

# 5. Output Summary Results Block for Paper Generation
print("\n" + "="*60)
print("final report for ideal hardware:::")
print("="*60)
print(f"{'Fold':<6} | {'Test Accuracy':<15} | {'L1 Spike Density':<18} | {'L2 Spike Density':<18}")
print("-"*60)
for entry in fold_summary_metrics_ideal:
  print(f"{entry['fold']:<6} | "
          f"{entry['accuracy']*100:<14.2f}% | "
          f"{entry['input_sparsity']*100:<14.4f}% | "
          f"{entry['l1_sparsity']*100:<17.4f}% | "
          f"{entry['l2_sparsity']*100:<17.4f}%")
print("-"*60)

--- CELL 6 ---
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

folds = np.arange(1, 9)
test_accuracy_ideal = np.array([item["accuracy"] for item in fold_summary_metrics_ideal])
l1_density_ideal = np.array([item["l1_sparsity"] for item in fold_summary_metrics_ideal])
l2_density_ideal = np.array([item["l2_sparsity"] for item in fold_summary_metrics_ideal])

# Globally force classic MATLAB inward ticks and formatting
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"]  = ["Times New Roman", "DejaVu Serif", "Computer Modern Roman"]

# Adjusted figure size slightly to make clean room for the bottom legend
fig, ax1 = plt.subplots(figsize=(8.5, 5.8), dpi=600) 

# Configure Primary Axis (Left)
ax1.set_xlabel('Folds (Test Trading Days)', fontweight='bold', labelpad=10)
ax1.set_ylabel('Test Accuracy', color="#005b96", fontweight='bold', labelpad=10)
line1 = ax1.plot(folds, test_accuracy_ideal, color="#005b96", marker='o', linewidth=2.5, markersize=7, label='Test Accuracy')

# Dynamic vertical calculation for accuracy bounds
ymin = min(test_accuracy_ideal) - 0.03
ymax = max(test_accuracy_ideal) + 0.03
ax1.set_ylim(min(0.40, ymin), max(0.65, ymax)) 

ax1.tick_params(axis='y', labelcolor="#005b96", length=6)
ax1.tick_params(axis='x', length=6)
ax1.set_xticks(folds) 

# MATLAB-style grid layout
ax1.grid(True, linestyle=':', color='gray', alpha=0.5, linewidth=1.0)

# Configure Secondary Axis (Right)
ax2 = ax1.twinx()
ax2.set_ylabel('Spike Activation Density (%) [Log Scale]', color='black', fontweight='bold', labelpad=12)

line2 = ax2.plot(folds, l1_density_ideal, color="#d95f02", marker='s', linestyle='--', linewidth=2, markersize=7, label='Layer 1 (Hidden)')
line3 = ax2.plot(folds, l2_density_ideal, color="#1b9e77", marker='^', linestyle='-.', linewidth=2, markersize=7, label='Layer 2 (Output)')

# Apply log scale and handle ticks
ax2.set_yscale('log')

# --- FIX: Set data limits and manually override the LogLocator ticks ---
all_densities = np.concatenate([l1_density_ideal, l2_density_ideal])
ax2.set_ylim(min(all_densities) * 0.9, max(all_densities) * 1.1)

# Tell the log scale to show ticks inside the fractional range (e.g., 0.2, 0.3, 0.4, 0.6, 0.8)
ax2.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)))

# Clean format string to guarantee they print cleanly on the right axis border
ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
ax2.yaxis.set_minor_formatter(ticker.NullFormatter()) 

# Explicitly make sure right axis ticks point inside
ax2.tick_params(axis='y', which='both', labelcolor='black', length=6, direction='in')

# Fix the broken top bounding box line caused by twinx
ax1.set_zorder(ax2.get_zorder() + 1)  
ax1.set_frame_on(False)               
ax2.set_frame_on(True)                

# Unified Legend - Positioned safely OUTSIDE below the X-axis
lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=3, 
           frameon=True, facecolor='white', edgecolor='black', framealpha=1.0, borderpad=0.8)

plt.title("Hardware performance under Ideal conditions", fontweight='bold', pad=15)

fig.tight_layout(pad=1.5)
plt.savefig("evaluation_plot_ideal.pdf", format='pdf', bbox_inches='tight')
plt.show()


--- CELL 7 ---
print(fold_summary_metrics_ideal)
print(l1_density_ideal)
print(l2_density_ideal)
print(l1_density_ideal.min(), l1_density_ideal.max())
print(l2_density_ideal.min(), l2_density_ideal.max())

--- CELL 8 ---
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Sample arrays matching your code structure
folds = np.arange(1, 9)
accuracy = np.array([item["accuracy"] for item in fold_summary_metrics])
l1_density = np.array([item["l1_sparsity"] for item in fold_summary_metrics])
l2_density = np.array([item["l2_sparsity"] for item in fold_summary_metrics])

# Globally force classic MATLAB inward ticks and formatting
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"]  = ["Times New Roman", "DejaVu Serif", "Computer Modern Roman"]

# Adjusted figure size slightly to make clean room for the bottom legend
fig, ax1 = plt.subplots(figsize=(8.5, 5.8), dpi=600) 

# Configure Primary Axis (Left)
ax1.set_xlabel('Folds (Test Trading Days)', fontweight='bold', labelpad=10)
ax1.set_ylabel('Test Accuracy', color="#005b96", fontweight='bold', labelpad=10)
line1 = ax1.plot(folds, accuracy, color="#005b96", marker='o', linewidth=2.5, markersize=7, label='Test Accuracy')

# --- FIX 1: Dynamic vertical bounds to prevent lines cutting through the top frame ---
ymin = min(accuracy) - 0.03
ymax = max(accuracy) + 0.03
ax1.set_ylim(min(0.40, ymin), max(0.65, ymax)) 

ax1.tick_params(axis='y', labelcolor="#005b96", length=6)
ax1.tick_params(axis='x', length=6)
ax1.set_xticks(folds) # Explicitly show every fold integer

# MATLAB-style grid layout
ax1.grid(True, linestyle=':', color='gray', alpha=0.5, linewidth=1.0)

# Configure Secondary Axis (Right)
ax2 = ax1.twinx()
ax2.set_ylabel('Spike Activation Density (%) [Log Scale]', color='black', fontweight='bold', labelpad=12)

line2 = ax2.plot(folds, l1_density, color="#d95f02", marker='s', linestyle='--', linewidth=2, markersize=7, label='Layer 1 (Hidden)')
line3 = ax2.plot(folds, l2_density, color="#1b9e77", marker='^', linestyle='-.', linewidth=2, markersize=7, label='Layer 2 (Output)')

# Apply log scale
ax2.set_yscale('log')

# --- FIX 2: Set precise data limits and force log sub-interval ticks to display values ---
all_densities = np.concatenate([l1_density, l2_density])
ax2.set_ylim(min(all_densities) * 0.9, max(all_densities) * 1.1)

# Places ticks at fractional intervals within a single decade to guarantee visibility
ax2.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)))

# Clean MATLAB scalar format (e.g., 0.25 instead of 2.5 x 10^-1)
ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
ax2.yaxis.set_minor_formatter(ticker.NullFormatter()) 

# Apply tick styling to major and minor ticks on the right axis border
ax2.tick_params(axis='y', which='both', labelcolor='black', length=6, direction='in')

# Fix the broken top bounding box line caused by twinx
ax1.set_zorder(ax2.get_zorder() + 1)  
ax1.set_frame_on(False)               
ax2.set_frame_on(True)                

# Unified Legend - Positioned safely OUTSIDE below the X-axis
lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]

# bbox_to_anchor points to (X, Y) coordinates relative to axes. (0.5, -0.18) drops it right below the center point.
ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=3, 
           frameon=True, facecolor='white', edgecolor='black', framealpha=1.0, borderpad=0.8)

# Title for the SAF dataset
plt.title("Hardware performance under ≈ 3% SAF", fontweight='bold', pad=15)

# Use tight_layout with specified padding to ensure everything fits inside the PDF boundary
fig.tight_layout(pad=1.5)

# bbox_inches='tight' is critical here to ensure the outside legend isn't cropped during export
plt.savefig("evaluation_plot.pdf", format='pdf', bbox_inches='tight')
plt.show()


--- CELL 9 ---
print(fold_summary_metrics[-1])

--- CELL 10 ---
# LSTM architecture to get hte metrics needed to benchmark against our MSNN
class LSTM_net(nn.Module): 
    def __init__(self, input_size=288, hidden_size=128, num_classes=3):
        super(LSTM_net, self).__init__()
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True) 
        self.drop = nn.Dropout(.3)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        
        final_step_out = out[:, -1, :]
        final_step_out = self.drop(final_step_out)
        logits = self.fc(final_step_out)
        
        return logits
    
print(f"starting LSTM training on {device}")
torch.manual_seed(123)
lstm_fold_summary = []

lstm_net = LSTM_net(input_size=288, hidden_size=128, num_classes=3).to(device)

for fold_idx in range(8):
    print("*"*50 +  f"\n LSTM fold n° {fold_idx + 1}")
    
    train_loader, test_loader, fold_train_dataset = get_anchored_fold_loaders(
        fold_idx=fold_idx, 
        train_files=train_files, 
        test_files=test_files, 
        batch_size=256
    )
    
    if isinstance(fold_train_dataset, torch.utils.data.ConcatDataset):
        active_labels = torch.cat([ds.Y for ds in fold_train_dataset.datasets])
    else:
        active_labels = fold_train_dataset.Y
        
    if isinstance(fold_train_dataset, torch.utils.data.ConcatDataset):
        active_labels = torch.cat([ds.Y for ds in fold_train_dataset.datasets])
    else:
        active_labels = fold_train_dataset.Y
        
    w = weights(device=device, labels=active_labels)
    loss_fn = FocalLoss(alpha=w, gamma=2)
    
    num_epochs_per_fold = 10  
    optimizer = torch.optim.Adam(lstm_net.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs_per_fold, eta_min=1e-6)
    
    best_fold_acc = 0.0
    patience_counter = 0
    patience = 4
    
    for epoch in range(num_epochs_per_fold):
        lstm_net.train()
        epoch_loss = 0.0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).long()
            
            optimizer.zero_grad(set_to_none=True)
            logits = lstm_net(data)
            loss_val = loss_fn(logits, targets)
            
            loss_val.backward()
            torch.nn.utils.clip_grad_norm_(lstm_net.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss_val.item()
            
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        
        # Evaluate 
        lstm_net.eval()
        with torch.no_grad():
            val_correct = 0
            val_total = 0
            for val_data, val_targets in test_loader:
                val_data = val_data.to(device, non_blocking=True)
                val_targets = val_targets.to(device, non_blocking=True).long()
                val_preds = lstm_net(val_data).argmax(dim=1)
                val_correct += (val_preds == val_targets).sum().item()
                val_total += val_targets.size(0)
            current_val_acc = val_correct / max(1, val_total)
            
        print(f"Fold {fold_idx+1} | Epoch [{epoch+1}/{num_epochs_per_fold}] | Loss: {avg_loss:.4f} | LSTM Test Acc: {current_val_acc*100:.2f}%")
        
        if current_val_acc > best_fold_acc:
            best_fold_acc = current_val_acc
            patience_counter = 0
            torch.save(lstm_net.state_dict(), f'best_lstm_fold_{fold_idx+1}.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("↳ Early stopping LSTM fold fine-tuning.")
                break

    print(f"\n--- Saving LSTM Metrics for Fold {fold_idx + 1} ---")
    lstm_fold_summary.append({
        "fold": fold_idx + 1,
        "accuracy": best_fold_acc
    })

# ---------------------------------------------------------
# 3. Final LSTM vs MSNN Summary Report
# ---------------------------------------------------------
print("\n" + "="*50)
print("FINAL LSTM BASELINE REPORT")
print("="*50)
print(f"{'Fold':<6} | {'LSTM Test Accuracy':<20}")
print("-"*50)
lstm_total_acc = 0
for entry in lstm_fold_summary:
    print(f"{entry['fold']:<6} | {entry['accuracy']*100:<19.2f}%")
    lstm_total_acc += entry['accuracy']

print("="*50)
print(f"LSTM 8-Fold Average Accuracy: {(lstm_total_acc / 8)*100:.2f}%")
    

--- CELL 11 ---
import os, json

def run_lstm_experiment(seed, train_files, test_files, num_folds=8, num_epochs_per_fold=10, patience=4,
                         checkpoint_dir='/kaggle/working/LSTM/sweep'):
    """
    Runs the full anchored-fold LSTM training+eval pipeline once, for one seed.
    Returns a list of per-fold {"fold": i, "accuracy": acc} dicts.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.manual_seed(seed)

    lstm_net = LSTM_net(input_size=288, hidden_size=128, num_classes=3).to(device)
    fold_summary = []

    for fold_idx in range(num_folds):
        print(f"\n***** LSTM seed {seed} — fold {fold_idx + 1} *****")

        train_loader, test_loader, fold_train_dataset = get_anchored_fold_loaders(
            fold_idx=fold_idx, train_files=train_files, test_files=test_files, batch_size=256
        )

        if isinstance(fold_train_dataset, torch.utils.data.ConcatDataset):
            active_labels = torch.cat([ds.Y for ds in fold_train_dataset.datasets])
        else:
            active_labels = fold_train_dataset.Y

        w = weights(device=device, labels=active_labels)
        loss_fn = FocalLoss(alpha=w, gamma=2)

        optimizer = torch.optim.Adam(lstm_net.parameters(), lr=3e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs_per_fold, eta_min=1e-6)

        best_fold_acc = 0.0
        patience_counter = 0
        ckpt_path = f'{checkpoint_dir}/lstm_seed{seed}_fold{fold_idx+1}.pt'

        for epoch in range(num_epochs_per_fold):
            lstm_net.train()
            epoch_loss = 0.0
            for data, targets in train_loader:
                data = data.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True).long()

                optimizer.zero_grad(set_to_none=True)
                loss_val = loss_fn(lstm_net(data), targets)
                loss_val.backward()
                torch.nn.utils.clip_grad_norm_(lstm_net.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss_val.item()
            scheduler.step()

            lstm_net.eval()
            with torch.no_grad():
                correct, total = 0, 0
                for val_data, val_targets in test_loader:
                    val_data = val_data.to(device, non_blocking=True)
                    val_targets = val_targets.to(device, non_blocking=True).long()
                    preds = lstm_net(val_data).argmax(dim=1)
                    correct += (preds == val_targets).sum().item()
                    total += val_targets.size(0)
                current_val_acc = correct / max(1, total)

            print(f"Fold {fold_idx+1} | Epoch [{epoch+1}/{num_epochs_per_fold}] | "
                  f"Loss: {epoch_loss/len(train_loader):.4f} | LSTM Test Acc: {current_val_acc*100:.2f}%")

            if current_val_acc > best_fold_acc:
                best_fold_acc = current_val_acc
                patience_counter = 0
                torch.save(lstm_net.state_dict(), ckpt_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("↳ Early stopping LSTM fold fine-tuning.")
                    break

        fold_summary.append({"fold": fold_idx + 1, "accuracy": best_fold_acc})

    return fold_summary


def run_lstm_seed_if_needed(seed, train_files, test_files, results_dir='/kaggle/working/LSTM/results'):
    """Resume-safe wrapper — skips a seed if its result was already saved."""
    os.makedirs(results_dir, exist_ok=True)
    result_path = os.path.join(results_dir, f"lstm_seed{seed}.json")

    if os.path.exists(result_path):
        print(f"Skipping LSTM seed {seed} — already completed.")
        with open(result_path) as f:
            return json.load(f)

    result = run_lstm_experiment(seed, train_files, test_files)

    with open(result_path, 'w') as f:
        json.dump(result, f)

    return result


# --- Run the sweep, same seed set as your MSNN runs ---
lstm_seeds = [123, 42, 7]
all_lstm_runs = []
for s in lstm_seeds:
    all_lstm_runs.append(run_lstm_seed_if_needed(s, train_files, test_files))


# --- Aggregate ---
import numpy as np

def aggregate_lstm(all_runs, num_folds=8):
    acc_matrix = np.array([[run[f]["accuracy"] for f in range(num_folds)] for run in all_runs])
    return acc_matrix.mean(axis=0), acc_matrix.std(axis=0)

lstm_mean, lstm_std = aggregate_lstm(all_lstm_runs)
print("LSTM accuracy (mean ± std) per fold:", lstm_mean, "±", lstm_std)
print(f"LSTM 8-fold average accuracy: {lstm_mean.mean()*100:.2f}%")

--- CELL 12 ---
def calculate_hardware_metrics(input_density=0.02, l1_density=0.0188, l2_density=0.0055, log=True):
    inputs = 288
    hidden = 128
    outputs = 3
    timesteps = 30

    # --- Corrected Hardware Parameters ---
    bits_per_device = 4
    devices_per_weight = 2  
    
    # Adjusted upward to account for parasitic line capacitance (RC delay)
    energy_per_synop_pj = 1.0       
    
    # Adjusted to reflect full column readout overhead (not just an isolated SAR step)
    adc_energy_pj_per_read = 8.0    
    
    # Added: Energy consumed by LIF neuron membrane integration per timestep
    energy_per_neuron_step_pj = 2.5 
    
    # Adjusted upward to reflect real peripheral control circuits + distribution networks
    static_power_uw = 75.0            
    latency_per_step_ns = 100

    # Topology and memory
    params_l1 = inputs * hidden
    params_l2 = hidden * outputs
    total_params = params_l1 + params_l2
    total_devices = total_params * devices_per_weight
    memory_footprint_kb = (total_devices * bits_per_device) / (8 * 1024)

    # Spike and SynOp math
    total_input_spikes = inputs * timesteps * input_density
    total_hidden_spikes = hidden * timesteps * l1_density
    synops_l1 = total_input_spikes * hidden
    synops_l2 = total_hidden_spikes * outputs
    total_synops = synops_l1 + synops_l2

    # --- Energy Components (Calculated in nJ) ---
    synop_energy_nj = (total_synops * energy_per_synop_pj) / 1000  
    
    total_adc_reads = (hidden + outputs) * timesteps
    adc_energy_nj = (total_adc_reads * adc_energy_pj_per_read) / 1000  

    # Added: Neuron integration energy component
    total_neuron_updates = (hidden + outputs) * timesteps
    neuron_energy_nj = (total_neuron_updates * energy_per_neuron_step_pj) / 1000

    latency_us = (timesteps * latency_per_step_ns) / 1000
    static_energy_nj = (static_power_uw * latency_us) / 1000  

    # Updated system energy sum
    total_energy_nj = synop_energy_nj + adc_energy_nj + neuron_energy_nj + static_energy_nj

    # ... keeping your logging format below ...

    metrics = []
    if log:
        print("*"*50)
        print("hardware metrics:")
        print("*"*50)
        print(f"Number of parameters:       {total_params:,}")
        print(f"Number of RRAM Devices:     {total_devices:,} (Differential)")
        print(f"Memory Footprint:           {memory_footprint_kb:.2f} KB")
        print("+" * 50)
        print(f"Average L1 Firing Rate:     {l1_density * 100:.2f}%")
        print(f"Average L2 Firing Rate:     {l2_density * 100:.2f}%")
        print(f"Total Input Spikes:         {int(total_input_spikes):,}")
        print(f"Total Hidden Spikes:        {int(total_hidden_spikes):,}")
        print("+" * 50)
        print(f"Total SynOps:               {int(total_synops):,}")
        print(f"  -> Device-level SynOp Energy:  {synop_energy_nj:.4f} nJ  (analog crossbar switching only)")
        print(f"  -> ADC Conversion Energy:      {adc_energy_nj:.4f} nJ  ({total_adc_reads} reads @ {adc_energy_pj_per_read} pJ/read)")
        print(f"  -> Static/Leakage Energy:      {static_energy_nj:.4f} nJ  (@ {static_power_uw} µW over {latency_us:.2f} µs)")
        print(f"  -> TOTAL System-Level Energy:  {total_energy_nj:.4f} nJ")
        print(f"Inference Latency:          {latency_us:.2f} µs")
        print("+"*50)
        print("NOTE: Device-level SynOp energy alone is NOT directly comparable to full-chip")
        print("      measurements (e.g. Wu et al. 2021, 10.3 µJ/sample) which include ADC,")
        print("      periphery, and static power over a much longer real sample duration.")
        print("+"*50)

    metrics.append({
        "num_param": total_params,
        "num_rram": total_devices,
        "memory": memory_footprint_kb,
        "total_hidden_spikes": int(total_hidden_spikes),
        "total_input_spikes": int(total_input_spikes),
        "total_synops": int(total_synops),
        "synop_energy_nj": synop_energy_nj,
        "adc_energy_nj": adc_energy_nj,
        "static_energy_nj": static_energy_nj,
        "total_energy_nj": total_energy_nj,
        "inference_latency": latency_us,
    })

    return metrics

avg_l1_density = np.mean([item["l1_sparsity"] for item in fold_summary_metrics])
avg_l2_density = np.mean([item["l2_sparsity"] for item in fold_summary_metrics])
avg_input_density = np.mean([item["input_sparsity"] for item in fold_summary_metrics])

ms = calculate_hardware_metrics(input_density=avg_input_density, l1_density=avg_l1_density, l2_density=avg_l2_density)
print(avg_input_density)

--- CELL 13 ---
# ======================================================================
# Digital LSTM vs. MSNN Hardware & Area Comparison Calculator
# ======================================================================

def calculate_baseline_comparison():
    # --- 1. Architectural & Topography Constants ---
    inputs = 288
    hidden = 128
    outputs = 3
    timesteps = 30
    
    # MSNN values from your hardware profile
    msnn_devices = ms[0]["num_rram"]
    msnn_synops = ms[0]["total_synops"]
    msnn_energy_nj = ms[0]["total_energy_nj"]
    
    # --- 2. LSTM Calculations ---
    # LSTM has 4 gates: Input, Forget, Cell, Output
    # Parameters per gate = (Inputs * Hidden) + (Hidden * Hidden) + Hidden (bias)
    params_per_gate = (inputs * hidden) + (hidden * hidden) + hidden
    lstm_params_layer = 4 * params_per_gate
    lstm_params_output = (hidden * outputs) + outputs
    lstm_total_params = lstm_params_layer + lstm_params_output
    
    # Dense LSTMs perform MAC operations for every parameter at every timestep
    lstm_total_macs = lstm_total_params * timesteps
    
    # Energy constant for 32-bit floating-point digital MAC (Horowitz ISSCC standard: ~3.1 pJ per MAC)
    energy_per_mac_pj = 3.1
    lstm_energy_nj = (lstm_total_macs * energy_per_mac_pj) / 1000
    
    # --- 3. Area Estimation Constants (RRAM CIM vs. Digital CMOS) ---
    # Approximate layout area per RRAM device (assuming standard crossbar node, e.g., 65nm or 28nm roughly ~4-10 F^2)
    # Let's use a standard literature estimate: ~0.04 um^2 per RRAM cell including minimal peripheral CMOS footprint share
    # Or macro-level density estimation: ~1000 um^2 per Kb for dense RRAM crossbars.
    rram_area_per_device_um2 = 0.04 
    msnn_crossbar_area_um2 = msnn_devices * rram_area_per_device_um2
    
    # Digital Standard Cell Area estimation for LSTM (32-bit floating point MAC units and registers)
    # A standard 32-bit FP MAC cell in CMOS takes roughly 10,000 to 20,000 gates (~5000 um^2 at 65nm per equivalent gate block)
    # Alternatively, total area proportional to parameter storage and active datapath logic.
    # Let's provide a clear transistor/gate equivalent layout approximation.
    digital_gate_area_um2 = 1.5  # Typical 65nm 2-input NAND equivalent area
    # A 32-bit floating point multiplier/accumulator takes roughly 15,000 equivalent gates. 
    # For a dense sequential LSTM running parallel blocks, let's derive footprint via standard synthesized cell area:
    lstm_estimated_area_um2 = lstm_total_params * 32 * digital_gate_area_um2 * 0.5 # compressed footprint factor

    # --- 4. Print Comparison Table ---
    print("="*65)
    print(f"{'HARDWARE METRIC':<30} | {'MSNN (RRAM CIM)':<15} | {'Digital LSTM':<15}")
    print("="*65)
    print(f"{'Parameters':<30} | {37248:<15,} | {lstm_total_params:<15,}")
    print(f"{'Memory Elements / Devices':<30} | {msnn_devices:<15,} | {lstm_total_params * 32:<15,}")
    print(f"{'Operations (SynOps vs MACs)':<30} | {msnn_synops:<15,} | {lstm_total_macs:<15,}")
    print(f"{'Estimated Energy (nJ)':<30} | {msnn_energy_nj:<15.4f} | {lstm_energy_nj:<15.2f}")
    print(f"{'Approx. Silicon Area (um²)':<30} | {msnn_crossbar_area_um2:<15.2f} | {lstm_estimated_area_um2:<15.2f}")
    print("="*65)
    
    # Efficiency Multipliers
    energy_speedup = lstm_energy_nj / msnn_energy_nj
    op_reduction = lstm_total_macs / msnn_synops
    print(f"\n[SUMMARY] MSNN achieves a {op_reduction:.1f}x reduction in operations")
    print(f"[SUMMARY] MSNN achieves a {energy_speedup:.1f}x improvement in energy efficiency over the LSTM baseline.")

calculate_baseline_comparison()

--- CELL 14 ---
import torch
import glob
import numpy as np
# (Include your other original imports here)

# 1. PASTE YOUR CLASSES AND FUNCTIONS HERE
# Paste: STEQuantize, MemristorCrossbar, MSNN, LOBDayDataset
# Paste: get_anchored_fold_loaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. DEFINE YOUR DATA FILES (Update these paths for your new environment)
train_files = sorted(glob.glob(
    "/kaggle/input/datasets/firmwired/fi-2010/data/published/BenchmarkDatasets/BenchmarkDatasets/BenchmarkDatasets/NoAuction/NoAuction_MinMax/NoAuction_MinMax_Training/Train_Dst_NoAuction_MinMax_CF_*.txt"
))
test_files = sorted(glob.glob(
    "/kaggle/input/datasets/firmwired/fi-2010/data/published/BenchmarkDatasets/BenchmarkDatasets/BenchmarkDatasets/NoAuction/NoAuction_MinMax/NoAuction_MinMax_Testing/Test_Dst_NoAuction_MinMax_CF_*.txt"
))

for i in range(8):
    # 3. SELECT THE FOLD YOU WANT TO TEST
    fold_idx = i # fold_idx = 0 means Fold 1

    # 4. LOAD THE ANCHORED DATA FOR THIS SPECIFIC FOLD
    # We only need the test_loader for inference, so we can ignore the train returns
    _, test_loader, _ = get_anchored_fold_loaders(
        fold_idx=fold_idx, 
        train_files=train_files, 
        test_files=test_files, 
        batch_size=256
    )

    # 5. INSTANTIATE THE MODEL
    new = MSNN(
        num_inputs=288, 
        num_hidden=128, 
        num_outputs=3, 
        beta=0.9, 
        num_levels=16, 
        saf_rate=0.03, 
        noise_std=0.05,
        force_ideal=False,
        force_noise_eval=True
    ).to(device)

    # 6. LOAD THE CORRESPONDING WEIGHTS
    # Match the model number to the fold_idx (fold_idx 0 -> model_1.pt)
    checkpoint_path = f'/kaggle/working/MSNN/noisy/model_8.pt'
    new.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))

    new.eval()
    new.reset_spike_metrics()

    # 7. RUN INFERENCE
    correct = 0
    total = 0

    print(f"\nRunning inference on Fold {fold_idx + 1}...")
    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).long()
            
            logits = new(data)
            preds = logits.argmax(dim=1)
            
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    accuracy = correct / max(1, total)
    print(accuracy)
    print(f"Test Accuracy for Fold {fold_idx + 1}: {accuracy * 100:.2f}%")

    


--- CELL 15 ---
def run_anchored_experiment(seed, force_ideal, force_noise_eval, train_files, test_files,
                             num_folds=8, num_epochs_per_fold=10, patience=4,
                             checkpoint_dir='/kaggle/working/MSNN/sweep'):
    """
    Runs the full anchored-fold training+eval pipeline once, for one seed and one
    hardware condition. Returns a list of per-fold metric dicts (same shape as
    your existing fold_summary_metrics).
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.manual_seed(seed)

    net_seeds = MSNN(num_inputs=288,
               num_hidden=128,
               num_outputs=3,
               beta=0.9,
               num_levels=16,
               saf_rate=0.03,
               noise_std=0.05,
               force_ideal=force_ideal,
               force_noise_eval=force_noise_eval
               ).to(device)
    
    fold_summary = []

    for fold_idx in range(num_folds):
        train_loader, test_loader, fold_train_dataset = get_anchored_fold_loaders(
            fold_idx=fold_idx, 
            train_files=train_files,
            test_files=test_files,
            batch_size=256
        )

        if isinstance(fold_train_dataset, torch.utils.data.ConcatDataset):
            active_labels = torch.cat([ds.Y for ds in fold_train_dataset.datasets])
        else:
            active_labels = fold_train_dataset.Y

        w = weights(device=device, labels=active_labels)
        loss_fn = FocalLoss(alpha=w, gamma=2)

        optimizer = torch.optim.Adam(net_seeds.parameters(), lr=3e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs_per_fold, eta_min=1e-6)

        best_fold_acc = 0.0
        patience_counter = 0
        ckpt_path = f'{checkpoint_dir}/seed{seed}_{"ideal" if force_ideal else "noisy"}_fold{fold_idx+1}.pt'

        for epoch in range(num_epochs_per_fold):
            net_seeds.train()
            
            for data, targets in train_loader:
                data = data.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True).long()
                
                optimizer.zero_grad(set_to_none=True)
                
                loss_val = loss_fn(net_seeds(data), targets)
                
                loss_val.backward()
                torch.nn.utils.clip_grad_norm_(net_seeds.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()

            net_seeds.eval()
            with torch.no_grad():
                correct, total = 0, 0
                for val_data, val_targets in test_loader:
                    val_data = val_data.to(device, non_blocking=True)
                    val_targets = val_targets.to(device, non_blocking=True).long()
                    preds = net_seeds(val_data).argmax(dim=1)
                    correct += (preds == val_targets).sum().item()
                    total += val_targets.size(0)
                current_val_acc = correct / max(1, total)

            if current_val_acc > best_fold_acc:
                best_fold_acc = current_val_acc
                patience_counter = 0
                torch.save(net_seeds.state_dict(), ckpt_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        net_seeds.load_state_dict(torch.load(ckpt_path, map_location=device))

        total_input_sparsity, num_batches = 0.0, 0
        with torch.no_grad():
            for val_data, _ in test_loader:
                total_input_sparsity += val_data.float().mean().item()
                num_batches += 1
        fold_input_sparsity = total_input_sparsity / num_batches

        fold_profile = compute_hardware_metrics(test_loader, net_seeds, device)
        fold_summary.append({
            "fold": fold_idx + 1,
            "accuracy": fold_profile["accuracy"],
            "l1_sparsity": fold_profile["l1_sparsity"],
            "l2_sparsity": fold_profile["l2_sparsity"],
            "input_sparsity": fold_input_sparsity
        })

    return fold_summary


# Run Sweeeep
seeds = [123, 42, 7]   
all_noisy_runs = []
all_ideal_runs = []

for s in seeds:
    print(f"\n++++++++ seed {s} — noisy++++++")
    all_noisy_runs.append(run_anchored_experiment(s, force_ideal=False, force_noise_eval=True,
                                                    train_files=train_files, test_files=test_files))
    print(f"\n++++++++ seed {s} — ideal++++++")
    all_ideal_runs.append(run_anchored_experiment(s, force_ideal=True, force_noise_eval=False,
                                                    train_files=train_files, test_files=test_files))

--- CELL 16 ---
def aggregate_across_seeds(all_runs, num_folds=8):
    """all_runs: list of fold_summary lists (one per seed). Returns per-fold mean/std."""
    acc_matrix = np.array([[run[f]["accuracy"] for f in range(num_folds)] for run in all_runs])
    l1_matrix = np.array([[run[f]["l1_sparsity"] for f in range(num_folds)] for run in all_runs])
    l2_matrix = np.array([[run[f]["l2_sparsity"] for f in range(num_folds)] for run in all_runs])
    return {
        "acc_mean": acc_matrix.mean(axis=0), "acc_std": acc_matrix.std(axis=0),
        "l1_mean": l1_matrix.mean(axis=0), "l1_std": l1_matrix.std(axis=0),
        "l2_mean": l2_matrix.mean(axis=0), "l2_std": l2_matrix.std(axis=0),
    }

noisy_agg = aggregate_across_seeds(all_noisy_runs)
ideal_agg = aggregate_across_seeds(all_ideal_runs)
print("Noisy accuracy (mean ± std) per fold:", noisy_agg["acc_mean"], "±", noisy_agg["acc_std"])
print("Ideal accuracy (mean ± std) per fold:", ideal_agg["acc_mean"], "±", ideal_agg["acc_std"])

--- CELL 17 ---
import matplotlib.pyplot as plt
import numpy as np

def plot_multiseed_results(noisy_agg, ideal_agg, num_folds=8):
    folds = np.arange(1, num_folds + 1)
    
    # Set up a 1x3 grid for the plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Hardware-Realistic MSNN vs. Ideal Software SNN (Averaged over 3 Seeds)", fontsize=16, fontweight='bold', y=1.05)

    # ==========================================
    # Plot 1: Test Accuracy
    # ==========================================
    # Ideal
    axes[0].plot(folds, ideal_agg["acc_mean"] * 100, label="Ideal SNN (FP32)", 
                 marker='o', linestyle='-', color='#1f77b4', linewidth=2)
    axes[0].fill_between(folds, 
                         (ideal_agg["acc_mean"] - ideal_agg["acc_std"]) * 100, 
                         (ideal_agg["acc_mean"] + ideal_agg["acc_std"]) * 100, 
                         color='#1f77b4', alpha=0.2)
    
    # Noisy
    axes[0].plot(folds, noisy_agg["acc_mean"] * 100, label="Noisy MSNN (4-bit, 3% SAF)", 
                 marker='s', linestyle='--', color='#ff7f0e', linewidth=2)
    axes[0].fill_between(folds, 
                         (noisy_agg["acc_mean"] - noisy_agg["acc_std"]) * 100, 
                         (noisy_agg["acc_mean"] + noisy_agg["acc_std"]) * 100, 
                         color='#ff7f0e', alpha=0.2)

    axes[0].set_title("Generalization Accuracy", fontsize=14)
    axes[0].set_xlabel("Fold (Test Trading Day)", fontsize=12)
    axes[0].set_ylabel("Test Accuracy (%)", fontsize=12)
    axes[0].legend(loc="lower right")
    axes[0].grid(True, linestyle=':', alpha=0.7)

    # ==========================================
    # Plot 2: Layer 1 Spike Density
    # ==========================================
    axes[1].plot(folds, ideal_agg["l1_mean"] * 100, label="Ideal SNN", 
                 marker='o', linestyle='-', color='#1f77b4', linewidth=2)
    axes[1].fill_between(folds, 
                         (ideal_agg["l1_mean"] - ideal_agg["l1_std"]) * 100, 
                         (ideal_agg["l1_mean"] + ideal_agg["l1_std"]) * 100, 
                         color='#1f77b4', alpha=0.2)
    
    axes[1].plot(folds, noisy_agg["l1_mean"] * 100, label="Noisy MSNN", 
                 marker='s', linestyle='--', color='#ff7f0e', linewidth=2)
    axes[1].fill_between(folds, 
                         (noisy_agg["l1_mean"] - noisy_agg["l1_std"]) * 100, 
                         (noisy_agg["l1_mean"] + noisy_agg["l1_std"]) * 100, 
                         color='#ff7f0e', alpha=0.2)

    axes[1].set_title("Layer 1 Firing Rate", fontsize=14)
    axes[1].set_xlabel("Fold (Test Trading Day)", fontsize=12)
    axes[1].set_ylabel("Spike Density (%)", fontsize=12)
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle=':', alpha=0.7)

    # ==========================================
    # Plot 3: Layer 2 Spike Density
    # ==========================================
    axes[2].plot(folds, ideal_agg["l2_mean"] * 100, label="Ideal SNN", 
                 marker='o', linestyle='-', color='#1f77b4', linewidth=2)
    axes[2].fill_between(folds, 
                         (ideal_agg["l2_mean"] - ideal_agg["l2_std"]) * 100, 
                         (ideal_agg["l2_mean"] + ideal_agg["l2_std"]) * 100, 
                         color='#1f77b4', alpha=0.2)
    
    axes[2].plot(folds, noisy_agg["l2_mean"] * 100, label="Noisy MSNN", 
                 marker='s', linestyle='--', color='#ff7f0e', linewidth=2)
    axes[2].fill_between(folds, 
                         (noisy_agg["l2_mean"] - noisy_agg["l2_std"]) * 100, 
                         (noisy_agg["l2_mean"] + noisy_agg["l2_std"]) * 100, 
                         color='#ff7f0e', alpha=0.2)

    axes[2].set_title("Layer 2 Firing Rate", fontsize=14)
    axes[2].set_xlabel("Fold (Test Trading Day)", fontsize=12)
    axes[2].set_ylabel("Spike Density (%)", fontsize=12)
    axes[2].legend(loc="upper right")
    axes[2].grid(True, linestyle=':', alpha=0.7)

    # Clean layout and render
    plt.tight_layout()
    plt.show()

# Run the plotting function using the dictionaries you just generated
plot_multiseed_results(noisy_agg, ideal_agg)

--- CELL 18 ---
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Globally force classic MATLAB inward ticks and formatting
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"]  = ["Times New Roman", "DejaVu Serif", "Computer Modern Roman"]

folds = np.arange(1, 9)

# =====================================================================
# PLOT 1: IDEAL HARDWARE
# =====================================================================
accuracy_ideal = ideal_agg["acc_mean"]
l1_density_ideal = ideal_agg["l1_mean"] * 100 
l2_density_ideal = ideal_agg["l2_mean"] * 100

fig1, ax1_ideal = plt.subplots(figsize=(8.5, 5.8), dpi=600) 

ax1_ideal.set_xlabel('Folds (Test Trading Days)', fontweight='bold', labelpad=10)
ax1_ideal.set_ylabel('Test Accuracy', color="#005b96", fontweight='bold', labelpad=10)
line1_ideal = ax1_ideal.plot(folds, accuracy_ideal, color="#005b96", marker='o', linewidth=2.5, markersize=7, label='Test Accuracy')

ymin_ideal = min(accuracy_ideal) - 0.03
ymax_ideal = max(accuracy_ideal) + 0.03
ax1_ideal.set_ylim(min(0.40, ymin_ideal), max(0.65, ymax_ideal)) 

ax1_ideal.tick_params(axis='y', labelcolor="#005b96", length=6)
ax1_ideal.tick_params(axis='x', length=6)
ax1_ideal.set_xticks(folds) 
ax1_ideal.grid(True, linestyle=':', color='gray', alpha=0.5, linewidth=1.0)

ax2_ideal = ax1_ideal.twinx()
ax2_ideal.set_ylabel('Spike Activation Density (%) [Log Scale]', color='black', fontweight='bold', labelpad=12)

line2_ideal = ax2_ideal.plot(folds, l1_density_ideal, color="#d95f02", marker='s', linestyle='--', linewidth=2, markersize=7, label='Layer 1 (Hidden)')
line3_ideal = ax2_ideal.plot(folds, l2_density_ideal, color="#1b9e77", marker='^', linestyle='-.', linewidth=2, markersize=7, label='Layer 2 (Output)')

ax2_ideal.set_yscale('log')
all_densities_ideal = np.concatenate([l1_density_ideal, l2_density_ideal])
ax2_ideal.set_ylim(min(all_densities_ideal) * 0.9, max(all_densities_ideal) * 1.5)
ax2_ideal.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)))
ax2_ideal.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
ax2_ideal.yaxis.set_minor_formatter(ticker.NullFormatter()) 
ax2_ideal.tick_params(axis='y', which='both', labelcolor='black', length=6, direction='in')

ax1_ideal.set_zorder(ax2_ideal.get_zorder() + 1)  
ax1_ideal.set_frame_on(False)                
ax2_ideal.set_frame_on(True)                

lines_ideal = line1_ideal + line2_ideal + line3_ideal
labels_ideal = [l.get_label() for l in lines_ideal]
ax1_ideal.legend(lines_ideal, labels_ideal, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=3, 
                 frameon=True, facecolor='white', edgecolor='black', framealpha=1.0, borderpad=0.8)

plt.title("Software baseline performance (Ideal FP32)", fontweight='bold', pad=15)
fig1.tight_layout(pad=1.5)
plt.savefig("evaluation_plot_ideal.pdf", format='pdf', bbox_inches='tight')
plt.show()

# =====================================================================
# PLOT 2: NOISY HARDWARE
# =====================================================================
accuracy_noisy = noisy_agg["acc_mean"]
l1_density_noisy = noisy_agg["l1_mean"] * 100 
l2_density_noisy = noisy_agg["l2_mean"] * 100

fig2, ax1_noisy = plt.subplots(figsize=(8.5, 5.8), dpi=600) 

ax1_noisy.set_xlabel('Folds (Test Trading Days)', fontweight='bold', labelpad=10)
ax1_noisy.set_ylabel('Test Accuracy', color="#005b96", fontweight='bold', labelpad=10)
line1_noisy = ax1_noisy.plot(folds, accuracy_noisy, color="#005b96", marker='o', linewidth=2.5, markersize=7, label='Test Accuracy')

ymin_noisy = min(accuracy_noisy) - 0.03
ymax_noisy = max(accuracy_noisy) + 0.03
ax1_noisy.set_ylim(min(0.40, ymin_noisy), max(0.65, ymax_noisy)) 

ax1_noisy.tick_params(axis='y', labelcolor="#005b96", length=6)
ax1_noisy.tick_params(axis='x', length=6)
ax1_noisy.set_xticks(folds) 
ax1_noisy.grid(True, linestyle=':', color='gray', alpha=0.5, linewidth=1.0)

ax2_noisy = ax1_noisy.twinx()
ax2_noisy.set_ylabel('Spike Activation Density (%) [Log Scale]', color='black', fontweight='bold', labelpad=12)

line2_noisy = ax2_noisy.plot(folds, l1_density_noisy, color="#d95f02", marker='s', linestyle='--', linewidth=2, markersize=7, label='Layer 1 (Hidden)')
line3_noisy = ax2_noisy.plot(folds, l2_density_noisy, color="#1b9e77", marker='^', linestyle='-.', linewidth=2, markersize=7, label='Layer 2 (Output)')

ax2_noisy.set_yscale('log')
all_densities_noisy = np.concatenate([l1_density_noisy, l2_density_noisy])
ax2_noisy.set_ylim(min(all_densities_noisy) * 0.9, max(all_densities_noisy) * 1.5)
ax2_noisy.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)))
ax2_noisy.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
ax2_noisy.yaxis.set_minor_formatter(ticker.NullFormatter()) 
ax2_noisy.tick_params(axis='y', which='both', labelcolor='black', length=6, direction='in')

ax1_noisy.set_zorder(ax2_noisy.get_zorder() + 1)  
ax1_noisy.set_frame_on(False)                  
ax2_noisy.set_frame_on(True)                

lines_noisy = line1_noisy + line2_noisy + line3_noisy
labels_noisy = [l.get_label() for l in lines_noisy] 
ax1_noisy.legend(lines_noisy, labels_noisy, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=3, 
                 frameon=True, facecolor='white', edgecolor='black', framealpha=1.0, borderpad=0.8)

plt.title("Hardware performance under ≈ 3% SAF & D2D/C2C Variations", fontweight='bold', pad=15)
fig2.tight_layout(pad=1.5)
plt.savefig("evaluation_plot_noisy.pdf", format='pdf', bbox_inches='tight')
plt.show()

--- CELL 19 ---
import numpy as np
import matplotlib.pyplot as plt

folds = np.arange(1, 9)
noisy_mean = np.array([0.43582997, 0.46826148, 0.48645279, 0.54074547, 0.52750982, 0.61965735, 0.59619011, 0.56121383])
noisy_std  = np.array([0.01650595, 0.00442006, 0.01186877, 0.00531782, 0.00712416, 0.00885089, 0.01941716, 0.0114676])
ideal_mean = np.array([0.47430275, 0.49601297, 0.5301875, 0.58900958, 0.57877814, 0.67195672, 0.6546489, 0.63940249])
ideal_std  = np.array([0.00995455, 0.00783967, 0.00749414, 0.00410673, 0.01159083, 0.0079149, 0.00405025, 0.00693974])

setup_academic_style()
fig, ax = plt.subplots(figsize=(8, 5.5))

ax.plot(folds, noisy_mean, color="#005b96", marker='o', linewidth=2.5, markersize=7, label='Noisy (Realistic Hardware)')
ax.fill_between(folds, noisy_mean - noisy_std, noisy_mean + noisy_std, color="#005b96", alpha=0.2)

ax.plot(folds, ideal_mean, color="#d95f02", marker='s', linewidth=2.5, markersize=7, label='Ideal (No Non-idealities)')
ax.fill_between(folds, ideal_mean - ideal_std, ideal_mean + ideal_std, color="#d95f02", alpha=0.2)

ax.set_xlabel('Folds (Test Trading Days)', fontweight='bold', labelpad=10)
ax.set_ylabel('Test Accuracy', fontweight='bold', labelpad=10)
ax.set_xticks(folds)
ax.grid(True, linestyle=':', color='gray', alpha=0.5, linewidth=1.0)
ax.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='black', framealpha=1.0)
plt.title("Accuracy: Realistic vs. Ideal Hardware (mean ± std, 3 seeds)", fontweight='bold', pad=15)
fig.tight_layout()
plt.savefig("noisy_vs_ideal_multiseed.pdf", bbox_inches='tight')
plt.show()

==============================
FILE: notebooks\prep.ipynb
==============================
--- CELL 0 ---
%pip install torch snntorch 
import numpy as np
import torch
import torch.nn as nn
import numpy as np
import snntorch as snn
import random
from snntorch import utils
from snntorch import spikegen, surrogate
import snntorch.functional as SF
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.patches as mpatches
from torch.nn.functional import cross_entropy
import torch.nn.functional as F
from torch.utils.data import Subset
# -------------- Parameters -----------------------
time_steps = 39512

--- CELL 1 ---
# Helper functions:
def create_sequences(inputs, labels, window_size, stride=5):
    X, Y = [], []
    for i in range(0, len(inputs) - window_size, stride):
        X.append(inputs[i : i + window_size])
        Y.append(labels[i + window_size - 1])
    return torch.stack(X), torch.stack(Y)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # weights
        self.gamma = gamma 
        self.reduction = reduction
    
    def forward(self, logits, targets):
        ce_loss = cross_entropy(logits, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma  
        loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        return loss.sum()


def per_class_accuracy(data_loader, net, device, num_classes=3):
    net.eval()
    correct = torch.zeros(num_classes)
    total = torch.zeros(num_classes)
    with torch.no_grad():
        for data, targets in data_loader:
            data = data.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).long()
            logits = net(data)
            preds = logits.argmax(dim=1)
            for c in range(num_classes):
                mask = targets == c
                total[c] += mask.sum().item()
                correct[c] += (preds[mask] == c).sum().item()
    return (correct / total.clamp(min=1)).cpu().numpy()
    
def evaluate_epoch(data_loader, net, loss_fn, device, num_classes=3):
    """Single pass: returns per-class loss AND per-class accuracy together"""
    net.eval()
    class_losses = [[] for _ in range(num_classes)]
    correct = torch.zeros(num_classes)
    total = torch.zeros(num_classes)
    
    with torch.no_grad():
        for data, targets in data_loader:
            data = data.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).long()
            
            logits = net(data)  # ONE forward pass per batch, not two
            preds = logits.argmax(dim=1)
            
            for c in range(num_classes):
                mask = targets == c
                if mask.sum() > 0:
                    class_loss = loss_fn(logits[mask], targets[mask])
                    class_losses[c].append(class_loss.item())
                total[c] += mask.sum().item()
                correct[c] += (preds[mask] == c).sum().item()
    
    avg_losses = [np.mean(l) if l else float('nan') for l in class_losses]
    per_class_acc = (correct / total.clamp(min=1)).cpu().numpy()
    return avg_losses, per_class_acc
    
def compute_loss_per_class(data_loader, net, loss_fn, device, num_classes=3):
    """compute loss for each class separately"""
    net.eval()
    class_losses = [[] for _ in range(num_classes)]
    
    with torch.no_grad():
        for data, targets in data_loader:
            data = data.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).long()
            
            logits = net(data)
            
            # compute loss for each sample then group by class really dont know why!!!
            for c in range(num_classes):
                mask = targets == c
                if mask.sum() > 0:
                    class_logits = logits[mask]
                    class_targets = targets[mask]
                    class_loss = loss_fn(class_logits, class_targets)
                    class_losses[c].append(class_loss.item())
    
    # avg per class
    avg_losses = [np.mean(losses) if losses else float('nan') for losses in class_losses]
    return avg_losses


def plot_loss_per_class(train_loader, test_loader, net, loss_fn, device, epoch=25):
    train_losses = compute_loss_per_class(train_loader, net, loss_fn, device)
    test_losses = compute_loss_per_class(test_loader, net, loss_fn, device)
    
    classes = ['Up', 'Stationary', 'Down']
    x = np.arange(len(classes))
    width = 0.35
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, train_losses, width, label='Train Loss', alpha=0.8)
    plt.bar(x + width/2, test_losses, width, label='Test Loss', alpha=0.8)
    
    plt.xlabel('Class')
    plt.ylabel('Cross Entropy Loss (focal loss)')
    plt.title(f'Loss Per Class (Final Epoch {epoch} )')
    plt.xticks(x, classes)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("train loss per class:", [f"{l:.4f}" for l in train_losses])
    print("test loss per class:", [f"{l:.4f}" for l in test_losses])

 


--- CELL 2 ---
class STEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, num_levels):
        scaled = x * (num_levels - 1)
        quantized = torch.round(scaled)
        return quantized / (num_levels - 1)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class MemristorCrossbarLinear(nn.Module):
    def __init__(self, in_features, out_features, num_levels=16, saf_rate=0.02, noise_std=0.03):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_levels = num_levels
        self.noise_std = noise_std
        self.saf_rate = saf_rate
        self.force_noise_eval = False  
        
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        nn.init.normal_(self.weight, mean=0.0, std=0.05)
        nn.init.zeros_(self.bias)
        
        self.register_buffer('saf_g_pos_lrs', torch.zeros(out_features, in_features, dtype=torch.bool), persistent=False)
        self.register_buffer('saf_g_pos_hrs', torch.zeros(out_features, in_features, dtype=torch.bool), persistent=False)
        self.register_buffer('saf_g_neg_lrs', torch.zeros(out_features, in_features, dtype=torch.bool), persistent=False)
        self.register_buffer('saf_g_neg_hrs', torch.zeros(out_features, in_features, dtype=torch.bool), persistent=False)
        
        self.generate_saf_masks(saf_rate)

    def generate_saf_masks(self, saf_rate):
        self.saf_rate = saf_rate
        device = self.weight.device
        
        self.saf_g_pos_lrs.copy_(torch.rand(self.out_features, self.in_features, device=device) < (saf_rate / 2))
        self.saf_g_pos_hrs.copy_(torch.rand(self.out_features, self.in_features, device=device) < (saf_rate / 2))
        self.saf_g_neg_lrs.copy_(torch.rand(self.out_features, self.in_features, device=device) < (saf_rate / 2))
        self.saf_g_neg_hrs.copy_(torch.rand(self.out_features, self.in_features, device=device) < (saf_rate / 2))

    def get_physical_weights(self):
        """Extracts the quantized, noisy physical weight matrix ONCE."""
        weight_scale = torch.max(torch.abs(self.weight)).detach().clamp(min=1e-5)
        w_normalized = torch.clamp(self.weight / weight_scale, -1.0, 1.0)
        
        g_pos = (w_normalized + 1.0) / 2.0
        g_neg = (1.0 - w_normalized) / 2.0
        
        # Apply SAF defects
        g_pos = torch.where(self.saf_g_pos_lrs, torch.ones_like(g_pos), g_pos)
        g_pos = torch.where(self.saf_g_pos_hrs, torch.zeros_like(g_pos), g_pos)
        g_neg = torch.where(self.saf_g_neg_lrs, torch.ones_like(g_neg), g_neg)
        g_neg = torch.where(self.saf_g_neg_hrs, torch.zeros_like(g_neg), g_neg)
        
        # Finite State Quantization
        g_pos_q = STEQuantize.apply(g_pos, self.num_levels)
        g_neg_q = STEQuantize.apply(g_neg, self.num_levels)
        
        if (self.training or self.force_noise_eval) and self.noise_std > 0:
            noise_pos = torch.randn_like(g_pos_q) * self.noise_std
            noise_neg = torch.randn_like(g_neg_q) * self.noise_std
            g_pos_q = torch.clamp(g_pos_q + noise_pos, 0.0, 1.0)
            g_neg_q = torch.clamp(g_neg_q + noise_neg, 0.0, 1.0)
            
        return (g_pos_q - g_neg_q) * weight_scale

    def forward(self, x):
        w_physical = self.get_physical_weights()
        return F.linear(x, w_physical, self.bias)
    

def reset_to_clean_eval_state(net, saf_rate=0.0, noise_std=0.0):
        for name, module in net.named_modules():
            if isinstance(module, MemristorCrossbarLinear):
                module.generate_saf_masks(saf_rate)
                module.noise_std = noise_std
                module.force_noise_eval = False

--- CELL 3 ---
import glob
import gc
from torch.utils.data import Dataset, ConcatDataset, DataLoader

class LOBDayDataset(Dataset):
    def __init__(self, filepath, window_size=30, stride=5, delta_threshold=0.005):
        data = np.loadtxt(filepath)
        features = data[:144, :].T
        labels = data[146, :].T - 1
        del data

        features_tensor = torch.from_numpy(features).float()
        labels_tensor = torch.from_numpy(labels).long()
        del features, labels

        spike = spikegen.delta(features_tensor, threshold=delta_threshold, padding=True, off_spike=True)
        spikes_pos = torch.where(spike > 0, spike, torch.zeros_like(spike))
        spikes_neg = torch.where(spike < 0, torch.abs(spike), torch.zeros_like(spike))
        del spike, features_tensor

        inputs = torch.cat((spikes_pos, spikes_neg), dim=1).half()  # float16 -> halves memory
        del spikes_pos, spikes_neg

        self.X, self.Y = self._create_sequences(inputs, labels_tensor, window_size, stride)
        del inputs, labels_tensor
        gc.collect()

    def _create_sequences(self, inputs, labels, window_size, stride):
        X, Y = [], []
        for i in range(0, len(inputs) - window_size, stride):
            X.append(inputs[i : i + window_size])
            Y.append(labels[i + window_size - 1])
        return torch.stack(X), torch.stack(Y)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx].float(), self.Y[idx]  # cast back per-sample, not stored as float32


# ---- Build combined training set from all 9 days ----
train_files = sorted(glob.glob(
    "/kaggle/input/datasets/firmwired/fi-2010/data/published/BenchmarkDatasets/BenchmarkDatasets/BenchmarkDatasets/NoAuction/NoAuction_MinMax/NoAuction_MinMax_Training/Train_Dst_NoAuction_MinMax_CF_*.txt"
))
print(f"Found {len(train_files)} training files")

train_datasets = []
for f in train_files:
    print(f"Loading {f.split('/')[-1]}...")
    ds = LOBDayDataset(f, window_size=30, stride=5, delta_threshold=0.005)
    train_datasets.append(ds)
    print(f"  -> {len(ds)} windows")

full_train_dataset = ConcatDataset(train_datasets)
print(f"Total training windows: {len(full_train_dataset)}")

train_loader = DataLoader(full_train_dataset, batch_size=256, shuffle=True,
                           pin_memory=True, num_workers=4)
 
test_files = sorted(glob.glob(
    "/kaggle/input/datasets/firmwired/fi-2010/data/published/BenchmarkDatasets/BenchmarkDatasets/BenchmarkDatasets/NoAuction/NoAuction_MinMax/NoAuction_MinMax_Testing/Test_Dst_NoAuction_MinMax_CF_*.txt"
))
print(f"Found {len(test_files)} test files")

test_datasets = []
for f in test_files:
    print(f"Loading {f.split('/')[-1]}...")
    ds = LOBDayDataset(f, window_size=30, stride=5, delta_threshold=0.005)
    test_datasets.append(ds)
    print(f"  -> {len(ds)} windows")

full_test_dataset = ConcatDataset(test_datasets)
print(f"Total test windows: {len(full_test_dataset)}")

test_loader = DataLoader(full_test_dataset, batch_size=256, shuffle=False, pin_memory=True)
 

--- CELL 4 ---
num_inputs = 288
num_hidden = 128
num_outputs = 3
beta = .9
window_size = 30  
spike_grad = surrogate.fast_sigmoid(slope=25)
best_test_loss = float('inf')
patience = 15
patience_counter = 0
best_test_metric = 0.0
 
 
class HFT_Net_Optimized(nn.Module):
    def __init__(self, num_levels=16, saf_rate=0.02, noise_std=0.03):
        super().__init__()
        self.fc1  = MemristorCrossbarLinear(num_inputs, num_hidden, 
                                            num_levels=num_levels, saf_rate=saf_rate, noise_std=noise_std)
        self.lif1 = snn.Leaky(threshold=0.5, beta=beta, spike_grad=spike_grad)
        
        self.drop = nn.Dropout(0.3) 
        
        self.fc2  = MemristorCrossbarLinear(num_hidden, num_outputs, 
                                            num_levels=num_levels, saf_rate=saf_rate, noise_std=noise_std)
        self.lif2 = snn.Leaky(threshold=1.0, beta=beta, spike_grad=spike_grad, reset_mechanism="none")  
        
    def forward(self, x):
        x = x.permute(1, 0, 2) # (Time, Batch, Features)
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        # 1. Compute ALL Layer 1 currents in one shot (Vectorized over Time & Batch)
        cur1_all = self.fc1(x) 
        
        # 2. Extract Layer 2's physical weights ONCE before the loop begins
        w_phys2 = self.fc2.get_physical_weights()
        bias2 = self.fc2.bias
        
        for step in range(x.shape[0]):
            spk1, mem1 = self.lif1(cur1_all[step], mem1)
            
            spk1_dropped = self.drop(spk1) 
            
            # 3. Apply the cached weights to avoid recalculating noise/quantization
            cur2 = F.linear(spk1_dropped, w_phys2, bias2)
            spk2, mem2 = self.lif2(cur2, mem2)
            
        return mem2
    
num_epochs = 40
loss_hist = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

torch.manual_seed(123) 
net = HFT_Net_Optimized(num_levels=16, saf_rate=0.02, noise_std=0.03).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999), weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
 
all_labels = torch.cat([ds.Y for ds in train_datasets])
w = weights(device=device, labels=all_labels)
loss_fn = FocalLoss(alpha=w, gamma=2)
sample_data, sample_targets = next(iter(train_loader))
print(f"train_loader batch shape: {sample_data.shape}, {len(train_loader)}")  # expect (batch, window_size, 288)
assert sample_data.dim() == 3, f"Expected 3D input, got {sample_data.dim()}D — check which train_loader is active!" 

eval_subset_size = min(5000, len(full_test_dataset))
eval_indices = random.sample(range(len(full_test_dataset)), eval_subset_size)
eval_subset = Subset(full_test_dataset, eval_indices)
eval_subset_loader = DataLoader(eval_subset, batch_size=256, shuffle=False, pin_memory=True)


for epoch in range(num_epochs):
    net.train()
    print("start")
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).long()
        
        # 1. DYNAMIC INJECTION: Randomize defects for every training batch
        for name, module in net.named_modules():
            if isinstance(module, MemristorCrossbarLinear):
                module.generate_saf_masks(saf_rate=0.03) # Training at 3% SAF
                module.noise_std = 0.05                 # Training at 5% Noise
        
        optimizer.zero_grad(set_to_none=True)
        
        # 2. Forward pass (now includes randomized noise/SAF)
        logits = net(data)
        loss_val = loss_fn(logits, targets)
        
        loss_val.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()
         
            
        
        loss_hist.append(loss_val.item())
        pass
    scheduler.step()
    
    net.eval()
    reset_to_clean_eval_state(net, saf_rate=0.0, noise_std=0.0) 
    test_losses, test_per_class = evaluate_epoch(eval_subset_loader, net, loss_fn, device)
    avg_test_loss = np.nanmean(test_losses)
    mean_test_acc = test_per_class.mean()
    print(f"acc {mean_test_acc}")
   
    # Save the model based on the highest balanced accuracy, NOT the lowest loss!
    if mean_test_acc > best_test_metric:
        best_test_metric = mean_test_acc
        patience_counter = 0
        torch.save(net.state_dict(), 'best_model.pt')
        print(f"Epoch {epoch}: NEW BEST CHECKPOINT | Mean Acc: {mean_test_acc*100:.2f}% | Loss: {avg_test_loss:.4f}")
    else:
        patience_counter += 1 
        print(f"Epoch {epoch}: Mean Acc: {mean_test_acc*100:.2f}% | Loss: {avg_test_loss:.4f} (Patience: {patience_counter}/{patience})")
        
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break
    print(f"**** epoch {epoch} complete ****\n")
    
net.load_state_dict(torch.load('best_model.pt', map_location=device))
net.eval()
reset_to_clean_eval_state(net, saf_rate=0.0, noise_std=0.0) 
print(f"Reloaded best checkpoint")

--- CELL 5 ---
print(f"num_epochs variable currently: {num_epochs}")
print(f"scheduler T_max: {scheduler.T_max}")
print(f"Total loss_hist entries: {len(loss_hist)}")
print(f"Batches per epoch: {len(train_loader)}")
print(f"Epochs actually completed: {len(loss_hist) / len(train_loader):.1f}")
import os, time
print(f"best_model.pt last modified: {time.ctime(os.path.getmtime('best_model.pt'))}")
print(f"Current time: {time.ctime()}")

--- CELL 6 ---
# visualizing results
import matplotlib.pyplot as plt
import numpy as np

batches_per_epoch = len(train_loader)
total_epochs = len(loss_hist) // batches_per_epoch
# average loss per epoch
loss_array = np.array(loss_hist)[:total_epochs * batches_per_epoch].reshape(total_epochs, batches_per_epoch)

epoch_losses = loss_array.mean(axis=1)
epochs_x = np.arange(1, total_epochs + 1)
batches_x = np.linspace(1, total_epochs, len(loss_hist))

plt.figure(figsize=(10, 6))
plt.plot(batches_x, loss_hist, color='skyblue', alpha=0.3, label='Batch Loss')
plt.plot(epochs_x, epoch_losses, color='darkblue', linewidth=2.5, marker='o', label='Epoch Average Loss')

plt.title("SNN Training Loss over Epochs", fontsize=12)
plt.xlabel("Epochs", fontsize=12)
plt.ylabel("Focal Entropy Loss", fontsize=12)
plt.xticks(epochs_x)  
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc="upper right", fontsize=11)
plt.tight_layout()

plt.show()

### The testing and ensuration
train_per_class = per_class_accuracy(train_loader, net, device)
test_per_class = per_class_accuracy(test_loader, net, device)
plot_loss_per_class(train_loader, test_loader, net, loss_fn, device, epoch=45)

print(f"train per-class accuracy: {train_per_class}")
print(f"test per-class accuracy: {test_per_class}")
print(f"train overall: {train_per_class.mean():.4f} ** test overall: {test_per_class.mean():.4f}")

--- CELL 7 ---
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def get_all_predictions(data_loader, net, device):
    net.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, targets in data_loader:
            data = data.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).long()
            logits = net(data)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
    return torch.cat(all_targets).numpy(), torch.cat(all_preds).numpy()


def evaluate_full(data_loader, net, device, split_name="Test"):
    y_true, y_pred = get_all_predictions(data_loader, net, device)
    
    class_names = ['Up', 'Stationary', 'Down']
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{split_name} Confusion Matrix')
    plt.tight_layout()
    plt.show()
    
    # Full classification report (precision, recall, F1 per class)
    report = classification_report(y_true, y_pred, labels=[0, 1, 2],
                                     target_names=class_names, digits=4)
    print(f"\n{split_name} Classification Report:")
    print(report)
    
    # Macro and weighted F1 (single numbers for comparing against the paper)
    macro_f1 = f1_score(y_true, y_pred, labels=[0, 1, 2], average='macro')
    weighted_f1 = f1_score(y_true, y_pred, labels=[0, 1, 2], average='weighted')
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    
    return cm, report, macro_f1, weighted_f1


# Run on both splits
print("="*60)
train_cm, train_report, train_macro_f1, train_weighted_f1 = evaluate_full(
    train_loader, net, device, split_name="Train"
)

print("="*60)
test_cm, test_report, test_macro_f1, test_weighted_f1 = evaluate_full(
    test_loader, net, device, split_name="Test"
)

--- CELL 8 ---
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def evaluate_and_visualize_concat(net, test_loader, first_test_file, device, window_size=30, limit=3000):
    """
    Evaluates a concatenated data_loader by accumulating batches until the 'limit' is reached.
    Assumes shuffle=False so the first N samples match the first raw text file.
    """
    net.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        # Iterate through the concatenated loader to gather enough batches
        for data, targets in test_loader:
            data = data.to(device)
            logits = net(data)
            preds = logits.argmax(dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())
            
            # Stop once we have enough samples for the visualization limit
            if len(all_preds) >= limit:
                break

    # Cap exactly at the limit
    slice_preds = np.array(all_preds[:limit])
    slice_targets = np.array(all_targets[:limit])
    actual_limit = len(slice_preds)

    # Load the raw data from the FIRST file to get the continuous mid-prices
    # Since test_loader has shuffle=False, these perfectly align with the first N predictions
    raw_data = np.loadtxt(first_test_file) 
    raw_ask = raw_data[0, :]
    raw_bid = raw_data[2, :]
    raw_mid_prices = (raw_ask + raw_bid) / 2

    time_steps = np.arange(actual_limit)
     
    correct_mask = (slice_preds == slice_targets)
    wrong_mask = (slice_preds != slice_targets)

    # Align predictions with the raw data accounting for the sliding window
    start_idx = window_size - 1
    slice_prices = raw_mid_prices[start_idx : start_idx + actual_limit]

    # ---------------------------
    # Plotting
    # ---------------------------
    plt.figure(figsize=(16, 8))
    color_map = {0: 'green', 1: 'gray', 2: 'red'}

    for i in range(len(time_steps) - 1):
        label = slice_targets[i]
        if label == 0:
            color, alpha = 'green', 0.15
        elif label == 1:
            color, alpha = 'gray', 0.05
        elif label == 2:
            color, alpha = 'red', 0.15
        plt.axvspan(time_steps[i], time_steps[i+1], color=color, alpha=alpha, lw=0)
        
    plt.plot(time_steps, slice_prices, color='black', linewidth=2.5, zorder=1, label='Mid-Price')

    for c in range(3):
        mask = slice_preds == c
        if np.any(mask):
            plt.scatter(time_steps[mask], slice_prices[mask], 
                        color=color_map[c], s=80, edgecolor='black', zorder=2, 
                        label=f'SNN Predicted: Class {c}', alpha=1)
            
    if np.any(correct_mask):
        plt.scatter(time_steps[correct_mask], slice_prices[correct_mask], color='none',
                    edgecolor='darkgreen', marker='o', s=250, linewidths=2.5, zorder=3, label='Prediction: CORRECT')

    if np.any(wrong_mask):
        plt.scatter(time_steps[wrong_mask], slice_prices[wrong_mask], color='darkred',
                    marker='x', s=150, linewidths=3, zorder=4, label='Prediction: WRONG')

    plt.title("SNN Predictions vs REAL Stock Price (Concatenated Dataset)", fontsize=16, fontweight='bold')
    plt.xlabel("Time Ticks", fontsize=12)
    plt.ylabel("Normalized Mid-Price", fontsize=12)
    plt.ylim(slice_prices.min() - 0.0005, slice_prices.max() + 0.0005) 

    # Legend
    true_up = mpatches.Patch(color='green', alpha=0.15, label='price up')
    true_flat = mpatches.Patch(color='gray', alpha=0.15, label='stationary')
    true_down = mpatches.Patch(color='red', alpha=0.15, label='down')
    plt.legend(handles=[plt.Line2D([0], [0], color='black', lw=2), true_up, true_flat, true_down], 
               fontsize=11, title="labels")

    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()

    # ---------------------------
    # Logging
    # ---------------------------
    print("\n" + "="*85)
    print(f"{'Tick':<6} | {'P_t':<10} | {'P_t+1':<10} | {'label':<13} | {'Target':<8} | {'Prediction':<10} | {'Status':<8}")
    print("="*85)

    for i in range(actual_limit - 1):
        current_p = slice_prices[i]
        next_p = slice_prices[i+1]
        
        if next_p > current_p:
            actual_move = "class[0]"
        elif next_p < current_p:
            actual_move = "class[2]"
        else:
            actual_move = "class[1]"
            
        target_val = slice_targets[i]
        pred_val = slice_preds[i]
        
        status = "correct" if target_val == pred_val else "wrong"
        
        if (actual_move.startswith("class[0]") and target_val != 0) or \
           (actual_move.startswith("class[1]") and target_val != 1) or \
           (actual_move.startswith("class[2]") and target_val != 2):
            status += " * target mismatch"

        print(f"{i:<6} | {current_p:<10.5f} | {next_p:<10.5f} | {actual_move:<13} | {target_val:<8} | {pred_val:<10} | {status}")

    print("="*85)
    correct = np.sum(slice_preds == slice_targets)
    wrong = np.sum(slice_preds != slice_targets)
    print(f"Total mapped ticks: {actual_limit}")
    print(f"Correct: {correct} ({(correct/actual_limit)*100:.1f}%)")
    print(f"Wrong: {wrong}")

# ==========================================
# Run the visualization
# ==========================================
first_test_file = "/kaggle/input/datasets/firmwired/fi-2010/data/published/BenchmarkDatasets/BenchmarkDatasets/BenchmarkDatasets/NoAuction/NoAuction_MinMax/NoAuction_MinMax_Testing/Test_Dst_NoAuction_MinMax_CF_1.txt"

evaluate_and_visualize_concat(
    net=net, 
    test_loader=test_loader, 
    first_test_file=first_test_file, 
    device=device,
    window_size=30, 
    limit=30000
)

--- CELL 9 ---
# def validate_hardware_robustness(model_path, test_loader, device, n_trials=5):
#     noise_sweeps = [0.0, 0.02, 0.05, 0.08, 0.12]
#     saf_sweeps = [0.0, 0.01, 0.03, 0.05, 0.10]

#     eval_net = HFT_Net_Optimized(num_levels=16, saf_rate=0.0, noise_std=0.0).to(device)
#     eval_net.load_state_dict(torch.load(model_path, map_location=device))
#     eval_net.eval()

#     results = {}
#     print("\n--- Running Memristor Crossbar Hardware Validation Sweep ---")
#     for saf in saf_sweeps:
#         results[saf] = {'mean': [], 'std': []}
#         for noise in noise_sweeps:
#             trial_accs = []
#             for trial in range(n_trials):
#                 for name, module in eval_net.named_modules():
#                     if isinstance(module, MemristorCrossbarLinear):
#                         module.generate_saf_masks(saf)  # fresh defect draw each trial
#                         module.noise_std = noise
#                         module.force_noise_eval = True
#                 accs = per_class_accuracy(test_loader, eval_net, device)
#                 trial_accs.append(accs.mean())

#             mean_acc = np.mean(trial_accs)
#             std_acc = np.std(trial_accs)
#             results[saf]['mean'].append(mean_acc)
#             results[saf]['std'].append(std_acc)
#             print(f"SAF: {saf*100:.1f}% | Noise: {noise*100:.1f}% | "
#                   f"Acc: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")

#     plt.figure(figsize=(10, 6))
#     for saf, data in results.items():
#         means = [a*100 for a in data['mean']]
#         stds = [s*100 for s in data['std']]
#         plt.errorbar([n*100 for n in noise_sweeps], means, yerr=stds,
#                      marker='o', capsize=4, label=f'SAF Rate {saf*100:.0f}%')

#     plt.title("SNN Physical Robustness: Accuracy Under RRAM Non-Idealities", fontsize=13, fontweight='bold')
#     plt.xlabel("Cycle-to-Cycle (C2C) Conductance Noise (%)", fontsize=11)
#     plt.ylabel("Mean Test Accuracy (%) ± std", fontsize=11)
#     plt.grid(True, linestyle=':', alpha=0.6)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

#     return results

# results = validate_hardware_robustness('best_model.pt', test_loader, device, n_trials=5)

--- CELL 10 ---
# %%bash
# # 1. Create a writable directory and copy the NeuroSIM codebase there
# mkdir -p /kaggle/working/NeuroSIM
# cp -r /kaggle/input/datasets/firmwired/neurosim/DNN_NeuroSim_V2.1/Training_pytorch/NeuroSIM/* /kaggle/working/NeuroSIM/

# cd /kaggle/working/NeuroSIM/

# # 2. Convert text line endings from Windows (CRLF) to Linux (LF) to prevent script parsing bugs
# sed -i 's/\r$//' *

# # 3. Clean up any previous partial compilation fragments and build the native Linux binary
# make clean
# make

# # 4. Explicitly mark the built binary as an executable file
# chmod +x ./main

# # 5. Confirm the executable exists in the writable working space
# ls -la ./main

--- CELL 11 ---
 
# %%bash
# cd /kaggle/working/NeuroSIM/

# # 1. Shrink BOTH Rows and Columns for the Inference Arrays
# sed -i 's/numRowSubArray = 128;/numRowSubArray = 4;/g' Param.cpp
# sed -i 's/numColSubArray = 128;/numColSubArray = 4;/g' Param.cpp

# # 2. Shrink BOTH Rows and Columns for the Weight Gradient Arrays (if applicable)
# sed -i 's/numRowSubArrayWG = 128;/numRowSubArrayWG = 4;/g' Param.cpp
# sed -i 's/numColSubArrayWG = 128;/numColSubArrayWG = 4;/g' Param.cpp

# # 3. Print the lines to verify the change
# grep -E "numRowSubArray|numColSubArray" Param.cpp

# # 4. Recompile with the proper square 32x32 arrays
# make clean
# make
# chmod +x ./main

--- CELL 12 ---
# import numpy as np
# import torch
# from sklearn.metrics import f1_score, classification_report

# def evaluate_environment_latency(model, dataloader):
#     # Lock model to a single T4 to ensure an identical hardware footprint
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)
#     model.eval()
    
#     all_preds = []
#     all_targets = []
#     latencies = []
    
#     # Microsecond-precise T4 hardware timers
#     start_event = torch.cuda.Event(enable_timing=True)
#     end_event = torch.cuda.Event(enable_timing=True)
    
#     # Warm up the T4 memory cache before timing
#     dummy_shape = next(iter(dataloader))[0][0:1].shape
#     dummy_input = torch.randn(dummy_shape).to(device)
#     for _ in range(20):
#         _ = model(dummy_input)
        
#     print(f"Timing inference on: {torch.cuda.get_device_name(0)}")
    
#     with torch.no_grad():
#         for batch_x, batch_y in dataloader:
#             batch_x = batch_x.to(device)
            
#             start_event.record()
#             outputs = model(batch_x)
#             end_event.record()
            
#             torch.cuda.synchronize()  # Force execution sync on T4 core
            
#             batch_ms = start_event.elapsed_time(end_event)
#             # Normalize to microseconds (µs) per window
#             per_sample_us = (batch_ms * 1000) / batch_x.size(0)
#             latencies.append(per_sample_us)
            
#             _, predicted = outputs.max(1)
#             all_preds.extend(predicted.cpu().numpy())
#             all_targets.extend(batch_y.numpy())
            
#     macro_f1 = f1_score(all_targets, all_preds, average='macro')
#     print(f"Avg T4 Latency: {np.mean(latencies):.2f} µs | Macro-F1: {macro_f1:.4f}")
#     return np.mean(latencies), macro_f1

# # Execute this in Environment 1 (LSTM) and Environment 2 (MSNN)
# avg_lat, macro_f1 = evaluate_environment_latency(net, test_loader)


--- CELL 13 ---
import torch
import torch.nn as nn
import numpy as np

# ---------------------------------------------------------
# 1. Define the Standard LSTM Architecture
# ---------------------------------------------------------
class HFT_LSTM_Baseline(nn.Module):
    def __init__(self, input_size=288, hidden_size=128, num_layers=1, num_classes=3):
        super(HFT_LSTM_Baseline, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Batch_first=True expects input shape: (batch_size, time_steps, features)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3 if num_layers > 1 else 0)
        
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # x shape from your dataloader: (batch_size, 30, 288)
        
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  
        
        # out shape: (batch_size, time_steps, hidden_size)
        # We only want the output from the final time step (step 30)
        final_step_out = out[:, -1, :]
        
        final_step_out = self.drop(final_step_out)
        logits = self.fc(final_step_out)
        
        return logits

# ---------------------------------------------------------
# 2. Setup the Training Loop
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running LSTM on: {device}")

# Initialize the LSTM
lstm_net = HFT_LSTM_Baseline(input_size=288, hidden_size=128, num_layers=1, num_classes=3).to(device)

# We reuse your exact FocalLoss and class weights
optimizer_lstm = torch.optim.Adam(lstm_net.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler_lstm = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_lstm, T_max=40, eta_min=1e-6)

num_epochs = 40
best_lstm_acc = 0.0
patience = 15
patience_counter = 0

print("\n--- Starting LSTM Baseline Training ---")
for epoch in range(num_epochs):
    lstm_net.train()
    
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Data comes in as (batch, 30, 288)
        data = data.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).long()
        
        optimizer_lstm.zero_grad(set_to_none=True)
        
        # Forward pass (no time-loop needed, LSTM handles it internally)
        logits = lstm_net(data)
        loss_val = loss_fn(logits, targets) # Reusing your FocalLoss
        
        loss_val.backward()
        torch.nn.utils.clip_grad_norm_(lstm_net.parameters(), max_norm=1.0)
        optimizer_lstm.step()
        
    scheduler_lstm.step()
    
    # ---------------------------------------------------------
    # 3. Evaluation
    # ---------------------------------------------------------
    lstm_net.eval()
    test_losses, test_per_class = evaluate_epoch(eval_subset_loader, lstm_net, loss_fn, device)
    avg_test_loss = np.nanmean(test_losses)
    mean_test_acc = test_per_class.mean()
    
    if mean_test_acc > best_lstm_acc:
        best_lstm_acc = mean_test_acc
        patience_counter = 0
        torch.save(lstm_net.state_dict(), 'best_lstm_model.pt')
        print(f"Epoch {epoch}: NEW BEST LSTM | Mean Acc: {mean_test_acc*100:.2f}% | Loss: {avg_test_loss:.4f}")
    else:
        patience_counter += 1 
        print(f"Epoch {epoch}: Mean Acc: {mean_test_acc*100:.2f}% | Loss: {avg_test_loss:.4f} (Patience: {patience_counter}/{patience})")
        
        if patience_counter >= patience:
            print(f"Early stopping triggered for LSTM at epoch {epoch}.")
            break

# ---------------------------------------------------------
# 4. Final Evaluation against the full Test Set
# ---------------------------------------------------------
lstm_net.load_state_dict(torch.load('best_lstm_model.pt', map_location=device))
lstm_net.eval()

print("\n--- Final LSTM Evaluation on Full Test Set ---")
lstm_test_losses, lstm_test_per_class = evaluate_epoch(test_loader, lstm_net, loss_fn, device)
print(f"LSTM Test per-class accuracy: {lstm_test_per_class}")
print(f"LSTM Overall Balanced Accuracy: {lstm_test_per_class.mean()*100:.2f}%")

--- CELL 14 ---
# =========================================================
# LSTM Macro Metrics & Confusion Matrix
# =========================================================

# Ensure the LSTM model is in evaluation mode
lstm_net.eval()

print("="*60)
print("LSTM TRAIN RESULTS")
lstm_train_cm, lstm_train_report, lstm_train_macro_f1, lstm_train_weighted_f1 = evaluate_full(
    train_loader, lstm_net, device, split_name="LSTM Train"
)

print("="*60)
print("LSTM TEST RESULTS")
lstm_test_cm, lstm_test_report, lstm_test_macro_f1, lstm_test_weighted_f1 = evaluate_full(
    test_loader, lstm_net, device, split_name="LSTM Test"
)

--- CELL 15 ---

def weights(device, labels, smoothing=.15):
    class_counts = torch.bincount(labels.long())
    total_samples = len(labels)
    
    smoothed_c = class_counts.float() + (smoothing * total_samples / len(class_counts))
    dynamic_weights = total_samples / (len(class_counts) * smoothed_c)
    print(dynamic_weights)
    return dynamic_weights.to(device=device)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # weights
        self.gamma = gamma 
        self.reduction = reduction
    
    def forward(self, logits, targets):
        ce_loss = cross_entropy(logits, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma  
        loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        return loss.sum()

class STEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, num_levels):
        if num_levels is None or num_levels <= 1:
            return x
        scaled = x * (num_levels - 1)
        quantized = torch.round(scaled)
        return quantized / (num_levels - 1)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

# Hardware-aware RRAM crossbar with: 
# Differential conductance mapping 
# Stuck at fauls (SAF)
# Cycle-to-cyle variation
# Device-to-Device variation
# Retention Drift    
class MemristorCrossbar(nn.Module):
    def __init__(self, in_features, out_features, num_levels=16, saf_rate=0.03, noise_std=0.05, 
                d2d_std=0.05,
                drift_rate=0.002,
                force_ideal=False,
                force_noise_eval=False
                ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        if force_ideal:
            num_levels = 0
            saf_rate = 0.0
            noise_std = 0.0
            d2d_std = 0.0
            drift_rate = 0.0
            
        self.num_levels = num_levels
        self.noise_std = noise_std
        self.d2d_std = d2d_std
        self.force_ideal = force_ideal
        self.drift_rate = drift_rate
        self.force_noise_eval = force_noise_eval
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        print(f"params memristor: {num_levels }, {saf_rate}, {noise_std}, {d2d_std}, {drift_rate}")
        nn.init.normal_(self.weight, mean=0.0, std=0.05)
        nn.init.zeros_(self.bias)

      
        self.register_buffer("saf_pos_lrs", torch.rand(out_features, in_features) < (saf_rate / 2))
        self.register_buffer("saf_pos_hrs", torch.rand(out_features, in_features) < (saf_rate / 2))
        self.register_buffer("saf_neg_lrs", torch.rand(out_features, in_features) < (saf_rate / 2))
        self.register_buffer("saf_neg_hrs",torch.rand(out_features, in_features) < (saf_rate / 2))

       # Device-to-Device variation D2D
        self.register_buffer("d2d_pos", torch.clamp( 1.0 + torch.randn(out_features, in_features) * d2d_std, 0.7, 1.3))
        self.register_buffer("d2d_neg", torch.clamp( 1.0 + torch.randn(out_features, in_features) * d2d_std, 0.7, 1.3))

    def get_hardware_weights(self):

        # weights normalized
        weight_scale = torch.max(torch.abs(self.weight)).detach().clamp(min=1e-5)
        w_norm = torch.clamp(self.weight / weight_scale, -1.0, 1.0)
        g_pos = (w_norm + 1.0) / 2.0
        g_neg = (1.0 - w_norm) / 2.0
        
        g_pos *= self.d2d_pos
        g_neg *= self.d2d_neg
        
        g_pos = torch.clamp(g_pos, 0.0, 1.0)
        g_neg = torch.clamp(g_neg, 0.0, 1.0)

        # stuch at faults (SAF)
        g_pos = torch.where(self.saf_pos_lrs, torch.ones_like(g_pos), g_pos)
        g_pos = torch.where(self.saf_pos_hrs, torch.zeros_like(g_pos), g_pos)
        g_neg = torch.where(self.saf_neg_lrs, torch.ones_like(g_neg), g_neg)
        g_neg = torch.where(self.saf_neg_hrs, torch.zeros_like(g_neg), g_neg)
        # qunatization
        g_pos_q = STEQuantize.apply(g_pos, self.num_levels)
        g_neg_q = STEQuantize.apply(g_neg, self.num_levels)
       
        #C2C cycle to cycle
        if (self.training or self.force_noise_eval) and self.noise_std > 0:
            g_pos_q = torch.clamp(g_pos_q + torch.randn_like(g_pos_q) * self.noise_std, 0.0, 1.0)
            g_neg_q = torch.clamp(g_neg_q + torch.randn_like(g_neg_q) * self.noise_std, 0.0, 1.0)

        # retention drift
        if not self.training:
            drift = torch.exp(torch.full_like(g_pos_q, -self.drift_rate))
            g_pos_q *= drift
            g_neg_q *= drift
        w_hardware = (g_pos_q - g_neg_q) * weight_scale
        
        return w_hardware

    def forward(self, x):
        w_hardware = self.get_hardware_weights()
        return F.linear(x, w_hardware, self.bias)
    
    
class MSNN(nn.Module):
    """hardware-constrained MSNN and tracking spike metrics"""
    def __init__(self, num_inputs=288, num_hidden=128, num_outputs=3, beta=0.9, num_levels=16, saf_rate=0.03, noise_std=0.05, force_ideal=False, force_noise_eval=False):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=25)
        
        self.fc1 = MemristorCrossbar(num_inputs, num_hidden, num_levels, saf_rate, noise_std, force_ideal=force_ideal, force_noise_eval=force_noise_eval)
        self.lif1 = snn.Leaky(threshold=0.5, beta=beta, spike_grad=spike_grad, reset_mechanism="subtract")
        self.drop = nn.Dropout(0.3) 
        
        self.fc2 = MemristorCrossbar(num_hidden, num_outputs, num_levels, saf_rate, noise_std, force_ideal=force_ideal, force_noise_eval=force_noise_eval)
        self.lif2 = snn.Leaky(threshold=1.0, beta=beta, spike_grad=spike_grad, reset_mechanism="subtract")
        
        # keep track of spike counts
        self.spike_counts = {"layer1": 0.0, "layer2": 0.0, "total_steps": 0}

    def reset_spike_metrics(self):
        self.spike_counts = {"layer1": 0.0, "layer2": 0.0, "total_steps": 0}

    def forward(self, x):
        # adjust the input shame from (batch, time, featurres) ->> (time, batch, features)
        x = x.permute(1, 0, 2)
        batch_size = x.shape[1]
        time_steps = x.shape[0]
        
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        # input current is calculated in a single vectorized operation
        cur1_all = self.fc1(x)
        
        # Cache Layer 2 hardware matrix weights for this specific forward timeline execution block
        w_phys2 = self.fc2.get_hardware_weights()
        bias2 = self.fc2.bias
        
        for step in range(time_steps):
            spk1, mem1 = self.lif1(cur1_all[step], mem1)
            spk1_dropped = self.drop(spk1)
            
            # record layer 1 spike track
            if not self.training:
                self.spike_counts["layer1"] += spk1.detach().sum().item()
            
            cur2 = F.linear(spk1_dropped, w_phys2, bias2)
            spk2, mem2 = self.lif2(cur2, mem2)
            
            # record layer 2 spike track
            if not self.training:
                self.spike_counts["layer2"] += spk2.detach().sum().item()
                
        if not self.training:
            self.spike_counts["total_steps"] += (batch_size * time_steps)
            
        return mem2


net = MSNN(
    num_inputs=288, 
    num_hidden=128, 
    num_outputs=3, 
    beta=0.9, 
    num_levels=16, 
    saf_rate=0.03, 
    noise_std=0.05,
    force_ideal=False,
    force_noise_eval=True
).to(device)
print(net)


==============================
FILE: notebooks\preprocessing.ipynb
==============================
--- CELL 0 ---
import numpy as np
import torch
import torch.nn as nn
import numpy as np
import snntorch as snn
from snntorch import utils
from snntorch import spikegen, surrogate
import snntorch.functional as SF
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

# -------------- Parameters -----------------------
time_steps = 39512

class plot():
    def plotxy(
        series_list,
        index=None,
        title="",
        xlabel="time stamps",
        ylabel="",
        figsize=(14, 7),
        axhline=None,
        show_legend=True,
        disable_scietific_format = False,
    ):
        plt.figure(figsize=figsize), 
        
        for series  in series_list: 
            y_vals = np.asarray(series["y"])
            if index is not None:
                y_vals = y_vals[:index]
                
            plt.plot(
            y_vals,
            label=series.get("label"),
            color=series.get("color", "#0275D8"),
            linewidth=series.get("linewidth", 1.5),
            alpha=series.get("alpha", 1.0),
            linestyle=series.get("linestyle", "-"),)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if disable_scietific_format == True:
            plt.ticklabel_format(style="plain", useOffset=False, axis="both")
        if axhline is not None:
            plt.axhline(axhline, color="black", linestyle="--", alpha=0.5)

        if show_legend:
            plt.legend()
        plt.show()
        
        
        
    def plot_spikes(
        series_list,
        index=None,
        value = 1.0,
        title="",
        xlabel="time stamps",
        ylabel="Spikes",
        figsize=(14, 7),
        axhline=None,
        show_legend=True,
        disable_scietific_format=False,
        
    ):
        plt.figure(figsize=figsize), 
        
        for series in series_list:
            
            y_spikes = np.asarray(series["y"])
            if index is not None:
                y_spikes = y_spikes[:index]
            
            spike_point = np.atleast_1d(y_spikes)
            spike_point = np.where(y_spikes == value)[0]
            plt.vlines(
                spike_point,
                ymax=series.get("ymax", 0.5),
                ymin=series.get("ymin", 1.5),
                label=series.get("label"),
                color=series.get("color", "red"),
                linewidth=series.get("linewidth", 1.5),
                alpha=series.get("alpha", 1.0),
                linestyle=series.get("linestyle", "-"),
           )
            
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        if disable_scietific_format == True:
            plt.ticklabel_format(style="plain", useOffset=False, axis="both")
        if axhline is not None:
            plt.axhline(axhline, color="black", linestyle="--", alpha=0.5)

        if show_legend:
            plt.legend()
        plt.show()
        

def test_accuracy(data_loader, net, num_steps, device, population_code=False, num_classes=False):
  with torch.no_grad():
    total = 0
    acc = 0
    net.eval()

    data_loader = iter(data_loader)
    for data, targets in data_loader:
      data = data.to(device)
      targets = targets.to(device)
      utils.reset(net)
      spk_rec, _ = net(data)

      if population_code:
        acc += SF.accuracy_rate(spk_rec.unsqueeze(0), targets, population_code=True, num_classes=10) * spk_rec.size(1)
      else:
        acc += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)

      total += spk_rec.size(1)

  return acc/total


        
plot_xy = plot.plotxy
plot_spike  = plot.plot_spikes



--- CELL 1 ---
"""
Two figures for the paper:

  Part A: I-V pinched hysteresis loop for a SINGLE physical memristor.
          Uses the standard HP TiO2 thin-film memristor model
          (Strukov, Snider, Stewart & Williams, "The missing memristor found",
          Nature 453, 80-83, 2008), with Joglekar's window function to keep
          the state variable bounded. This is a device-physics model -- it is
          NOT derived from your MemristorCrossbar code, since that code has no
          voltage/current dynamics. Cite Strukov et al. 2008 for this figure.

  Part B: Memristance variance of a SINGLE RRAM cell as modeled by YOUR actual
          MemristorCrossbar (D2D + C2C + quantization), converted from
          normalized conductance into an illustrative physical resistance
          range. This one IS your model -- cite yourself / your Methods
          section for this figure, not Strukov et al.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "Computer Modern Roman"]


# ======================================================================
# PART A: I-V pinched hysteresis loop (HP memristor device model)
# ======================================================================

def hp_memristor_iv(V0=1.0, freq=1.0, R_on=100.0, R_off=16000.0, D=10e-9,
                     k=15000.0, p=5, x0=0.3, n_cycles=1, steps_per_cycle=4000):
    """
    Simulates a single HP TiO2 memristor driven by V(t) = V0*sin(2*pi*f*t).

    State variable x in [0,1] represents the normalized width of the doped
    (low-resistance) region. Memristance (resistance) is a linear
    interpolation between R_on (fully doped) and R_off (fully undoped):

        M(x) = R_on * x + R_off * (1 - x)

    State dynamics (voltage-controlled, with Joglekar window f(x) to prevent
    x from leaving [0,1] and to model nonlinear ion-drift boundary effects):

        dx/dt = k * i(t) * f(x),   f(x) = 1 - (2x - 1)**(2p)
        i(t)  = V(t) / M(x)

    NOTE: k folds together the physical constants (mu_v * R_on / D**2) from
    the original Strukov et al. formulation. We report it directly (rather
    than mu_v/D separately) because with textbook SI values (mu_v~1e-14
    m^2/V/s, D~10nm) the ion-drift timescale is far faster than a human-scale
    excitation frequency and the loop collapses to a straight line; k=15000
    here is chosen (consistent with how most tutorial/simulation
    implementations do it) purely so the loop is visible at freq=0.5-2 Hz --
    state this in your figure caption rather than presenting k as a measured
    physical constant.
    """
    T = n_cycles / freq
    dt = T / (n_cycles * steps_per_cycle)
    n_steps = n_cycles * steps_per_cycle

    t = np.linspace(0, T, n_steps)
    V = V0 * np.sin(2 * np.pi * freq * t)

    x = np.zeros(n_steps)
    x[0] = x0
    I = np.zeros(n_steps)

    for n in range(n_steps - 1):
        M = R_on * x[n] + R_off * (1 - x[n])
        I[n] = V[n] / M
        f_x = 1 - (2 * x[n] - 1) ** (2 * p)          # Joglekar window
        dx = k * I[n] * f_x * dt
        x[n + 1] = np.clip(x[n] + dx, 0.0, 1.0)

    M_last = R_on * x[-1] + R_off * (1 - x[-1])
    I[-1] = V[-1] / M_last

    return t, V, I, x


def plot_iv_curve(save_path="memristor_iv_curve1.pdf"):
    fig, ax = plt.subplots(figsize=(6, 5.5), dpi=600)

    # Plot at a couple of frequencies to show the classic result:
    # the pinched loop narrows toward a straight line (linear resistor)
    # as frequency increases -- a well-known memristor fingerprint.
    for freq, color, label in [(0.5, "#005b96", "f = 0.5 Hz"),
                                (2.0, "#d95f02", "f = 2.0 Hz")]:
        t, V, I, x = hp_memristor_iv(V0=1.0, freq=freq)
        ax.plot(V, I * 1000, color=color, linewidth=2, label=label)  # mA for readability

    ax.axhline(0, color="gray", linewidth=0.8)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("Voltage (V)", fontweight="bold")
    ax.set_ylabel("Current (mA)", fontweight="bold")
    ax.set_title("Pinched I-V Hysteresis Loop\n(single HP TiO$_2$ memristor model)",
                  fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(frameon=True, edgecolor="black")
    fig.tight_layout()
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    print(f"Saved I-V curve to {save_path}")
    plt.close(fig)


# ======================================================================
# PART A2: Memristance M(t) under a PULSED voltage (the direct "memory" plot)
# ======================================================================

def pulse_train(t, pulse_amp=1.0, pulse_width=0.05, gap=0.05, pattern=None):
    """Builds a square-wave voltage pulse train. `pattern` is a list of
    (sign, n_pulses) pairs, e.g. [(+1, 4), (-1, 4)] = 4 SET pulses then
    4 RESET pulses. Each pulse is `pulse_width` seconds high, then `gap`
    seconds at 0V before the next pulse."""
    if pattern is None:
        pattern = [(+1, 4), (-1, 4)]
    V = np.zeros_like(t)
    period = pulse_width + gap
    idx = 0
    time_cursor = 0.0
    for sign, n in pattern:
        for _ in range(n):
            mask = (t >= time_cursor) & (t < time_cursor + pulse_width)
            V[mask] = sign * pulse_amp
            time_cursor += period
    return V


def simulate_memristance_under_pulses(R_on=100.0, R_off=16000.0, k=15000.0, p=5,
                                       x0=0.5, pulse_amp=1.0, pulse_width=0.05, gap=0.05,
                                       pattern=None, steps_per_sec=20000):
    if pattern is None:
        pattern = [(+1, 4), (-1, 4)]
    n_pulses = sum(n for _, n in pattern)
    T = n_pulses * (pulse_width + gap)
    n_steps = int(T * steps_per_sec)
    t = np.linspace(0, T, n_steps)
    dt = t[1] - t[0]
    V = pulse_train(t, pulse_amp, pulse_width, gap, pattern)

    x = np.zeros(n_steps)
    x[0] = x0
    M = np.zeros(n_steps)

    for n in range(n_steps - 1):
        M[n] = R_on * x[n] + R_off * (1 - x[n])
        I_n = V[n] / M[n]
        f_x = 1 - (2 * x[n] - 1) ** (2 * p)
        dx = k * I_n * f_x * dt
        x[n + 1] = np.clip(x[n] + dx, 0.0, 1.0)
    M[-1] = R_on * x[-1] + R_off * (1 - x[-1])

    return t, V, M, x


def plot_memristance_vs_time(save_path="memristance_vs_time1.pdf"):
    t, V, M, x = simulate_memristance_under_pulses(
        # A HIGH-FREQUENCY pulse train (many short pulses) instead of a few
        # large ones. Physically this is the same SET/RESET switching, just
        # sampled finely enough that the underlying staircase (each individual
        # pulse nudging M by a small amount) blends into a visually smooth
        # curve -- useful when you want to emphasize the overall trend rather
        # than individual switching events.
        pattern=[(+1, 150), (-1, 150)],
        pulse_amp=0.6, pulse_width=0.0008, gap=0.0013,
    )

    fig, ax2 = plt.subplots(figsize=(7, 4.5), dpi=600)

    ax2.plot(t, M / 1e3, color="#005b96", linewidth=2)
    ax2.set_xlabel("Time (s)", fontweight="bold")
    ax2.set_ylabel("Memristance (k$\\Omega$)", fontweight="bold")
    ax2.set_title("Memristance Response to a Programming Pulse Train\n"
                   "(150 SET + 150 RESET pulses)", fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.axvspan(t[0], t[len(t)//2], color="#005b96", alpha=0.05)
    ax2.axvspan(t[len(t)//2], t[-1], color="#d95f02", alpha=0.05)
    ax2.text(t[len(t)//4], M.max()/1e3, "SET pulses", ha="center", va="bottom", fontsize=9, color="#005b96")
    ax2.text(t[3*len(t)//4], M.max()/1e3, "RESET pulses", ha="center", va="bottom", fontsize=9, color="#d95f02")

    fig.tight_layout()
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    print(f"Saved memristance-vs-time plot to {save_path}")
    print(f"  M starts at {M[0]/1e3:.2f} k-ohm, "
          f"drops to {M.min()/1e3:.2f} k-ohm after SET pulses, "
          f"rises to {M.max()/1e3:.2f} k-ohm after RESET pulses")
    plt.close(fig)


# ======================================================================
# PART B: Memristance variance of a single RRAM cell (YOUR crossbar model)
# ======================================================================

# --- copy of your actual model, so this plot reflects your real code ---
class STEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, num_levels):
        if num_levels is None or num_levels <= 1:
            return x
        scaled = x * (num_levels - 1)
        return torch.round(scaled) / (num_levels - 1)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def sample_single_device_conductance(g_programmed, num_levels=16, noise_std=0.05,
                                      d2d_factor=1.0, n_reads=2000):
    """
    Simulates repeatedly READING one already-programmed RRAM device
    (fixed target conductance g_programmed, fixed D2D factor for THIS
    device) across many cycles, exactly as your MemristorCrossbar's
    quantization + C2C noise steps do.
    """
    g = torch.full((n_reads,), g_programmed) * d2d_factor
    g = torch.clamp(g, 0.0, 1.0)
    g_q = STEQuantize.apply(g, num_levels)
    g_noisy = torch.clamp(g_q + torch.randn(n_reads) * noise_std, 0.0, 1.0)
    return g_noisy.numpy()


def plot_memristance_variance(save_path="memristance_variance1.pdf",
                               R_on=1e3, R_off=1e5,
                               g_programmed=0.5, num_levels=16, noise_std=0.05):
    """
    Converts normalized conductance g in [0,1] to an illustrative physical
    resistance range [R_on, R_off] via M = R_off - g*(R_off - R_on), so the
    plot reads in Ohms. R_on/R_off here are ASSUMED illustrative values --
    replace with your actual device's datasheet/literature values before
    using this in the paper.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=600)

    # (1) Distribution across repeated reads of ONE device (C2C + quantization only)
    g_single = sample_single_device_conductance(g_programmed, num_levels, noise_std, d2d_factor=1.0)
    M_single = R_off - g_single * (R_off - R_on)

    axes[0].hist(M_single / 1e3, bins=40, color="#005b96", edgecolor="black", alpha=0.85)
    axes[0].set_xlabel("Memristance (k$\\Omega$)", fontweight="bold")
    axes[0].set_ylabel("Count (out of 2000 reads)", fontweight="bold")
    axes[0].set_title("(a) Single device,\nrepeated reads (C2C noise)", fontweight="bold")
    axes[0].grid(True, linestyle=":", alpha=0.5)

    # (2) Distribution across many DIFFERENT devices programmed to the same target (D2D variation)
    torch.manual_seed(0)
    d2d_factors = torch.clamp(1.0 + torch.randn(2000) * 0.05, 0.7, 1.3).numpy()
    g_pop = np.clip(g_programmed * d2d_factors, 0.0, 1.0)
    M_pop = R_off - g_pop * (R_off - R_on)

    axes[1].hist(M_pop / 1e3, bins=40, color="#d95f02", edgecolor="black", alpha=0.85)
    axes[1].set_xlabel("Memristance (k$\\Omega$)", fontweight="bold")
    axes[1].set_ylabel("Count (out of 2000 devices)", fontweight="bold")
    axes[1].set_title("(b) Many devices,\nsame target (D2D variation)", fontweight="bold")
    axes[1].grid(True, linestyle=":", alpha=0.5)

    fig.suptitle(f"Memristance variance around target conductance g={g_programmed}",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    print(f"Saved memristance variance plot to {save_path}")
    print(f"  Single-device (C2C) std: {M_single.std()/1e3:.3f} k-ohm  "
          f"(mean {M_single.mean()/1e3:.3f} k-ohm)")
    print(f"  Cross-device (D2D)  std: {M_pop.std()/1e3:.3f} k-ohm  "
          f"(mean {M_pop.mean()/1e3:.3f} k-ohm)")
    plt.close(fig)


if __name__ == "__main__":
    plot_iv_curve()
    plot_memristance_vs_time()
    plot_memristance_variance()

--- CELL 2 ---
### Setup the training Data
# -------------- Parameters -----------------------
time_steps = 39512
train_file = "data/published/BenchmarkDatasets/BenchmarkDatasets/BenchmarkDatasets/NoAuction/NoAuction_MinMax/NoAuction_MinMax_Training/Train_Dst_NoAuction_MinMax_CF_1.txt"

data = np.loadtxt(train_file)

ask_price = data[0, :]
bid_price = data[2, :]
mid_price = (ask_price + bid_price) / 2
features  = data[:144, :].T 
labels    = data[146, :].T - 1

plot_xy(
    series_list=[
        {"y": mid_price, "label": "Positive Spikes"},
    ],
    index=time_steps,
    title="Positive Spikes"
)


features_tensor = torch.from_numpy(features).float()
labels_tensor = torch.from_numpy(labels).float()


spike = spikegen.delta(features_tensor, threshold=0.01, padding=True, off_spike=True)
spikes_pos = torch.where(spike > 0, spike, torch.zeros_like(spike))
spikes_neg = torch.where(spike < 0, torch.abs(spike), torch.zeros_like(spike))

plot_spike(
    series_list=[
        {"y": spikes_pos, "label": "Positive Spikes"},
    ],
    index=time_steps,
    title="Positive Spikes"
)
plot_spike(
    series_list=[
        {"y": spikes_neg, "label": "Negative Spikes", "color": "blue"},
    ],
    index=time_steps,
    title="Negative Spikes"
)

Z = features_tensor[:time_steps, :].numpy()

X, Y = np.meshgrid(np.arange(time_steps), np.arange(144))
Z = Z.T

fig = plt.figure(figsize=(14, 9))
ax = fig.add_subplot(111, projection="3d")

surface = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="skyblue", linewidth=.2)
ax.set_xlabel('Time Stamps')
ax.set_ylabel("Feature Index")
ax.set_zlabel("Feature Value (Normalized)")
# ax.view_init(elev=0, azim=180)
fig.colorbar(surface, shrink=.5, aspect=5)

plt.show()

--- CELL 3 ---
### Setup the Testing Data
test_file  = "data/published/BenchmarkDatasets/BenchmarkDatasets/BenchmarkDatasets/NoAuction/NoAuction_MinMax/NoAuction_MinMax_Testing/Test_Dst_NoAuction_MinMax_CF_1.txt"

test_data = np.loadtxt(test_file)
features_test  = test_data[:144, :].T 
labels_test    = test_data[146, :].T - 1



features_tensor_test = torch.from_numpy(features_test).float()
labels_tensor_test = torch.from_numpy(labels_test).long()


spike_test = spikegen.delta(features_tensor_test, threshold=0.01, padding=True, off_spike=True)
spikes_pos_test = torch.where(spike_test > 0, spike_test, torch.zeros_like(spike_test))
spikes_neg_test = torch.where(spike_test < 0, torch.abs(spike_test), torch.zeros_like(spike_test))

 
test_inputs = torch.cat((spikes_pos_test, spikes_neg_test), dim=1)
test_dataset = TensorDataset(test_inputs, labels_tensor_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

inputs_train = torch.cat((spikes_pos, spikes_neg), dim=1)

dataset = TensorDataset(inputs_train, labels_tensor)

batch_size = 16
# shuffle is set to false to keep the time-series sequence order (took me 3 days to debug ughhh)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

print(f"dataloader {len(train_loader)}")

--- CELL 4 ---
# Parameters of the network
num_inputs = 288
num_hidden = 128
num_outputs = 3
beta = .9
num_seps = 25

spike_grad = surrogate.fast_sigmoid(slope=25)

class HFT_Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc1  = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(threshold=.5,  beta=beta, spike_grad=spike_grad)
        self.fc2  = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(threshold=.5, beta=beta, spike_grad=spike_grad)
        
    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        spk2_rec = []
        mem2_rec = []
        
        for step in range(x.shape[0]):
            cur1 = self.fc1(x[step])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)
            
        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = HFT_Net().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=2e-4, betas=(0.9, 0.999))
class_weights = torch.tensor([2.0, 0.5, 2.0]).to(device)
loss_fn = SF.ce_rate_loss(weight=class_weights)

num_epochs = 25
loss_hist = []
spk_re = []
for epoch in range(num_epochs):
    net.train()
    
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device).long()
        
        data_seq = data.unsqueeze(1) 
        target_seq = targets[-1].unsqueeze(0)
        
        spk_rec, mem_rec = net(data_seq)
        
        loss_val = loss_fn(spk_rec, target_seq)
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        
        loss_hist.append(loss_val.item())
        
         
        if batch_idx % 50 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx} \tTraining Loss: {loss_val.item():.4f}")
            


--- CELL 5 ---
import matplotlib.pyplot as plt
import numpy as np

# 1. Calculate how many batches were in a single epoch
batches_per_epoch = len(train_loader)
total_epochs = len(loss_hist) // batches_per_epoch

# 2. Reshape the flat loss list into a 2D array: (Epochs, Batches)
# and calculate the average loss per epoch
loss_array = np.array(loss_hist)[:total_epochs * batches_per_epoch].reshape(total_epochs, batches_per_epoch)

epoch_losses = loss_array.mean(axis=1)

# 3. Create the X-axis values (Epochs 1 to N)
epochs_x = np.arange(1, total_epochs + 1)
batches_x = np.linspace(1, total_epochs, len(loss_hist))

# 4. Plotting
plt.figure(figsize=(10, 6))

# Plot the raw batch loss in the background (faded) to show the actual LOB volatility
plt.plot(batches_x, loss_hist, color='skyblue', alpha=0.3, label='Batch Loss')

# Plot the clean, averaged epoch loss on top
plt.plot(epochs_x, epoch_losses, color='darkblue', linewidth=2.5, marker='o', label='Epoch Average Loss')

# Formatting for a research paper
plt.title("SNN Training Loss over Epochs", fontsize=14, fontweight='bold')
plt.xlabel("Epochs", fontsize=12)
plt.ylabel("Cross Entropy Rate Loss", fontsize=12)
plt.xticks(epochs_x) # Force the X-axis to show whole numbers for epochs
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc="upper right", fontsize=11)
plt.tight_layout()

plt.show()


 

==============================
FILE: notebooks\training.ipynb
==============================
--- CELL 0 ---
%pip install snntorch
import torch
import torch.nn as nn
import glob
from torch.nn.functional import cross_entropy
import torch.nn.functional as F
import snntorch as snn
from snntorch import spikegen
from snntorch import surrogate
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import gc
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


--- CELL 1 ---

class LOBDayDataset(Dataset):
    """Loads a single day file from the FI-2010 dataset without cross-day leakage"""
    def __init__(self, filepath, window_size=30, stride=5, delta_threshold=0.005):
        
        data = np.loadtxt(filepath)
        features = data[:144, :].T  # features
        labels = data[146, :].T - 1 # labels (we map them from {1, 2, 3...} to {0, 1, 2, 3.....})
        del data

        features_tensor = torch.from_numpy(features).float()
        labels_tensor = torch.from_numpy(labels).long()
        del features, labels

        # spike encoding using delta modulation see https://snntorch.readthedocs.io/en/latest/tutorials/legacy/tutorial_1_old.html#delta-modulation
        spike = spikegen.delta(features_tensor, threshold=delta_threshold, padding=True, off_spike=True)
        spikes_pos = torch.where(spike > 0, spike, torch.zeros_like(spike))
        spikes_neg = torch.where(spike < 0, torch.abs(spike), torch.zeros_like(spike))
        del spike, features_tensor

        # inputs <= spike_pos + spike_neg
        inputs = torch.cat((spikes_pos, spikes_neg), dim=1).half() 
        del spikes_pos, spikes_neg

        
        
        # we slice the inputs into sequence windows 
        self.X, self.Y = self._create_sequences(inputs, labels_tensor, window_size, stride)
        del inputs, labels_tensor
        gc.collect()
    
    # create sequence of the inputs based on window_size and stride
    def _create_sequences(self, inputs, labels, window_size, stride):
        X, Y = [], []
        for i in range(0, len(inputs) - window_size, stride):
            X.append(inputs[i : i + window_size])
            Y.append(labels[i + window_size - 1])
        return torch.stack(X), torch.stack(Y)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx].float(), self.Y[idx]

# this function implements the paper exact anchred cross-validation protocol
# example: for a fold of 2 (fold_idx = 2)
# we train the MSNN on day 1 + day 2 and test the MSNN on day 3 data  
def get_anchored_fold_loaders(fold_idx, train_files, test_files, batch_size=256):
    print(f"\n***  fold == {fold_idx + 1} ***")
    
    # identify the days for the training 
    active_train_files = train_files[:fold_idx + 1]
    # The test target is always the next sequential day window
    active_test_file = test_files[fold_idx + 1]
    
    print(f"training files are up to: {active_train_files[-1].split('/')[-1]}")
    print(f"testing file: {active_test_file.split('/')[-1]}")
    
    train_datasets = [LOBDayDataset(f) for f in active_train_files]
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)

    test_dataset = LOBDayDataset(active_test_file)
    
    
    
    # set shuffle=False to preserve internal temporal sequence blocks
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    return train_loader, test_loader, train_dataset

--- CELL 2 ---

def weights(device, labels, smoothing=.15):
    class_counts = torch.bincount(labels.long())
    total_samples = len(labels)
    
    smoothed_c = class_counts.float() + (smoothing * total_samples / len(class_counts))
    dynamic_weights = total_samples / (len(class_counts) * smoothed_c)
    print(dynamic_weights)
    return dynamic_weights.to(device=device)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # weights
        self.gamma = gamma 
        self.reduction = reduction
    
    def forward(self, logits, targets):
        ce_loss = cross_entropy(logits, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma  
        loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        return loss.sum()

class STEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, num_levels):
        if num_levels is None or num_levels <= 1:
            return x
        scaled = x * (num_levels - 1)
        quantized = torch.round(scaled)
        return quantized / (num_levels - 1)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

# Hardware-aware RRAM crossbar with: 
# Differential conductance mapping 
# Stuck at fauls (SAF)
# Cycle-to-cyle variation
# Device-to-Device variation
# Retention Drift    
class MemristorCrossbar(nn.Module):
    def __init__(self, in_features, out_features, num_levels=16, saf_rate=0.03, noise_std=0.05, 
                d2d_std=0.05,
                drift_rate=0.002,
                force_ideal=False,
                force_noise_eval=False
                ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        if force_ideal:
            num_levels = 0
            saf_rate = 0.0
            noise_std = 0.0
            d2d_std = 0.0
            drift_rate = 0.0
            
        self.num_levels = num_levels
        self.noise_std = noise_std
        self.d2d_std = d2d_std
        self.force_ideal = force_ideal
        self.drift_rate = drift_rate
        self.force_noise_eval = force_noise_eval
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        print(f"params memristor: {num_levels }, {saf_rate}, {noise_std}, {d2d_std}, {drift_rate}")
        nn.init.normal_(self.weight, mean=0.0, std=0.05)
        nn.init.zeros_(self.bias)

      
        self.register_buffer("saf_pos_lrs", torch.rand(out_features, in_features) < (saf_rate / 2))  # stuck-at-low resistance state ; stuck at one
        self.register_buffer("saf_pos_hrs", torch.rand(out_features, in_features) < (saf_rate / 2))   # stuck-at-high resistance state ; stuck at 0
        self.register_buffer("saf_neg_lrs", torch.rand(out_features, in_features) < (saf_rate / 2))
        self.register_buffer("saf_neg_hrs",torch.rand(out_features, in_features) < (saf_rate / 2))

       # Device-to-Device variation D2D
        self.register_buffer("d2d_pos", torch.clamp( 1.0 + torch.randn(out_features, in_features) * d2d_std, 0.7, 1.3))
        self.register_buffer("d2d_neg", torch.clamp( 1.0 + torch.randn(out_features, in_features) * d2d_std, 0.7, 1.3))

    def get_hardware_weights(self):

        # weights normalized
        weight_scale = torch.max(torch.abs(self.weight)).detach().clamp(min=1e-5)
        w_norm = torch.clamp(self.weight / weight_scale, -1.0, 1.0)
        g_pos = (w_norm + 1.0) / 2.0
        g_neg = (1.0 - w_norm) / 2.0
        
        g_pos *= self.d2d_pos
        g_neg *= self.d2d_neg
        
        g_pos = torch.clamp(g_pos, 0.0, 1.0)
        g_neg = torch.clamp(g_neg, 0.0, 1.0)

        # stuch at faults (SAF)
        g_pos = torch.where(self.saf_pos_lrs, torch.ones_like(g_pos), g_pos)
        g_pos = torch.where(self.saf_pos_hrs, torch.zeros_like(g_pos), g_pos)
        g_neg = torch.where(self.saf_neg_lrs, torch.ones_like(g_neg), g_neg)
        g_neg = torch.where(self.saf_neg_hrs, torch.zeros_like(g_neg), g_neg)
        # qunatization
        g_pos_q = STEQuantize.apply(g_pos, self.num_levels)
        g_neg_q = STEQuantize.apply(g_neg, self.num_levels)
       
        #C2C cycle to cycle
        if (self.training or self.force_noise_eval) and self.noise_std > 0:
            g_pos_q = torch.clamp(g_pos_q + torch.randn_like(g_pos_q) * self.noise_std, 0.0, 1.0)
            g_neg_q = torch.clamp(g_neg_q + torch.randn_like(g_neg_q) * self.noise_std, 0.0, 1.0)

        # retention drift
        if not self.training:
            drift = torch.exp(torch.full_like(g_pos_q, -self.drift_rate))
            g_pos_q *= drift
            g_neg_q *= drift
        w_hardware = (g_pos_q - g_neg_q) * weight_scale
        
        return w_hardware

    def forward(self, x):
        w_hardware = self.get_hardware_weights()
        return F.linear(x, w_hardware, self.bias)
    
    
class MSNN(nn.Module):
    """hardware-constrained MSNN and tracking spike metrics"""
    def __init__(self, num_inputs=288, num_hidden=128, num_outputs=3, beta=0.9, num_levels=16, saf_rate=0.03, noise_std=0.05, force_ideal=False, force_noise_eval=False):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=25)
        
        self.fc1 = MemristorCrossbar(num_inputs, num_hidden, num_levels, saf_rate, noise_std, force_ideal=force_ideal, force_noise_eval=force_noise_eval)
        self.lif1 = snn.Leaky(threshold=0.5, beta=beta, spike_grad=spike_grad, reset_mechanism="subtract")
        self.drop = nn.Dropout(0.3) 
        
        self.fc2 = MemristorCrossbar(num_hidden, num_outputs, num_levels, saf_rate, noise_std, force_ideal=force_ideal, force_noise_eval=force_noise_eval)
        self.lif2 = snn.Leaky(threshold=1.0, beta=beta, spike_grad=spike_grad, reset_mechanism="subtract")
        
        # keep track of spike counts
        self.spike_counts = {"layer1": 0.0, "layer2": 0.0, "total_steps": 0}

    def reset_spike_metrics(self):
        self.spike_counts = {"layer1": 0.0, "layer2": 0.0, "total_steps": 0}

    def forward(self, x):
        # adjust the input shame from (batch, time, featurres) ->> (time, batch, features)
        x = x.permute(1, 0, 2)
        batch_size = x.shape[1]
        time_steps = x.shape[0]
        
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        # input current is calculated in a single vectorized operation
        cur1_all = self.fc1(x)
        
        # Cache Layer 2 hardware matrix weights for this specific forward timeline execution block
        w_phys2 = self.fc2.get_hardware_weights()
        bias2 = self.fc2.bias
        
        for step in range(time_steps):
            spk1, mem1 = self.lif1(cur1_all[step], mem1)
            spk1_dropped = self.drop(spk1)
            
            # record layer 1 spike track
            if not self.training:
                self.spike_counts["layer1"] += spk1.detach().sum().item()
            
            cur2 = F.linear(spk1_dropped, w_phys2, bias2)
            spk2, mem2 = self.lif2(cur2, mem2)
            
            # record layer 2 spike track
            if not self.training:
                self.spike_counts["layer2"] += spk2.detach().sum().item()
                
        if not self.training:
            self.spike_counts["total_steps"] += (batch_size * time_steps)
            
        return mem2



--- CELL 3 ---
def compute_hardware_metrics(data_loader, net, device, window_size=30):
    """evaluates the model and computes hardware metrics etc...
        window_size is a must, unless if its 30
    """
    net.eval()
    net.reset_spike_metrics()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, targets in data_loader:
            data = data.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).long()
            logits = net(data)
            preds = logits.argmax(dim=1)
            
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
    # calculates sparsity metrics
    total_samples_processed = net.spike_counts["total_steps"] / window_size  
    total_neuron_steps_l1 = net.spike_counts["total_steps"] * net.fc1.out_features
    total_neuron_steps_l2 = net.spike_counts["total_steps"] * net.fc2.out_features
    
    l1_sparsity = net.spike_counts["layer1"] / max(1, total_neuron_steps_l1)
    l2_sparsity = net.spike_counts["layer2"] / max(1, total_neuron_steps_l2)
    
    accuracy = correct / total
    
    print("\n*** extracted hardware metrics ***")
    print(f"accuracy: {accuracy*100:.2f}%")
    print(f"layer 1 firing density : {l1_sparsity*100:.3f}% spikes/neuron/step")
    print(f"layer 2 firing density : {l2_sparsity*100:.3f}% spikes/neuron/step")
    print(f"total spikes emitted per sequence window ({window_size}): {(net.spike_counts['layer1'] + net.spike_counts['layer2']) / max(1, total_samples_processed):.2f}")
    
    return {
        "accuracy": accuracy,
        "l1_sparsity": l1_sparsity,
        "l2_sparsity": l2_sparsity,
    }


--- CELL 4 ---
import torch
import numpy as np
import os

num_epochs = 20  #significantly lower as anchored dataset scale up easily
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

# update with ur own paths 
train_files = sorted(glob.glob(
    "/kaggle/input/datasets/firmwired/fi-2010/data/published/BenchmarkDatasets/BenchmarkDatasets/BenchmarkDatasets/NoAuction/NoAuction_MinMax/NoAuction_MinMax_Training/Train_Dst_NoAuction_MinMax_CF_*.txt"
))
test_files = sorted(glob.glob(
    "/kaggle/input/datasets/firmwired/fi-2010/data/published/BenchmarkDatasets/BenchmarkDatasets/BenchmarkDatasets/NoAuction/NoAuction_MinMax/NoAuction_MinMax_Testing/Test_Dst_NoAuction_MinMax_CF_*.txt"
))

output_dir = '/kaggle/working/MSNN/noisy'
os.makedirs(output_dir, exist_ok=True)
torch.manual_seed(123)
fold_summary_metrics = []

net = MSNN(
    num_inputs=288, 
    num_hidden=128, 
    num_outputs=3, 
    beta=0.9, 
    num_levels=16, 
    saf_rate=0.03, 
    noise_std=0.05,
    force_ideal=False,
    force_noise_eval=True
).to(device)


# loop across all 9 files
for fold_idx in range(8):
    print("\n" + "*"*70)
    print(f"* anchord fold  {fold_idx + 1} / 9 *")
    print("*"*70)
    
    # get the needed fold time (based on fold_idx)
    train_loader, test_loader, fold_train_dataset = get_anchored_fold_loaders(
        fold_idx=fold_idx, 
        train_files=train_files, 
        test_files=test_files, 
        batch_size=256
    )
    
    if isinstance(fold_train_dataset, torch.utils.data.ConcatDataset):
        active_labels = torch.cat([ds.Y for ds in fold_train_dataset.datasets])
    else:
        active_labels = fold_train_dataset.Y
        
    w = weights(device=device, labels=active_labels)
    loss_fn = FocalLoss(alpha=w, gamma=2)
    
   
    num_epochs_per_fold = 10  
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs_per_fold, eta_min=1e-6)
    
    best_fold_acc = 0.0
    patience_counter = 0
    patience = 4
    
    # training loop
    for epoch in range(num_epochs_per_fold):
        net.train()
        epoch_loss = 0.0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).long()
            
            optimizer.zero_grad(set_to_none=True)
            logits = net(data)
            loss_val = loss_fn(logits, targets)
            
            loss_val.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss_val.item()
            
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        
        net.eval()
        with torch.no_grad():
            val_correct = 0
            val_total = 0
            for val_data, val_targets in test_loader:
                val_data = val_data.to(device, non_blocking=True)
                val_targets = val_targets.to(device, non_blocking=True).long()
                
                val_preds = net(val_data).argmax(dim=1)
                val_correct += (val_preds == val_targets).sum().item()
                val_total += val_targets.size(0)
            current_val_acc = val_correct / max(1, val_total)
            
        print(f"Fold {fold_idx+1} | Epoch [{epoch+1}/{num_epochs_per_fold}] | Loss: {avg_loss:.4f} | Current Test Acc: {current_val_acc*100:.2f}%")
        
        if current_val_acc > best_fold_acc:
            best_fold_acc = current_val_acc
            patience_counter = 0
            torch.save(net.state_dict(), os.path.join(output_dir, f'model_{fold_idx+1}.pt'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("↳ Early stopping fold fine-tuning profile.")
                break
                
    # 4. Extract Final Hardware Profiling numbers for this fold's configuration
    print(f"\n--- compiling hardware metrics for fold: {fold_idx + 1} ---") 
    checkpoint_file = os.path.join(output_dir, f'model_{fold_idx+1}.pt')
    net.load_state_dict(torch.load(checkpoint_file, map_location=device, weights_only=True))
    
    total_input_sparsity = 0.0
    num_batches = 0
    with torch.no_grad():
        for val_data, _ in test_loader:
            val_data = val_data.to(device, non_blocking=True)
            batch_density = val_data.float().mean().item()             
            total_input_sparsity += batch_density
            num_batches += 1
            
    fold_input_sparsity = total_input_sparsity / num_batches
    
    fold_profile = compute_hardware_metrics(test_loader, net, device)
    fold_summary_metrics.append({
        "fold": fold_idx + 1,
        "accuracy": fold_profile["accuracy"],
        "l1_sparsity": fold_profile["l1_sparsity"],
        "l2_sparsity": fold_profile["l2_sparsity"],
        "input_sparsity": fold_input_sparsity
    })

# 5. Output Summary Results Block for Paper Generation
print("\n" + "="*60)
print("final report:::")
print("="*60)
print(f"{'Fold':<6} | {'Test Accuracy':<15} | {'L1 Spike Density':<18} | {'L2 Spike Density':<18}")
print("-"*60)
for entry in fold_summary_metrics:
  print(f"{entry['fold']:<6} | "
          f"{entry['accuracy']*100:<14.2f}% | "
          f"{entry['input_sparsity']*100:<14.4f}% | "
          f"{entry['l1_sparsity']*100:<17.4f}% | "
          f"{entry['l2_sparsity']*100:<17.4f}%")
print("-"*60)

--- CELL 5 ---
output_dir_ideal = '/kaggle/working/MSNN/ideal'
os.makedirs(output_dir_ideal, exist_ok=True)

torch.manual_seed(123)
net_ideal = MSNN(
    num_inputs=288, 
    num_hidden=128, 
    num_outputs=3, 
    beta=0.9, 
    force_ideal=True
).to(device)
# net_ideal.force_noise_eval = True
fold_summary_metrics_ideal = []
# loop across all 9 files
for fold_idx in range(8):
    print("\n" + "*"*70)
    print(f"* anchord fold  {fold_idx + 1} / 9 *")
    print("*"*70)
    
    # get the needed fold time (based on fold_idx)
    train_loader, test_loader, fold_train_dataset = get_anchored_fold_loaders(
        fold_idx=fold_idx, 
        train_files=train_files, 
        test_files=test_files, 
        batch_size=256
    )
    
    if isinstance(fold_train_dataset, torch.utils.data.ConcatDataset):
        active_labels = torch.cat([ds.Y for ds in fold_train_dataset.datasets])
    else:
        active_labels = fold_train_dataset.Y
        
    w = weights(device=device, labels=active_labels)
    loss_fn = FocalLoss(alpha=w, gamma=2)
    
   
    num_epochs_per_fold = 10  
    optimizer = torch.optim.Adam(net_ideal.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs_per_fold, eta_min=1e-6)
    
    best_fold_acc = 0.0
    patience_counter = 0
    patience = 4
    
    # training loop
    for epoch in range(num_epochs_per_fold):
        net_ideal.train()
        epoch_loss = 0.0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).long()
            
            optimizer.zero_grad(set_to_none=True)
            logits = net_ideal(data)
            loss_val = loss_fn(logits, targets)
            
            loss_val.backward()
            torch.nn.utils.clip_grad_norm_(net_ideal.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss_val.item()
            
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        
        net_ideal.eval()
        with torch.no_grad():
            val_correct = 0
            val_total = 0
            for val_data, val_targets in test_loader:
                val_data = val_data.to(device, non_blocking=True)
                val_targets = val_targets.to(device, non_blocking=True).long()
                
                val_preds = net_ideal(val_data).argmax(dim=1)
                val_correct += (val_preds == val_targets).sum().item()
                val_total += val_targets.size(0)
            current_val_acc = val_correct / max(1, val_total)
            
        print(f"Fold {fold_idx+1} | Epoch [{epoch+1}/{num_epochs_per_fold}] | Loss: {avg_loss:.4f} | Current Test Acc: {current_val_acc*100:.2f}%")
        
        if current_val_acc > best_fold_acc:
            best_fold_acc = current_val_acc
            patience_counter = 0
            torch.save(net.state_dict(), os.path.join(output_dir_ideal, f'model_{fold_idx+1}.pt'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"patience counter exceeded the patience threshold {patience}")
                break
                
    # 4. Extract Final Hardware Profiling numbers for this fold's configuration
    print(f"\n compiling hardware metrics for fold: {fold_idx + 1} ")
    checkpoint_file = os.path.join(output_dir_ideal, f'model_{fold_idx+1}.pt')
    net.load_state_dict(torch.load(checkpoint_file, map_location=device, weights_only=True))
    
    total_input_sparsity_ideal = 0.0
    num_batches_ideal = 0
    with torch.no_grad():
        for val_data, _ in test_loader:
            val_data = val_data.to(device, non_blocking=True)
            
            # NOTE: If val_data is already binary spikes, this works as-is. 
            # If your MSNN encodes data into spikes internally, apply that 
            # specific encoding step to val_data here first.
            batch_density_ideal = val_data.float().mean().item() 
            
            total_input_sparsity_ideal += batch_density_ideal
            num_batches_ideal += 1
            
    fold_input_sparsity_ideal = total_input_sparsity_ideal / num_batches_ideal
    
    
    fold_profile_ideal = compute_hardware_metrics(test_loader, net_ideal, device)
    fold_summary_metrics_ideal.append({
        "fold": fold_idx + 1,
        "accuracy": fold_profile_ideal["accuracy"],
        "l1_sparsity": fold_profile_ideal["l1_sparsity"],
        "l2_sparsity": fold_profile_ideal["l2_sparsity"],
        "input_sparsity": fold_input_sparsity_ideal
    })

# 5. Output Summary Results Block for Paper Generation
print("\n" + "="*60)
print("final report for ideal hardware  :::")
print("="*60)
print(f"{'Fold':<6} | {'Test Accuracy':<15} | {'L1 Spike Density':<18} | {'L2 Spike Density':<18}")
print("-"*60)
for entry in fold_summary_metrics_ideal:
  print(f"{entry['fold']:<6} | "
          f"{entry['accuracy']*100:<14.2f}% | "
          f"{entry['input_sparsity']*100:<14.4f}% | "
          f"{entry['l1_sparsity']*100:<17.4f}% | "
          f"{entry['l2_sparsity']*100:<17.4f}%")
print("-"*60)

--- CELL 6 ---
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

folds = np.arange(1, 9)
test_accuracy_ideal = np.array([item["accuracy"] for item in fold_summary_metrics_ideal])
l1_density_ideal = np.array([item["l1_sparsity"] for item in fold_summary_metrics_ideal])
l2_density_ideal = np.array([item["l2_sparsity"] for item in fold_summary_metrics_ideal])

# matplotlib params
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"]  = ["Times New Roman", "DejaVu Serif", "Computer Modern Roman"]

fig, ax1 = plt.subplots(figsize=(8.5, 5.8), dpi=600) 

ax1.set_xlabel('Folds (Test Trading Days)', fontweight='bold', labelpad=10)
ax1.set_ylabel('Test Accuracy', color="#005b96", fontweight='bold', labelpad=10)
line1 = ax1.plot(folds, test_accuracy_ideal, color="#005b96", marker='o', linewidth=2.5, markersize=7, label='Test Accuracy')

ymin = min(test_accuracy_ideal) - 0.03
ymax = max(test_accuracy_ideal) + 0.03
ax1.set_ylim(min(0.40, ymin), max(0.65, ymax)) 

ax1.tick_params(axis='y', labelcolor="#005b96", length=6)
ax1.tick_params(axis='x', length=6)
ax1.set_xticks(folds) 

ax1.grid(True, linestyle=':', color='gray', alpha=0.5, linewidth=1.0)

ax2 = ax1.twinx()
ax2.set_ylabel('Spike Activation Density (%) [Log Scale]', color='black', fontweight='bold', labelpad=12)

line2 = ax2.plot(folds, l1_density_ideal, color="#d95f02", marker='s', linestyle='--', linewidth=2, markersize=7, label='Layer 1 (Hidden)')
line3 = ax2.plot(folds, l2_density_ideal, color="#1b9e77", marker='^', linestyle='-.', linewidth=2, markersize=7, label='Layer 2 (Output)')
ax2.set_yscale('log')

all_densities = np.concatenate([l1_density_ideal, l2_density_ideal])
ax2.set_ylim(min(all_densities) * 0.9, max(all_densities) * 1.1)
ax2.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)))

ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
ax2.yaxis.set_minor_formatter(ticker.NullFormatter()) 

ax2.tick_params(axis='y', which='both', labelcolor='black', length=6, direction='in')

ax1.set_zorder(ax2.get_zorder() + 1)  
ax1.set_frame_on(False)               
ax2.set_frame_on(True)                

lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=3, 
           frameon=True, facecolor='white', edgecolor='black', framealpha=1.0, borderpad=0.8)

plt.title("Hardware performance under Ideal conditions", fontweight='bold', pad=15)

fig.tight_layout(pad=1.5)
plt.savefig("evaluation_plot_ideal.pdf", format='pdf', bbox_inches='tight')
plt.show()


--- CELL 7 ---
print(fold_summary_metrics_ideal)
print(l1_density_ideal)
print(l2_density_ideal)
print(l1_density_ideal.min(), l1_density_ideal.max())
print(l2_density_ideal.min(), l2_density_ideal.max())

--- CELL 8 ---
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Sample arrays matching your code structure
folds = np.arange(1, 9)
accuracy = np.array([item["accuracy"] for item in fold_summary_metrics])
l1_density = np.array([item["l1_sparsity"] for item in fold_summary_metrics])
l2_density = np.array([item["l2_sparsity"] for item in fold_summary_metrics])

# Globally force classic MATLAB inward ticks and formatting
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"]  = ["Times New Roman", "DejaVu Serif", "Computer Modern Roman"]

# Adjusted figure size slightly to make clean room for the bottom legend
fig, ax1 = plt.subplots(figsize=(8.5, 5.8), dpi=600) 

# Configure Primary Axis (Left)
ax1.set_xlabel('Folds (Test Trading Days)', fontweight='bold', labelpad=10)
ax1.set_ylabel('Test Accuracy', color="#005b96", fontweight='bold', labelpad=10)
line1 = ax1.plot(folds, accuracy, color="#005b96", marker='o', linewidth=2.5, markersize=7, label='Test Accuracy')

# --- FIX 1: Dynamic vertical bounds to prevent lines cutting through the top frame ---
ymin = min(accuracy) - 0.03
ymax = max(accuracy) + 0.03
ax1.set_ylim(min(0.40, ymin), max(0.65, ymax)) 

ax1.tick_params(axis='y', labelcolor="#005b96", length=6)
ax1.tick_params(axis='x', length=6)
ax1.set_xticks(folds) # Explicitly show every fold integer

# MATLAB-style grid layout
ax1.grid(True, linestyle=':', color='gray', alpha=0.5, linewidth=1.0)

# Configure Secondary Axis (Right)
ax2 = ax1.twinx()
ax2.set_ylabel('Spike Activation Density (%) [Log Scale]', color='black', fontweight='bold', labelpad=12)

line2 = ax2.plot(folds, l1_density, color="#d95f02", marker='s', linestyle='--', linewidth=2, markersize=7, label='Layer 1 (Hidden)')
line3 = ax2.plot(folds, l2_density, color="#1b9e77", marker='^', linestyle='-.', linewidth=2, markersize=7, label='Layer 2 (Output)')

# Apply log scale
ax2.set_yscale('log')

# --- FIX 2: Set precise data limits and force log sub-interval ticks to display values ---
all_densities = np.concatenate([l1_density, l2_density])
ax2.set_ylim(min(all_densities) * 0.9, max(all_densities) * 1.1)

# Places ticks at fractional intervals within a single decade to guarantee visibility
ax2.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)))

# Clean MATLAB scalar format (e.g., 0.25 instead of 2.5 x 10^-1)
ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
ax2.yaxis.set_minor_formatter(ticker.NullFormatter()) 

# Apply tick styling to major and minor ticks on the right axis border
ax2.tick_params(axis='y', which='both', labelcolor='black', length=6, direction='in')

# Fix the broken top bounding box line caused by twinx
ax1.set_zorder(ax2.get_zorder() + 1)  
ax1.set_frame_on(False)               
ax2.set_frame_on(True)                

# Unified Legend - Positioned safely OUTSIDE below the X-axis
lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]

# bbox_to_anchor points to (X, Y) coordinates relative to axes. (0.5, -0.18) drops it right below the center point.
ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=3, 
           frameon=True, facecolor='white', edgecolor='black', framealpha=1.0, borderpad=0.8)

# Title for the SAF dataset
plt.title("Hardware performance under ≈ 3% SAF", fontweight='bold', pad=15)

# Use tight_layout with specified padding to ensure everything fits inside the PDF boundary
fig.tight_layout(pad=1.5)

# bbox_inches='tight' is critical here to ensure the outside legend isn't cropped during export
plt.savefig("evaluation_plot.pdf", format='pdf', bbox_inches='tight')
plt.show()


--- CELL 9 ---
print(fold_summary_metrics[-1])

--- CELL 10 ---
# LSTM architecture to get hte metrics needed to benchmark against our MSNN
class LSTM_net(nn.Module): 
    def __init__(self, input_size=288, hidden_size=128, num_classes=3):
        super(LSTM_net, self).__init__()
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True) 
        self.drop = nn.Dropout(.3)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        
        final_step_out = out[:, -1, :]
        final_step_out = self.drop(final_step_out)
        logits = self.fc(final_step_out)
        
        return logits
    
print(f"starting LSTM training on {device}")
torch.manual_seed(123)
lstm_fold_summary = []

lstm_net = LSTM_net(input_size=288, hidden_size=128, num_classes=3).to(device)

for fold_idx in range(8):
    print("*"*50 +  f"\n LSTM fold n° {fold_idx + 1}")
    
    train_loader, test_loader, fold_train_dataset = get_anchored_fold_loaders(
        fold_idx=fold_idx, 
        train_files=train_files, 
        test_files=test_files, 
        batch_size=256
    )
    
    if isinstance(fold_train_dataset, torch.utils.data.ConcatDataset):
        active_labels = torch.cat([ds.Y for ds in fold_train_dataset.datasets])
    else:
        active_labels = fold_train_dataset.Y
        
    if isinstance(fold_train_dataset, torch.utils.data.ConcatDataset):
        active_labels = torch.cat([ds.Y for ds in fold_train_dataset.datasets])
    else:
        active_labels = fold_train_dataset.Y
        
    w = weights(device=device, labels=active_labels)
    loss_fn = FocalLoss(alpha=w, gamma=2)
    
    num_epochs_per_fold = 10  
    optimizer = torch.optim.Adam(lstm_net.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs_per_fold, eta_min=1e-6)
    
    best_fold_acc = 0.0
    patience_counter = 0
    patience = 4
    
    for epoch in range(num_epochs_per_fold):
        lstm_net.train()
        epoch_loss = 0.0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).long()
            
            optimizer.zero_grad(set_to_none=True)
            logits = lstm_net(data)
            loss_val = loss_fn(logits, targets)
            
            loss_val.backward()
            torch.nn.utils.clip_grad_norm_(lstm_net.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss_val.item()
            
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        
        # Evaluate 
        lstm_net.eval()
        with torch.no_grad():
            val_correct = 0
            val_total = 0
            for val_data, val_targets in test_loader:
                val_data = val_data.to(device, non_blocking=True)
                val_targets = val_targets.to(device, non_blocking=True).long()
                val_preds = lstm_net(val_data).argmax(dim=1)
                val_correct += (val_preds == val_targets).sum().item()
                val_total += val_targets.size(0)
            current_val_acc = val_correct / max(1, val_total)
            
        print(f"Fold {fold_idx+1} | Epoch [{epoch+1}/{num_epochs_per_fold}] | Loss: {avg_loss:.4f} | LSTM Test Acc: {current_val_acc*100:.2f}%")
        
        if current_val_acc > best_fold_acc:
            best_fold_acc = current_val_acc
            patience_counter = 0
            torch.save(lstm_net.state_dict(), f'best_lstm_fold_{fold_idx+1}.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("↳ Early stopping LSTM fold fine-tuning.")
                break

    print(f"\n--- Saving LSTM Metrics for Fold {fold_idx + 1} ---")
    lstm_fold_summary.append({
        "fold": fold_idx + 1,
        "accuracy": best_fold_acc
    })

# ---------------------------------------------------------
# 3. Final LSTM vs MSNN Summary Report
# ---------------------------------------------------------
print("\n" + "="*50)
print("FINAL LSTM BASELINE REPORT")
print("="*50)
print(f"{'Fold':<6} | {'LSTM Test Accuracy':<20}")
print("-"*50)
lstm_total_acc = 0
for entry in lstm_fold_summary:
    print(f"{entry['fold']:<6} | {entry['accuracy']*100:<19.2f}%")
    lstm_total_acc += entry['accuracy']

print("="*50)
print(f"LSTM 8-Fold Average Accuracy: {(lstm_total_acc / 8)*100:.2f}%")
    

--- CELL 11 ---
def calculate_hardware_metrics(input_density=0.02, l1_density=0.0188, l2_density=0.0055, log=True):
    inputs = 288
    hidden = 128
    outputs = 3
    timesteps = 30

    # --- Corrected Hardware Parameters ---
    bits_per_device = 4
    devices_per_weight = 2  
    
    # Adjusted upward to account for parasitic line capacitance (RC delay)
    energy_per_synop_pj = 1.0       
    
    # Adjusted to reflect full column readout overhead (not just an isolated SAR step)
    adc_energy_pj_per_read = 8.0    
    
    # Added: Energy consumed by LIF neuron membrane integration per timestep
    energy_per_neuron_step_pj = 2.5 
    
    # Adjusted upward to reflect real peripheral control circuits + distribution networks
    static_power_uw = 75.0            
    latency_per_step_ns = 100

    # Topology and memory
    params_l1 = inputs * hidden
    params_l2 = hidden * outputs
    total_params = params_l1 + params_l2
    total_devices = total_params * devices_per_weight
    memory_footprint_kb = (total_devices * bits_per_device) / (8 * 1024)

    # Spike and SynOp math
    total_input_spikes = inputs * timesteps * input_density
    total_hidden_spikes = hidden * timesteps * l1_density
    synops_l1 = total_input_spikes * hidden
    synops_l2 = total_hidden_spikes * outputs
    total_synops = synops_l1 + synops_l2

    # --- Energy Components (Calculated in nJ) ---
    synop_energy_nj = (total_synops * energy_per_synop_pj) / 1000  
    
    total_adc_reads = (hidden + outputs) * timesteps
    adc_energy_nj = (total_adc_reads * adc_energy_pj_per_read) / 1000  

    # Added: Neuron integration energy component
    total_neuron_updates = (hidden + outputs) * timesteps
    neuron_energy_nj = (total_neuron_updates * energy_per_neuron_step_pj) / 1000

    latency_us = (timesteps * latency_per_step_ns) / 1000
    static_energy_nj = (static_power_uw * latency_us) / 1000  

    # Updated system energy sum
    total_energy_nj = synop_energy_nj + adc_energy_nj + neuron_energy_nj + static_energy_nj

    # ... keeping your logging format below ...

    metrics = []
    if log:
        print("*"*50)
        print("hardware metrics:")
        print("*"*50)
        print(f"Number of parameters:       {total_params:,}")
        print(f"Number of RRAM Devices:     {total_devices:,} (Differential)")
        print(f"Memory Footprint:           {memory_footprint_kb:.2f} KB")
        print("+" * 50)
        print(f"Average L1 Firing Rate:     {l1_density * 100:.2f}%")
        print(f"Average L2 Firing Rate:     {l2_density * 100:.2f}%")
        print(f"Total Input Spikes:         {int(total_input_spikes):,}")
        print(f"Total Hidden Spikes:        {int(total_hidden_spikes):,}")
        print("+" * 50)
        print(f"Total SynOps:               {int(total_synops):,}")
        print(f"  -> Device-level SynOp Energy:  {synop_energy_nj:.4f} nJ  (analog crossbar switching only)")
        print(f"  -> ADC Conversion Energy:      {adc_energy_nj:.4f} nJ  ({total_adc_reads} reads @ {adc_energy_pj_per_read} pJ/read)")
        print(f"  -> Static/Leakage Energy:      {static_energy_nj:.4f} nJ  (@ {static_power_uw} µW over {latency_us:.2f} µs)")
        print(f"  -> TOTAL System-Level Energy:  {total_energy_nj:.4f} nJ")
        print(f"Inference Latency:          {latency_us:.2f} µs")
        print("+"*50)
        print("NOTE: Device-level SynOp energy alone is NOT directly comparable to full-chip")
        print("      measurements (e.g. Wu et al. 2021, 10.3 µJ/sample) which include ADC,")
        print("      periphery, and static power over a much longer real sample duration.")
        print("+"*50)

    metrics.append({
        "num_param": total_params,
        "num_rram": total_devices,
        "memory": memory_footprint_kb,
        "total_hidden_spikes": int(total_hidden_spikes),
        "total_input_spikes": int(total_input_spikes),
        "total_synops": int(total_synops),
        "synop_energy_nj": synop_energy_nj,
        "adc_energy_nj": adc_energy_nj,
        "static_energy_nj": static_energy_nj,
        "total_energy_nj": total_energy_nj,
        "inference_latency": latency_us,
    })

    return metrics

avg_l1_density = np.mean([item["l1_sparsity"] for item in fold_summary_metrics])
avg_l2_density = np.mean([item["l2_sparsity"] for item in fold_summary_metrics])
avg_input_density = np.mean([item["input_sparsity"] for item in fold_summary_metrics])

ms = calculate_hardware_metrics(input_density=avg_input_density, l1_density=avg_l1_density, l2_density=avg_l2_density)
print(avg_input_density)

--- CELL 12 ---
# ======================================================================
# Digital LSTM vs. MSNN Hardware & Area Comparison Calculator
# ======================================================================

def calculate_baseline_comparison():
    # --- 1. Architectural & Topography Constants ---
    inputs = 288
    hidden = 128
    outputs = 3
    timesteps = 30
    
    # MSNN values from your hardware profile
    msnn_devices = ms[0]["num_rram"]
    msnn_synops = ms[0]["total_synops"]
    msnn_energy_nj = ms[0]["total_energy_nj"]
    
    # --- 2. LSTM Calculations ---
    # LSTM has 4 gates: Input, Forget, Cell, Output
    # Parameters per gate = (Inputs * Hidden) + (Hidden * Hidden) + Hidden (bias)
    params_per_gate = (inputs * hidden) + (hidden * hidden) + hidden
    lstm_params_layer = 4 * params_per_gate
    lstm_params_output = (hidden * outputs) + outputs
    lstm_total_params = lstm_params_layer + lstm_params_output
    
    # Dense LSTMs perform MAC operations for every parameter at every timestep
    lstm_total_macs = lstm_total_params * timesteps
    
    # Energy constant for 32-bit floating-point digital MAC (Horowitz ISSCC standard: ~3.1 pJ per MAC)
    energy_per_mac_pj = 3.1
    lstm_energy_nj = (lstm_total_macs * energy_per_mac_pj) / 1000
    
    # --- 3. Area Estimation Constants (RRAM CIM vs. Digital CMOS) ---
    # Approximate layout area per RRAM device (assuming standard crossbar node, e.g., 65nm or 28nm roughly ~4-10 F^2)
    # Let's use a standard literature estimate: ~0.04 um^2 per RRAM cell including minimal peripheral CMOS footprint share
    # Or macro-level density estimation: ~1000 um^2 per Kb for dense RRAM crossbars.
    rram_area_per_device_um2 = 0.04 
    msnn_crossbar_area_um2 = msnn_devices * rram_area_per_device_um2
    
    # Digital Standard Cell Area estimation for LSTM (32-bit floating point MAC units and registers)
    # A standard 32-bit FP MAC cell in CMOS takes roughly 10,000 to 20,000 gates (~5000 um^2 at 65nm per equivalent gate block)
    # Alternatively, total area proportional to parameter storage and active datapath logic.
    # Let's provide a clear transistor/gate equivalent layout approximation.
    digital_gate_area_um2 = 1.5  # Typical 65nm 2-input NAND equivalent area
    # A 32-bit floating point multiplier/accumulator takes roughly 15,000 equivalent gates. 
    # For a dense sequential LSTM running parallel blocks, let's derive footprint via standard synthesized cell area:
    lstm_estimated_area_um2 = lstm_total_params * 32 * digital_gate_area_um2 * 0.5 # compressed footprint factor

    # --- 4. Print Comparison Table ---
    print("="*65)
    print(f"{'HARDWARE METRIC':<30} | {'MSNN (RRAM CIM)':<15} | {'Digital LSTM':<15}")
    print("="*65)
    print(f"{'Parameters':<30} | {37248:<15,} | {lstm_total_params:<15,}")
    print(f"{'Memory Elements / Devices':<30} | {msnn_devices:<15,} | {lstm_total_params * 32:<15,}")
    print(f"{'Operations (SynOps vs MACs)':<30} | {msnn_synops:<15,} | {lstm_total_macs:<15,}")
    print(f"{'Estimated Energy (nJ)':<30} | {msnn_energy_nj:<15.4f} | {lstm_energy_nj:<15.2f}")
    print(f"{'Approx. Silicon Area (um²)':<30} | {msnn_crossbar_area_um2:<15.2f} | {lstm_estimated_area_um2:<15.2f}")
    print("="*65)
    
    # Efficiency Multipliers
    energy_speedup = lstm_energy_nj / msnn_energy_nj
    op_reduction = lstm_total_macs / msnn_synops
    print(f"\n[SUMMARY] MSNN achieves a {op_reduction:.1f}x reduction in operations")
    print(f"[SUMMARY] MSNN achieves a {energy_speedup:.1f}x improvement in energy efficiency over the LSTM baseline.")

calculate_baseline_comparison()

--- CELL 13 ---
import torch
import glob
import numpy as np
# (Include your other original imports here)

# 1. PASTE YOUR CLASSES AND FUNCTIONS HERE
# Paste: STEQuantize, MemristorCrossbar, MSNN, LOBDayDataset
# Paste: get_anchored_fold_loaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. DEFINE YOUR DATA FILES (Update these paths for your new environment)
train_files = sorted(glob.glob(
    "/kaggle/input/datasets/firmwired/fi-2010/data/published/BenchmarkDatasets/BenchmarkDatasets/BenchmarkDatasets/NoAuction/NoAuction_MinMax/NoAuction_MinMax_Training/Train_Dst_NoAuction_MinMax_CF_*.txt"
))
test_files = sorted(glob.glob(
    "/kaggle/input/datasets/firmwired/fi-2010/data/published/BenchmarkDatasets/BenchmarkDatasets/BenchmarkDatasets/NoAuction/NoAuction_MinMax/NoAuction_MinMax_Testing/Test_Dst_NoAuction_MinMax_CF_*.txt"
))

for i in range(8):
    # 3. SELECT THE FOLD YOU WANT TO TEST
    fold_idx = i # fold_idx = 0 means Fold 1

    # 4. LOAD THE ANCHORED DATA FOR THIS SPECIFIC FOLD
    # We only need the test_loader for inference, so we can ignore the train returns
    _, test_loader, _ = get_anchored_fold_loaders(
        fold_idx=fold_idx, 
        train_files=train_files, 
        test_files=test_files, 
        batch_size=256
    )

    # 5. INSTANTIATE THE MODEL
    new = MSNN(
        num_inputs=288, 
        num_hidden=128, 
        num_outputs=3, 
        beta=0.9, 
        num_levels=16, 
        saf_rate=0.03, 
        noise_std=0.05,
        force_ideal=False,
        force_noise_eval=True
    ).to(device)

    # 6. LOAD THE CORRESPONDING WEIGHTS
    # Match the model number to the fold_idx (fold_idx 0 -> model_1.pt)
    checkpoint_path = f'/kaggle/working/MSNN/noisy/model_8.pt'
    new.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))

    new.eval()
    new.reset_spike_metrics()

    # 7. RUN INFERENCE
    correct = 0
    total = 0

    print(f"\nRunning inference on Fold {fold_idx + 1}...")
    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).long()
            
            logits = new(data)
            preds = logits.argmax(dim=1)
            
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    accuracy = correct / max(1, total)
    print(accuracy)
    print(f"Test Accuracy for Fold {fold_idx + 1}: {accuracy * 100:.2f}%")

    


--- CELL 14 ---
def run_anchored_experiment(seed, force_ideal, force_noise_eval, train_files, test_files,
                             num_folds=8, num_epochs_per_fold=10, patience=4,
                             checkpoint_dir='/kaggle/working/MSNN/sweep'):
    """
    Runs the full anchored-fold training+eval pipeline once, for one seed and one
    hardware condition. Returns a list of per-fold metric dicts (same shape as
    your existing fold_summary_metrics).
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.manual_seed(seed)

    net_seeds = MSNN(num_inputs=288,
               num_hidden=128,
               num_outputs=3,
               beta=0.9,
               num_levels=16,
               saf_rate=0.03,
               noise_std=0.05,
               force_ideal=force_ideal,
               force_noise_eval=force_noise_eval
               ).to(device)
    
    fold_summary = []

    for fold_idx in range(num_folds):
        train_loader, test_loader, fold_train_dataset = get_anchored_fold_loaders(
            fold_idx=fold_idx, 
            train_files=train_files,
            test_files=test_files,
            batch_size=256
        )

        if isinstance(fold_train_dataset, torch.utils.data.ConcatDataset):
            active_labels = torch.cat([ds.Y for ds in fold_train_dataset.datasets])
        else:
            active_labels = fold_train_dataset.Y

        w = weights(device=device, labels=active_labels)
        loss_fn = FocalLoss(alpha=w, gamma=2)

        optimizer = torch.optim.Adam(net_seeds.parameters(), lr=3e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs_per_fold, eta_min=1e-6)

        best_fold_acc = 0.0
        patience_counter = 0
        ckpt_path = f'{checkpoint_dir}/seed{seed}_{"ideal" if force_ideal else "noisy"}_fold{fold_idx+1}.pt'

        for epoch in range(num_epochs_per_fold):
            net_seeds.train()
            
            for data, targets in train_loader:
                data = data.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True).long()
                
                optimizer.zero_grad(set_to_none=True)
                
                loss_val = loss_fn(net_seeds(data), targets)
                
                loss_val.backward()
                torch.nn.utils.clip_grad_norm_(net_seeds.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()

            net_seeds.eval()
            with torch.no_grad():
                correct, total = 0, 0
                for val_data, val_targets in test_loader:
                    val_data = val_data.to(device, non_blocking=True)
                    val_targets = val_targets.to(device, non_blocking=True).long()
                    preds = net_seeds(val_data).argmax(dim=1)
                    correct += (preds == val_targets).sum().item()
                    total += val_targets.size(0)
                current_val_acc = correct / max(1, total)

            if current_val_acc > best_fold_acc:
                best_fold_acc = current_val_acc
                patience_counter = 0
                torch.save(net_seeds.state_dict(), ckpt_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        net_seeds.load_state_dict(torch.load(ckpt_path, map_location=device))

        total_input_sparsity, num_batches = 0.0, 0
        with torch.no_grad():
            for val_data, _ in test_loader:
                total_input_sparsity += val_data.float().mean().item()
                num_batches += 1
        fold_input_sparsity = total_input_sparsity / num_batches

        fold_profile = compute_hardware_metrics(test_loader, net_seeds, device)
        fold_summary.append({
            "fold": fold_idx + 1,
            "accuracy": fold_profile["accuracy"],
            "l1_sparsity": fold_profile["l1_sparsity"],
            "l2_sparsity": fold_profile["l2_sparsity"],
            "input_sparsity": fold_input_sparsity
        })

    return fold_summary


# Run Sweeeep
seeds = [42, 7, 3]   
all_noisy_runs = [fold_summary_metrics]
all_ideal_runs = []

for s in seeds:
    print(f"\n===== SEED {s} — NOISY =====")
    all_noisy_runs.append(run_anchored_experiment(s, force_ideal=False, force_noise_eval=True,
                                                    train_files=train_files, test_files=test_files))
    print(f"\n===== SEED {s} — IDEAL =====")
    all_ideal_runs.append(run_anchored_experiment(s, force_ideal=True, force_noise_eval=False,
                                                    train_files=train_files, test_files=test_files))

--- CELL 15 ---
def aggregate_across_seeds(all_runs, num_folds=8):
    """all_runs: list of fold_summary lists (one per seed). Returns per-fold mean/std."""
    acc_matrix = np.array([[run[f]["accuracy"] for f in range(num_folds)] for run in all_runs])
    l1_matrix = np.array([[run[f]["l1_sparsity"] for f in range(num_folds)] for run in all_runs])
    l2_matrix = np.array([[run[f]["l2_sparsity"] for f in range(num_folds)] for run in all_runs])
    return {
        "acc_mean": acc_matrix.mean(axis=0), "acc_std": acc_matrix.std(axis=0),
        "l1_mean": l1_matrix.mean(axis=0), "l1_std": l1_matrix.std(axis=0),
        "l2_mean": l2_matrix.mean(axis=0), "l2_std": l2_matrix.std(axis=0),
    }

noisy_agg = aggregate_across_seeds(all_noisy_runs)
ideal_agg = aggregate_across_seeds(all_ideal_runs)
print("Noisy accuracy (mean ± std) per fold:", noisy_agg["acc_mean"], "±", noisy_agg["acc_std"])
print("Ideal accuracy (mean ± std) per fold:", ideal_agg["acc_mean"], "±", ideal_agg["acc_std"])

--- CELL 16 ---
import json
import pandas as pd
with open('/kaggle/input/datasets/mfxg3zrz4r2dfr/noisy-runs/all_noisy_runs.txt') as f:
    all_noisy_runs = json.load(f)

vals = [fold["accuracy"] for seed_run in all_noisy_runs for fold in seed_run]
   
print( np.mean(vals) * 100)
# a = np.mean([item["l1_sparsity"] for item in all_noisy_runs])

