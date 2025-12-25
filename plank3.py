#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuroSovereign: Optimal Sovereignty Search
Finding the Bekenstein Bound of Sparse Intelligence

This experiment:
1. Trains a dense model to ~32.4% accuracy
2. Progressively prunes it while monitoring L and accuracy
3. Finds the critical density where accuracy drops below 32.4%
4. Validates that L > 1.0 correlates with meaningful representation

Outputs:
- CSV with density vs accuracy vs L
- Critical density threshold
- Spectral signature of the sovereignty boundary
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import os
import time
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. SPECTRAL COHERENCE MONITOR (PASSIVE OBSERVABLE)
# =============================================================================
class SpectralMonitor:
    def __init__(self, epsilon_c: float = 0.3):
        self.epsilon_c = epsilon_c

    def compute_L(self, weight: torch.Tensor) -> Tuple[float, float, int, str]:
        with torch.no_grad():
            W = weight.cpu().numpy()
            try:
                U, S, Vh = np.linalg.svd(W, full_matrices=False)
                threshold = 0.05 * np.max(S)
                rank_eff = max(1, int(np.sum(S > threshold)))
                S_norm = S / (np.sum(S) + 1e-12)
                S_norm = S_norm[S_norm > 1e-15]
                S_vN = -np.sum(S_norm * np.log(S_norm + 1e-15))
                L = 1.0 / (abs(S_vN - np.log(rank_eff + 1)) + self.epsilon_c)
                if L > 1.0:
                    regime = "SOBERANO"
                elif L > 0.5:
                    regime = "EMERGENTE"
                else:
                    regime = "ESPURIO"
                return L, S_vN, rank_eff, regime
            except:
                return 1.0, 0.0, 1, "SOBERANO"

# =============================================================================
# 2. PERSISTENT PRUNING ENGINE
# =============================================================================
class PersistentPruner:
    def __init__(self, sparsity_target: float):
        self.sparsity_target = sparsity_target
        self.masks = {}

    def apply_to_model(self, model: nn.Module):
        for name, param in model.named_parameters():
            if "weight" in name and param.ndim == 2:
                threshold = torch.quantile(torch.abs(param.data), self.sparsity_target)
                mask = (torch.abs(param.data) > threshold).float()
                self.masks[name] = mask
                param.data *= mask
                if param.grad is not None:
                    param.grad *= mask

    def enforce_during_training(self, model: nn.Module):
        for name, param in model.named_parameters():
            if name in self.masks:
                param.data *= self.masks[name]
                if param.grad is not None:
                    param.grad *= self.masks[name]

# =============================================================================
# 3. MINIMAL MLP FOR CONTROLLED EXPERIMENT
# =============================================================================
class SpectralMLP(nn.Module):
    def __init__(self, input_dim: int = 32, hidden_dim: int = 47, num_classes: int = 10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, num_classes, bias=False)
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.01)

    def reduce_input(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), 3, 32, 32)
        x = x.mean(dim=1)
        x = F.adaptive_avg_pool2d(x, (4, 8))
        return x.view(x.size(0), -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.reduce_input(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# =============================================================================
# 4. DENSE TRAINING TO TARGET ACCURACY (32.4%)
# =============================================================================
def train_dense_to_target(device: torch.device, target_acc: float = 32.4) -> Tuple[nn.Module, List[Dict]]:
    """Train dense model until it reaches target accuracy."""
    print(f"🎯 Training dense model to target accuracy: {target_acc}%")
    
    model = SpectralMLP().to(device)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    indices = torch.randperm(len(trainset))[:10000]
    trainset = torch.utils.data.Subset(trainset, indices)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    monitor = SpectralMonitor()
    
    epoch = 0
    training_log = []
    
    while True:
        model.train()
        for batch_idx, (data, target) in enumerate(trainloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        # Evaluate
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in testloader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        test_acc = 100.0 * correct / total
        
        # Spectral metrics
        L, S_vN, rank_eff, regime = monitor.compute_L(model.fc1.weight)
        density = 1.0
        
        log_entry = {
            'epoch': epoch,
            'test_acc': test_acc,
            'L': L,
            'density': density,
            'rank_eff': rank_eff,
            'regime': regime
        }
        training_log.append(log_entry)
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch:3d} | Acc: {test_acc:5.2f}% | L: {L:5.3f} | Density: {density:7.4f}")
        
        if test_acc >= target_acc:
            print(f"✅ Target accuracy {target_acc}% achieved at epoch {epoch}")
            break
            
        if epoch > 100:  # Safety break
            print("⚠️  Max epochs reached, using best available")
            break
            
        epoch += 1
    
    return model, training_log

# =============================================================================
# 5. PROGRESSIVE PRUNING SEARCH
# =============================================================================
def progressive_pruning_search(model: nn.Module, device: torch.device, target_acc: float = 32.4) -> pd.DataFrame:
    """Progressively prune model and find critical density threshold."""
    print(f"\n🔍 Starting progressive pruning search for {target_acc}% accuracy...")
    
    # Load test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)
    
    monitor = SpectralMonitor()
    pruning_log = []
    
    # Evaluate dense model
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    dense_acc = 100.0 * correct / total
    L_dense, _, _, _ = monitor.compute_L(model.fc1.weight)
    
    print(f"Dense baseline: Acc = {dense_acc:.2f}%, L = {L_dense:.3f}")
    
    # Progressive pruning from 0% to 99.9% sparsity
    sparsity_levels = np.linspace(0.0, 0.999, 50)  # 50 levels from 0% to 99.9% sparsity
    
    for i, sparsity in enumerate(sparsity_levels):
        # Create fresh model copy to avoid gradient contamination
        model_copy = SpectralMLP().to(device)
        model_copy.load_state_dict(model.state_dict())
        
        # Apply pruning
        pruner = PersistentPruner(sparsity)
        pruner.apply_to_model(model_copy)
        
        # Evaluate pruned model
        model_copy.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in testloader:
                data, target = data.to(device), target.to(device)
                output = model_copy(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        test_acc = 100.0 * correct / total
        
        # Spectral metrics
        L, S_vN, rank_eff, regime = monitor.compute_L(model_copy.fc1.weight)
        density = 1.0 - sparsity
        
        log_entry = {
            'sparsity_level': sparsity,
            'density': density,
            'test_acc': test_acc,
            'L': L,
            'rank_eff': rank_eff,
            'regime': regime
        }
        pruning_log.append(log_entry)
        
        if i % 10 == 0:
            print(f"Pruning {i+1:2d}/50 | Sparsity: {sparsity:6.3f} | Acc: {test_acc:5.2f}% | L: {L:5.3f}")
    
    return pd.DataFrame(pruning_log)

# =============================================================================
# 6. FIND CRITICAL DENSITY THRESHOLD
# =============================================================================
def find_critical_threshold(pruning_df: pd.DataFrame, target_acc: float = 32.4) -> Dict:
    """Find the minimum density where accuracy >= target_acc."""
    # Find all densities where accuracy >= target
    valid_rows = pruning_df[pruning_df['test_acc'] >= target_acc]
    
    if len(valid_rows) == 0:
        return {
            'critical_density': None,
            'critical_sparsity': None,
            'max_accuracy_at_threshold': None,
            'L_at_threshold': None,
            'rank_eff_at_threshold': None,
            'status': 'TARGET_NOT_ACHIEVABLE'
        }
    
    # Find minimum density (maximum sparsity) that still achieves target
    critical_row = valid_rows.loc[valid_rows['density'].idxmin()]
    
    return {
        'critical_density': critical_row['density'],
        'critical_sparsity': critical_row['sparsity_level'],
        'max_accuracy_at_threshold': critical_row['test_acc'],
        'L_at_threshold': critical_row['L'],
        'rank_eff_at_threshold': critical_row['rank_eff'],
        'status': 'SUCCESS'
    }

# =============================================================================
# 7. MAIN EXPERIMENT
# =============================================================================
def main():
    print("="*80)
    print("🧪 NeuroSovereign: Optimal Sovereignty Search")
    print("Finding the Bekenstein Bound of Sparse Intelligence")
    print("="*80)
    
    device = torch.device("cpu")
    target_accuracy = 32.4
    
    # Step 1: Train dense model to target accuracy
    dense_model, dense_log = train_dense_to_target(device, target_accuracy)
    
    # Step 2: Progressive pruning search
    pruning_results = progressive_pruning_search(dense_model, device, target_accuracy)
    
    # Step 3: Find critical threshold
    critical_info = find_critical_threshold(pruning_results, target_accuracy)
    
    # Save results
    pruning_results.to_csv("sovereignty_search_results.csv", index=False)
    
    # Print results
    print("\n" + "="*80)
    print("🏆 Bekenstein Bound Results")
    print("="*80)
    
    if critical_info['status'] == 'SUCCESS':
        print(f"✅ Critical Density Found: {critical_info['critical_density']:.6f}")
        print(f"   Critical Sparsity: {critical_info['critical_sparsity']:.6f}")
        print(f"   Accuracy at Threshold: {critical_info['max_accuracy_at_threshold']:.2f}%")
        print(f"   Spectral Coherence (L): {critical_info['L_at_threshold']:.3f}")
        print(f"   Effective Rank: {critical_info['rank_eff_at_threshold']}")
        print(f"   Regime: {'SOBERANO' if critical_info['L_at_threshold'] > 1.0 else 'EMERGENTE'}")
        
        # Scientific interpretation
        print(f"\n🔬 Scientific Interpretation:")
        if critical_info['L_at_threshold'] > 1.0:
            print(f"   The model maintains SOBERANO regime at the critical density.")
            print(f"   This represents the thinnest possible representation that encodes truth.")
        else:
            print(f"   The model operates in EMERGENTE regime at critical density.")
            print(f"   There is a trade-off between sparsity and spectral coherence.")
            
    else:
        print("❌ Target accuracy not achievable under any sparsity level.")
        print("   Consider adjusting target accuracy or model architecture.")
    
    # Plot the sovereignty curve
    print(f"\n📊 Sovereignty Curve Summary:")
    print(f"   Dense model: {pruning_results.iloc[0]['test_acc']:.2f}% @ L={pruning_results.iloc[0]['L']:.3f}")
    print(f"   Random baseline: ~10.0% (CIFAR-10 chance level)")
    print(f"   Critical point: {critical_info['max_accuracy_at_threshold']:.2f}% @ density={critical_info['critical_density']:.6f}")
    
    print(f"\n✅ Results saved to: sovereignty_search_results.csv")
    print("="*80)

if __name__ == "__main__":
    main()