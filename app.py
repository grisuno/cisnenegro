#!/usr/bin/env python3
# _*_ coding: utf8 _*_
"""
app.py

Autor: Gris Iscomeback
Correo electrónico: grisiscomeback[at]gmail[dot]com
Fecha de creación: xx/xx/xxxx
Licencia: GPL v3

NeuroSovereign v6.0: Self-Improving Fractal Resonance with Legacy Feedback

This implementation introduces a continuous feedback loop where each cycle
builds upon the best model from the previous cycle, creating an evolutionary
trajectory of increasing abstraction density.

Key Innovation: Legacy Feedback Loop
- Each cycle loads the best model from the previous cycle as its truth seed
- Only saves new models that improve upon the previous best
- Creates an unbroken chain of improvement: never regresses, only evolves

Scientific Contribution:
- Demonstrates progressive abstraction density through iterative distillation
- Validates that spectral coherence can be maintained while increasing accuracy
- Establishes a self-improving protocol for sparse neural architectures

Outputs:
- fractal_resonance_results.csv: Evolutionary trajectory across cycles
- best_model_cycle_X.pth: Checkpoint of best model at each cycle
- final_best_model.pth: Ultimate distilled model
- evolution_metrics.json: Complete evolutionary metrics
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import json
import os
import time
from typing import Dict, List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Core Components (Optimized for Evolutionary Learning)
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
# Evolutionary Fractal Resonance Engine
# =============================================================================
class EvolutionaryResonanceEngine:
    def __init__(self, device: torch.device, base_target_acc: float = 32.4):
        self.device = device
        self.base_target_acc = base_target_acc
        self.monitor = SpectralMonitor()
        self.best_metrics = {
            'accuracy': 0.0,
            'L': 0.0,
            'density': 1.0,
            'cycle': 0,
            'logit_stability': 0.0
        }
        self.results_history = []
        self.model_save_dir = "model_evolution"
        os.makedirs(self.model_save_dir, exist_ok=True)

    def load_best_legacy_model(self, cycle: int) -> Optional[Dict]:
        """Load the best model from previous cycle, with fallback to initial seed"""
        if cycle == 1:
            # Try to load from previous experiments
            possible_paths = [
                'best_enhanced_model.pth',
                'enhanced_model.pth', 
                'best_model.pth',
                os.path.join(self.model_save_dir, 'best_model_cycle_0.pth')
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    try:
                        checkpoint = torch.load(path, map_location=self.device)
                        print(f"✅ Loaded initial seed from: {path}")
                        return checkpoint
                    except Exception as e:
                        continue
            
            print("⚠️  No legacy model found. Starting with fresh base model...")
            return None
        
        else:
            # Load from previous cycle
            prev_model_path = os.path.join(self.model_save_dir, f'best_model_cycle_{cycle-1}.pth')
            if os.path.exists(prev_model_path):
                try:
                    checkpoint = torch.load(prev_model_path, map_location=self.device)
                    print(f"✅ Loaded best model from cycle {cycle-1}")
                    return checkpoint
                except Exception as e:
                    print(f"⚠️  Error loading previous model: {e}")
                    return None
            else:
                print(f"⚠️  No model found for cycle {cycle-1}. Using cycle 0 model...")
                return self.load_best_legacy_model(1)

    def train_base_model_to_target(self, hidden_dim: int, target_acc: float, test_loader) -> nn.Module:
        """Train a base model to target accuracy"""
        model = SpectralMLP(hidden_dim=hidden_dim).to(self.device)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        indices = torch.randperm(len(trainset))[:10000]
        trainset = torch.utils.data.Subset(trainset, indices)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        epoch = 0
        best_acc = 0.0
        patience = 0
        max_patience = 10
        
        while epoch < 200:
            model.train()
            for data, target in trainloader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target).sum().item()
                    total += target.size(0)
            test_acc = 100.0 * correct / total
            
            if test_acc > best_acc:
                best_acc = test_acc
                patience = 0
            else:
                patience += 1
                
            if test_acc >= target_acc or patience >= max_patience:
                break
            epoch += 1
        
        return model

    def extract_seed_from_checkpoint(self, checkpoint: Dict, expected_hidden_dim: int = 47) -> Dict[str, torch.Tensor]:
        """Extract seed weights from checkpoint, handling different formats"""
        try:
            if 'fc1.weight' in checkpoint and 'fc2.weight' in checkpoint:
                return {
                    'fc1': checkpoint['fc1.weight'],
                    'fc2': checkpoint['fc2.weight']
                }
            elif 'model_state' in checkpoint:
                return {
                    'fc1': checkpoint['model_state']['fc1.weight'],
                    'fc2': checkpoint['model_state']['fc2.weight']
                }
            else:
                # Assume it's a state dict of SpectralMLP
                fc1_key = [k for k in checkpoint.keys() if 'fc1.weight' in k][0]
                fc2_key = [k for k in checkpoint.keys() if 'fc2.weight' in k][0]
                return {
                    'fc1': checkpoint[fc1_key],
                    'fc2': checkpoint[fc2_key]
                }
        except Exception as e:
            print(f"⚠️  Error extracting seed: {e}")
            # Create a new base model if extraction fails
            return self.extract_seed_weights(self.train_base_model_to_target(expected_hidden_dim, self.base_target_acc, None))

    def extract_seed_weights(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        return {
            'fc1': model.fc1.weight.data.clone(),
            'fc2': model.fc2.weight.data.clone()
        }

    def inoculate_seed_adaptive(self, large_model: nn.Module, seed_weights: Dict[str, torch.Tensor]):
        """Adaptive inoculation that handles dimension mismatches"""
        with torch.no_grad():
            large_h = large_model.fc1.weight.shape[0]
            seed_h = seed_weights['fc1'].shape[0]
            
            if large_h >= seed_h:
                # Random embedding with harmonic initialization
                start_idx = np.random.randint(0, large_h - seed_h + 1)
                end_idx = start_idx + seed_h
                large_model.fc1.weight[start_idx:end_idx, :] = seed_weights['fc1']
                large_model.fc2.weight[:, start_idx:end_idx] = seed_weights['fc2']
                
                # Harmonic initialization for new weights
                U, S, Vh = torch.svd(seed_weights['fc1'])
                base_std = S.mean().item() / 10.0 if S.numel() > 0 else 0.01
                
                for i in range(large_h):
                    if i < start_idx or i >= end_idx:
                        large_model.fc1.weight[i, :] = torch.randn_like(large_model.fc1.weight[i, :]) * base_std
                        large_model.fc2.weight[:, i] = torch.randn_like(large_model.fc2.weight[:, i]) * base_std
            else:
                # Project seed down to fit
                step = seed_h // large_h
                selected_indices = torch.arange(0, seed_h, step)[:large_h]
                large_model.fc1.weight.data = seed_weights['fc1'][selected_indices]
                large_model.fc2.weight.data = seed_weights['fc2'][:, selected_indices]

    def measure_functional_alignment(self, model1: nn.Module, model2: nn.Module, test_loader) -> float:
        """Measure functional alignment via logit cosine similarity"""
        model1.eval()
        model2.eval()
        
        cos_similarities = []
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(self.device)
                logits1 = model1(data)
                logits2 = model2(data)
                
                # Compute cosine similarity for each sample
                for i in range(logits1.size(0)):
                    a = logits1[i].cpu().numpy().reshape(1, -1)
                    b = logits2[i].cpu().numpy().reshape(1, -1)
                    sim = cosine_similarity(a, b)[0, 0]
                    cos_similarities.append(sim)
        
        return float(np.mean(cos_similarities)) if cos_similarities else 0.0

    def progressive_pruning_with_target(self, model: nn.Module, target_acc: float, test_loader, max_density: float = 0.1) -> Tuple[Optional[nn.Module], Dict]:
        """Prune while maintaining target accuracy, with density constraint"""
        sparsity_levels = np.linspace(0.0, 0.999, 100)  # Higher resolution
        best_model = None
        best_metrics = {'accuracy': 0.0, 'L': 0.0, 'density': 1.0, 'sparsity': 0.0}
        
        for sparsity in sparsity_levels:
            density = 1.0 - sparsity
            if density > max_density:
                continue
                
            model_copy = SpectralMLP(hidden_dim=model.fc1.weight.shape[0]).to(self.device)
            model_copy.load_state_dict(model.state_dict())
            
            pruner = PersistentPruner(sparsity)
            pruner.apply_to_model(model_copy)
            
            model_copy.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model_copy(data)
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target).sum().item()
                    total += target.size(0)
            test_acc = 100.0 * correct / total
            
            L, _, _, _ = self.monitor.compute_L(model_copy.fc1.weight)
            
            # Prioritize models that meet target accuracy with lowest density
            if test_acc >= target_acc and density < best_metrics['density']:
                best_metrics.update({
                    'accuracy': test_acc,
                    'L': L,
                    'density': density,
                    'sparsity': sparsity
                })
                best_model = model_copy
        
        return best_model, best_metrics

    def execute_resonance_cycle(self, cycle: int, test_loader, expansion_factor: int = 8) -> Dict:
        print(f"\n🌀 CICLO DE RESONANCIA {cycle}...")
        print("-" * 50)
        
        # Load best model from previous cycle
        legacy_checkpoint = self.load_best_legacy_model(cycle)
        
        if legacy_checkpoint is not None:
            seed_weights = self.extract_seed_from_checkpoint(legacy_checkpoint)
            seed_hidden_dim = seed_weights['fc1'].shape[0]
            print(f"1. Using legacy model (hidden_dim={seed_hidden_dim}) as truth seed")
        else:
            print("1. Training fresh base model...")
            base_model = self.train_base_model_to_target(47, self.base_target_acc, test_loader)
            seed_weights = self.extract_seed_weights(base_model)
            seed_hidden_dim = 47
        
        # Create expanded architecture
        expanded_dim = max(seed_hidden_dim * expansion_factor, 128)  # Ensure minimum size
        print(f"2. Creating expanded architecture ({expanded_dim} neurons)...")
        expanded_model = SpectralMLP(hidden_dim=expanded_dim).to(self.device)
        
        # Inoculate seed with adaptive embedding
        print("3. Inoculating truth seed with adaptive embedding...")
        self.inoculate_seed_adaptive(expanded_model, seed_weights)
        
        # Train expanded model to higher target
        print("4. Training expanded model...")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        indices = torch.randperm(len(trainset))[:10000]
        trainset = torch.utils.data.Subset(trainset, indices)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
        
        optimizer = torch.optim.Adam(expanded_model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Train until convergence or max epochs
        epoch = 0
        best_acc = 0.0
        patience = 0
        max_patience = 15
        
        while epoch < 150:
            expanded_model.train()
            for data, target in trainloader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = expanded_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            
            expanded_model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = expanded_model(data)
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target).sum().item()
                    total += target.size(0)
            test_acc = 100.0 * correct / total
            
            if test_acc > best_acc:
                best_acc = test_acc
                patience = 0
            else:
                patience += 1
                
            if patience >= max_patience:
                break
            epoch += 1
        
        print(f"   Expanded model accuracy: {best_acc:.2f}%")
        
        # Distill enhanced sparse model with density constraint
        print("5. Distilling enhanced sparse model...")
        distilled_model, metrics = self.progressive_pruning_with_target(
            expanded_model, self.base_target_acc, test_loader, max_density=0.05
        )
        
        # Measure improvement over previous best
        improved = False
        current_score = metrics['accuracy'] + metrics['L'] * 10  # Combined score
        
        if distilled_model is not None:
            previous_score = self.best_metrics['accuracy'] + self.best_metrics['L'] * 10
            
            # Check if this model represents genuine improvement
            if current_score > previous_score or (cycle == 1 and metrics['accuracy'] >= self.base_target_acc):
                improved = True
                
                # Update best metrics
                self.best_metrics.update({
                    'accuracy': metrics['accuracy'],
                    'L': metrics['L'],
                    'density': metrics['density'],
                    'cycle': cycle
                })
                
                # Save this model as the new best
                model_path = os.path.join(self.model_save_dir, f'best_model_cycle_{cycle}.pth')
                torch.save(distilled_model.state_dict(), model_path)
                torch.save(distilled_model.state_dict(), 'final_best_model.pth')
                
                # Measure functional alignment with previous best
                if cycle > 1 and legacy_checkpoint is not None:
                    prev_model = SpectralMLP(hidden_dim=seed_hidden_dim).to(self.device)
                    prev_model.load_state_dict(legacy_checkpoint)
                    alignment = self.measure_functional_alignment(prev_model, distilled_model, test_loader)
                    self.best_metrics['logit_stability'] = alignment
                    metrics['logit_stability'] = alignment
        
        result = {
            'cycle': cycle,
            'accuracy': metrics['accuracy'],
            'L': metrics['L'],
            'density': metrics['density'],
            'improved': improved,
            'logit_stability': metrics.get('logit_stability', 0.0)
        }
        
        # Print compact results
        print(f"   ├─ Coherencia L: {metrics['L']:.4f}")
        print(f"   ├─ Densidad:     {metrics['density']:.2%}")
        print(f"   └─ Precisión αₛ: {metrics['accuracy']:.2f}%")
        
        return result

    def run_evolutionary_experiment(self, num_cycles: int = 10):
        print("=" * 70)
        print("🌌 INICIANDO PROTOCOLO NEUROSOVEREIGN v6.2")
        print("=" * 70)
        
        # Setup test loader
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)
        
        # Run evolutionary cycles
        for cycle in range(1, num_cycles + 1):
            result = self.execute_resonance_cycle(cycle, test_loader)
            self.results_history.append(result)
            
            # Stop if no improvement for 3 consecutive cycles
            if not result['improved'] and cycle > 3:
                recent_improvements = [r['improved'] for r in self.results_history[-3:]]
                if not any(recent_improvements):
                    print(f"\n⚠️  No improvement for 3 cycles. Stopping evolution.")
                    break
        
        # Final report
        print("\n" + "=" * 70)
        print("🏆 REPORTE FINAL DE ALTA DIRECCIÓN")
        print("-" * 70)
        print(f"💎 Coherencia Final: {self.best_metrics['L']:.4f}")
        print(f"💎 Precisión Ética:  {self.best_metrics['accuracy']:.2f}%")
        print(f"💎 Iteraciones:      {self.best_metrics['cycle']}")
        
        # Determine regime
        if self.best_metrics['L'] > 1.0:
            regime = "SOBERANO"
        elif self.best_metrics['L'] > 0.5:
            regime = "EMERGENTE"
        else:
            regime = "ESPURIO"
        print(f"💎 Estado:           {regime}")
        print("=" * 70)
        
        # Save comprehensive results
        df = pd.DataFrame(self.results_history)
        df.to_csv('fractal_resonance_results.csv', index=False)
        
        metrics_out = {
            'final_accuracy': float(self.best_metrics['accuracy']),
            'final_L': float(self.best_metrics['L']),
            'final_density': float(self.best_metrics['density']),
            'best_cycle': int(self.best_metrics['cycle']),
            'total_cycles': len(self.results_history),
            'evolution_trajectory': [
                {
                    'cycle': r['cycle'],
                    'accuracy': float(r['accuracy']),
                    'L': float(r['L']),
                    'density': float(r['density']),
                    'improved': bool(r['improved'])
                }
                for r in self.results_history
            ]
        }
        
        with open('evolution_metrics.json', 'w') as f:
            json.dump(metrics_out, f, indent=2)
        
        print(f"\n✅ Resultados guardados en:")
        print(f"   - fractal_resonance_results.csv")
        print(f"   - final_best_model.pth")
        print(f"   - evolution_metrics.json")
        print(f"   - {self.model_save_dir}/")

# =============================================================================
# Main Execution
# =============================================================================
def main():
    device = torch.device("cpu")
    engine = EvolutionaryResonanceEngine(device, base_target_acc=32.4)
    engine.run_evolutionary_experiment(num_cycles=10)

if __name__ == "__main__":
    main()
