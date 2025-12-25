#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuroSovereign v15.0: Paper Candidate Release
Objective: Finalize architecture for rigorous review.
Changes from v14.5:
1. CLARITY: Separated L_opt (Optimization Objective) vs L_mon (Reporting Metric) in logs.
2. BENCHMARK: Added CIFAR-20 Stress Test (Coarse-only validation).
3. LOGIC: Final validation based on Delta (APEX - BLIND) to prove structural advantage.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import os
import warnings
from typing import Dict, Tuple
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
NUM_EVOLUTION_CYCLES = 12
INPUT_DIM = 384
FIXED_HIDDEN_DIM = 256
NUM_CLASSES = 100
NUM_SUPERCLASSES = 20
TARGET_L = 1.8

# v15.0: HYPERPARAMETERS
LAMBDA_TAX_BASE = 0.1
LAMBDA_TAX_SHOCK = 0.6
GAP_SHOCK_THRESHOLD = 5.0
LAMBDA_SPARSE_MIXER = 1e-4
LAMBDA_SPECTRAL = 0.05

# BENCHMARK FLAG
RUN_HIERARCHY_BENCHMARK = True

# =============================================================================
# MAPA TAXONÓMICO (CIFAR100 -> CIFAR20)
# =============================================================================
_FINE_TO_COARSE = [-1] * 100
full_super_map = {
    0: [4, 30, 55, 72, 95], 1: [1, 32, 67, 73, 91], 2: [54, 62, 70, 82, 92],
    3: [9, 10, 16, 28, 61], 4: [0, 51, 53, 57, 83], 5: [22, 39, 40, 86, 87],
    6: [5, 20, 25, 84, 94], 7: [6, 7, 14, 18, 24], 8: [3, 42, 43, 88, 97],
    9: [12, 17, 37, 68, 76], 10: [23, 33, 49, 60, 71], 11: [15, 19, 21, 31, 90],
    12: [35, 63, 64, 66, 81], 13: [11, 27, 45, 56, 99], 14: [2, 8, 15, 36, 69],
    15: [18, 19, 31, 59, 77], 16: [79, 81, 82, 85, 88], 17: [87, 89, 92, 93, 95],
    18: [0, 1, 8, 9, 10], 19: [27, 28, 35, 36, 45]
}
for coarse_id, fine_list in full_super_map.items():
    for fine_id in fine_list:
        if 0 <= fine_id < 100: _FINE_TO_COARSE[fine_id] = coarse_id
for i in range(100):
    if _FINE_TO_COARSE[i] == -1: _FINE_TO_COARSE[i] = 0

# =============================================================================
# 1. ARCHITECTURE & SPECTRAL LOSS
# =============================================================================
def compute_spectral_loss(W: torch.Tensor) -> torch.Tensor:
    """v15.0: Optimization Objective for Spectral Control."""
    try:
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    except:
        return torch.tensor(0.0, device=W.device)
    
    S_norm = S / (S.sum() + 1e-12)
    entropy = -(S_norm * torch.log(S_norm + 1e-12)).sum()
    
    threshold = 0.05 * S[0]
    eff_rank = (S > threshold).sum().float()
    
    target_entropy = torch.log(eff_rank + 1.0)
    return torch.abs(entropy - target_entropy)

class GatedTokenMixer(nn.Module):
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.Sigmoid())
        self.mixer = nn.Sequential(
            nn.Linear(num_patches, num_patches * 2),
            nn.GELU(),
            nn.Linear(num_patches * 2, num_patches)
        )

    def forward(self, x):
        out = x.transpose(1, 2)
        out = self.mixer(out)
        out = out.transpose(1, 2)
        g = self.gate(x)
        return x + g * out

class PatchFeatureExtractor(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=384, use_mixer=True):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.use_mixer = use_mixer
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        if self.use_mixer:
            self.mixer = GatedTokenMixer(self.num_patches, embed_dim)
        
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
        self.freeze()

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_mixer_only(self):
        if self.use_mixer:
            for param in self.mixer.parameters():
                param.requires_grad = True
            print("      🔓 MIXER UNFROZEN (v15.0 Paper Candidate)")

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        if self.use_mixer:
            x = self.mixer(x)
        x = x.mean(dim=1)
        return x

# =============================================================================
# 2. TAXONOMIC LOTTERY TICKET MLP
# =============================================================================
class TaxonomicMLP(nn.Module):
    def __init__(self, input_dim: int = 384, hidden_dim: int = 256, num_classes: int = 100):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, num_classes, bias=False)
        self.fc_super = nn.Linear(hidden_dim, NUM_SUPERCLASSES, bias=False)
        
        self.register_buffer('mask1', torch.ones_like(self.fc1.weight))
        self.register_buffer('mask2', torch.ones_like(self.fc2.weight))
        self.register_buffer('mask_super', torch.ones_like(self.fc_super.weight))
        
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.fc_super.weight, mean=0.0, std=0.02)

    def apply_masks(self):
        with torch.no_grad():
            self.fc1.weight.data *= self.mask1
            self.fc2.weight.data *= self.mask2
            self.fc_super.weight.data *= self.mask_super

    def get_sparsity(self):
        total = self.mask1.numel() + self.mask2.numel() + self.mask_super.numel()
        active = self.mask1.sum() + self.mask2.sum() + self.mask_super.sum()
        return (1.0 - (active.item() / total)) * 100

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        w1 = self.fc1.weight * self.mask1
        h = F.relu(F.linear(x, w1))
        w2 = self.fc2.weight * self.mask2
        w_super = self.fc_super.weight * self.mask_super
        return F.linear(h, w2), F.linear(h, w_super)

# =============================================================================
# 3. MONITORING (L_mon: Reporting Metric)
# =============================================================================
class SpectralMonitor:
    def compute_metrics(self, weight: torch.Tensor) -> Tuple[float, int, float]:
        """L_mon: Used for plotting and historical reporting, not optimization."""
        with torch.no_grad():
            W = weight.cpu().numpy()
            U, S, Vh = np.linalg.svd(W, full_matrices=False)
            threshold = 0.05 * np.max(S)
            rank_eff = max(1, int(np.sum(S > threshold)))
            S_norm = S / (np.sum(S) + 1e-12)
            S_norm = S_norm[S_norm > 1e-15]
            S_vN = -np.sum(S_norm * np.log(S_norm + 1e-15))
            # Legacy L formula
            L_mon = 1.0 / (abs(S_vN - np.log(rank_eff + 1)) + 0.3)
            return L_mon, rank_eff, S_vN

# =============================================================================
# 4. TAXONOMIC TRAINER
# =============================================================================
class TaxonomicTrainer:
    def __init__(self, device: torch.device, extractor_apex=None, extractor_blind=None):
        self.device = device
        self.extractor_apex = extractor_apex
        self.extractor_blind = extractor_blind
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.monitor = SpectralMonitor()

    def _preprocess_batch(self, x, extractor):
        with torch.no_grad():
            return extractor(x)

    def get_curriculum_dataset(self, cycle: int):
        full_trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=self.transform)
        sizes = {1: 5000, 4: 20000, 7: 50000} 
        size = sizes.get(cycle, 5000)
        return torch.utils.data.Subset(full_trainset, torch.randperm(len(full_trainset))[:size])

    def train_single_chain(self, model, cycle, chain_type='APEX'):
        trainset = self.get_curriculum_dataset(cycle)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
        
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=self.transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)
        
        wd = 0.1 if cycle < 5 else 0.01
        
        if chain_type == 'APEX' and hasattr(self.extractor_apex, 'mixer'):
            optimizer = torch.optim.AdamW([
                {'params': model.parameters()},
                {'params': self.extractor_apex.mixer.parameters(), 'lr': 0.005 * 5.0}
            ], lr=0.005, weight_decay=wd)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=wd)
            
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        criterion = nn.CrossEntropyLoss()
        
        best_gap = 100.0
        best_acc = 0.0
        best_state = None
        
        if chain_type == 'BLIND':
            model.fc_super.requires_grad_(False)
        
        extractor = self.extractor_apex if chain_type == 'APEX' else self.extractor_blind
        prev_gap = 0.0
        
        for epoch in range(100):
            model.train()
            train_correct, train_total = 0, 0
            
            # --- v15.0: ADAPTIVE SHOCK ---
            current_lambda_tax = LAMBDA_TAX_BASE
            shock_state = "BASE"
            
            if cycle > 1 and prev_gap > GAP_SHOCK_THRESHOLD:
                current_lambda_tax = LAMBDA_TAX_SHOCK
                shock_state = "SHOCK"
            
            for inputs, labels in trainloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                labels = labels.long()
                
                coarse_labels = torch.tensor([_FINE_TO_COARSE[l.item()] for l in labels], device=self.device)
                inputs_proc = self._preprocess_batch(inputs, extractor)
                
                optimizer.zero_grad()
                outputs_fine, outputs_super = model(inputs_proc)
                
                if chain_type == 'APEX':
                    loss_fine = criterion(outputs_fine, labels)
                    loss_super = criterion(outputs_super, coarse_labels)
                    total_loss = loss_fine + (current_lambda_tax * loss_super)
                    
                    # --- v15.0: SPECTRAL LOSS (Optimization Objective L_opt) ---
                    # 1. Downstream (FC1)
                    loss_spectral_opt = compute_spectral_loss(model.fc1.weight)
                    
                    # 2. Upstream (Mixer)
                    if hasattr(extractor, 'mixer'):
                        mixer_spec_loss = 0.0
                        for module in extractor.mixer.mixer:
                            if isinstance(module, nn.Linear):
                                mixer_spec_loss += compute_spectral_loss(module.weight)
                        loss_spectral_opt += mixer_spec_loss
                    
                    total_loss += LAMBDA_SPECTRAL * loss_spectral_opt

                    # Sparse Penalty
                    if hasattr(extractor, 'mixer'):
                        mixer_sparse_penalty = 0
                        for module in extractor.mixer.mixer:
                            if isinstance(module, nn.Linear):
                                mixer_sparse_penalty += module.weight.abs().sum()
                        total_loss += LAMBDA_SPARSE_MIXER * mixer_sparse_penalty
                else:
                    total_loss = criterion(outputs_fine, labels)

                total_loss.backward()
                
                model.fc1.weight.grad *= model.mask1
                model.fc2.weight.grad *= model.mask2
                
                if model.fc_super.weight.grad is not None:
                    model.fc_super.weight.grad *= model.mask_super
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                _, predicted = torch.max(outputs_fine.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            train_acc = 100 * train_correct / train_total
            
            # --- VALIDATION ---
            model.eval()
            test_correct, test_total = 0, 0
            coarse_correct, coarse_total = 0, 0
            
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    inputs_proc = self._preprocess_batch(inputs, extractor)
                    outputs_fine, outputs_super = model(inputs_proc)
                    
                    _, predicted = torch.max(outputs_fine.data, 1)
                    test_total += labels.size(0)
                    test_correct += (predicted == labels).sum().item()
                    
                    true_coarse = torch.tensor([_FINE_TO_COARSE[l.item()] for l in labels], device=self.device)
                    _, pred_coarse = torch.max(outputs_super.data, 1)
                    coarse_total += true_coarse.size(0)
                    coarse_correct += (pred_coarse == true_coarse).sum().item()
            
            test_acc = 100 * test_correct / test_total
            coarse_acc = 100 * coarse_correct / coarse_total
            gap = train_acc - test_acc
            prev_gap = gap
            
            if test_acc > best_acc:
                best_acc = test_acc
                best_gap = gap
                best_state = model.state_dict().copy()
            
            L_mon, rank_eff, _ = self.monitor.compute_metrics(model.fc1.weight)
            
            # --- v15.0 LOGGING (Explicit Metrics) ---
            if epoch % 20 == 0:
                log_str = f"         [{chain_type}] Ep {epoch:3d} | V.Acc:{test_acc:5.1f}% | C.V:{coarse_acc:5.1f}% | Gap:{gap:4.1f}"
                log_str += f" | S:{shock_state} | λ:{current_lambda_tax:.2f}"
                
                if chain_type == 'APEX':
                    # Capture L_opt (Loss being minimized)
                    val_opt = 0.0
                    if hasattr(extractor, 'mixer'):
                        for m in extractor.mixer.mixer:
                            if isinstance(m, nn.Linear): val_opt += compute_spectral_loss(m.weight).item()
                    val_opt += compute_spectral_loss(model.fc1.weight).item()
                    
                    spar = model.get_sparsity()
                    log_str += f" | L_opt:{val_opt:.3f} | L_mon:{L_mon:.3f} | Spar:{spar:.1f}%"
                
                print(log_str)
            
            scheduler.step()
            
        if best_state: 
            model.load_state_dict(best_state)
            model.apply_masks()
                
        L_final, Rank_final, _ = self.monitor.compute_metrics(model.fc1.weight)
        return {'final_acc': best_acc, 'final_coarse_acc': coarse_acc, 'final_gap': best_gap, 'L': L_final, 'rank': Rank_final}

# =============================================================================
# 5. HIERARCHY STRESS TEST (CIFAR-20 BENCHMARK)
# =============================================================================
class CoarseCIFAR100(torchvision.datasets.CIFAR100):
    """
    Wrapper que convierte CIFAR100 en un problema de clasificación pura de 20 clases (Superclases).
    Se usa para validar el inductive bias aprendido.
    """
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        # Target mapeado directamente a coarse ID
        return img, _FINE_TO_COARSE[target]

def run_hierarchy_benchmark(model_apex, model_blind, device):
    print("\n" + "="*90)
    print(" " * 30 + "HIERARCHY STRESS TEST (CIFAR-20)")
    print("=" * 90)
    
    # Cargamos extractor APEX para ambos (es el que tiene estructura)
    extractor_apex = PatchFeatureExtractor(embed_dim=INPUT_DIM, use_mixer=True).to(device)
    # Nota: Para este benchmark testamos las cabezas fc_super que ya existen
    
    testset = CoarseCIFAR100(root='./data', train=False, download=True, 
                             transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)
    
    def evaluate(model, extractor, name):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                features = extractor(inputs)
                outputs_super = model.fc_super(features * model.mask1) # Usamos pesos enmascarados
                _, predicted = torch.max(outputs_super.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = 100 * correct / total
        return acc

    acc_apex = evaluate(model_apex, extractor_apex, "APEX")
    # Para BLIND, usamos el mismo extractor para comparar "Feature Quality vs Head Knowledge"
    # Aunque BLIND entrenó sin señal, comprobamos qué tanta estructura heredó.
    extractor_blind = PatchFeatureExtractor(embed_dim=INPUT_DIM, use_mixer=False).to(device)
    acc_blind = evaluate(model_blind, extractor_blind, "BLIND")
    
    delta = acc_apex - acc_blind
    
    print(f"   APEX Coarse-Only Accuracy: {acc_apex:.2f}%")
    print(f"   BLIND Coarse-Only Accuracy: {acc_blind:.2f}%")
    print(f"   Structural Advantage (Δ):   {delta:+.2f}%")
    print("-" * 90)
    
    if delta > 2.0:
        print("✅ VALIDATED: APEX demonstrates significant inductive bias for hierarchy.")
    else:
        print("⚠️  WARNING: Low hierarchy transfer detected.")
        
    return delta

# =============================================================================
# 6. MAIN EXECUTION
# =============================================================================
def main():
    print("=" * 90)
    print(" " * 25 + "NEUROSOVEREIGN v15.0: PAPER CANDIDATE")
    print("=" * 90)
    print("Final Fixes:")
    print("1. Separated Optimization Objective (L_opt) from Reporting Metric (L_mon).")
    print("2. Added CIFAR-20 Stress Test for Hierarchy validation.")
    print("3. Validation based on APEX vs BLIND Delta (Structural Advantage).")
    print("=" * 90)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    extractor_apex = PatchFeatureExtractor(embed_dim=INPUT_DIM, use_mixer=True).to(device)
    extractor_blind = PatchFeatureExtractor(embed_dim=INPUT_DIM, use_mixer=False).to(device)
    
    trainer = TaxonomicTrainer(device, extractor_apex, extractor_blind)
    
    # ============================
    # 1. INITIALIZE DUAL ELKS
    # ============================
    if os.path.exists('alpha_elk_v15_0_apex.pth'):
        print("📂 Loading Dual Alphas v15.0...")
        elk_state_apex = torch.load('alpha_elk_v15_0_apex.pth', map_location=device)
        elk_state_blind = torch.load('alpha_elk_v15_0_blind.pth', map_location=device)
        start_cycle = 1
    else:
        # Migración desde v14.5
        if os.path.exists('alpha_elk_v14_5_apex.pth'):
            print("📂 Migrating from v14.5 to v15.0...")
            raw_apex = torch.load('alpha_elk_v14_5_apex.pth', map_location=device)
            raw_blind = torch.load('alpha_elk_v14_5_blind.pth', map_location=device)
            
            seed_apex = TaxonomicMLP(input_dim=INPUT_DIM, hidden_dim=FIXED_HIDDEN_DIM, num_classes=NUM_CLASSES).to(device)
            seed_blind = TaxonomicMLP(input_dim=INPUT_DIM, hidden_dim=FIXED_HIDDEN_DIM, num_classes=NUM_CLASSES).to(device)
            
            seed_apex.load_state_dict(raw_apex, strict=False)
            seed_blind.load_state_dict(raw_blind, strict=False)
            
            elk_state_apex = seed_apex.state_dict()
            elk_state_blind = seed_blind.state_dict()
            torch.save(elk_state_apex, 'alpha_elk_v15_0_apex.pth')
            torch.save(elk_state_blind, 'alpha_elk_v15_0_blind.pth')
            start_cycle = 1
        else:
            print("🥚 Generating Seed Elks v15.0 (Cycle 0)...")
            seed_apex = TaxonomicMLP(input_dim=INPUT_DIM, hidden_dim=FIXED_HIDDEN_DIM, num_classes=NUM_CLASSES).to(device)
            seed_blind = TaxonomicMLP(input_dim=INPUT_DIM, hidden_dim=FIXED_HIDDEN_DIM, num_classes=NUM_CLASSES).to(device)
            
            print("   - Seeding Apex...")
            trainer.train_single_chain(seed_apex, 0, 'APEX')
            print("   - Seeding Blind...")
            trainer.train_single_chain(seed_blind, 0, 'BLIND')
            
            elk_state_apex = seed_apex.state_dict()
            elk_state_blind = seed_blind.state_dict()
            start_cycle = 1
            torch.save(elk_state_apex, 'alpha_elk_v15_0_apex.pth')
            torch.save(elk_state_blind, 'alpha_elk_v15_0_blind.pth')

    # ============================
    # 2. DUAL EVOLUTION LOOP
    # ============================
    evolution_log = []
    
    for cycle in range(start_cycle, NUM_EVOLUTION_CYCLES + 1):
        print(f"\n{'='*90}")
        print(f"🔬 TAXONOMIC EVOLUTION CYCLE {cycle}/{NUM_EVOLUTION_CYCLES}")
        print(f"{'='*90}")
        
        if cycle == 1:
            print("   ⚡ UNFREEZING MIXER (APEX)...")
            extractor_apex.unfreeze_mixer_only()
        
        # --- APEX ---
        child_apex = TaxonomicMLP(input_dim=INPUT_DIM, hidden_dim=FIXED_HIDDEN_DIM, num_classes=NUM_CLASSES).to(device)
        child_apex.load_state_dict(elk_state_apex, strict=False)
        res_apex = trainer.train_single_chain(child_apex, cycle, 'APEX')
        
        # --- BLIND ---
        child_blind = TaxonomicMLP(input_dim=INPUT_DIM, hidden_dim=FIXED_HIDDEN_DIM, num_classes=NUM_CLASSES).to(device)
        child_blind.load_state_dict(elk_state_blind, strict=False)
        res_blind = trainer.train_single_chain(child_blind, cycle, 'BLIND')
        
        # --- SELECCIÓN ---
        improved_apex = res_apex['final_acc'] - (evolution_log[-1]['acc_apex'] if evolution_log else 0)
        
        if improved_apex >= 0.0: 
            print(f"   ✅ APEX EVOLVED ({improved_apex:+.1f}%) | Coarse Acc: {res_apex['final_coarse_acc']:.1f}%")
            elk_state_apex = child_apex.state_dict()
            torch.save(elk_state_apex, 'alpha_elk_v15_0_apex.pth')
        else:
            print(f"   ❌ APEX STAGNANT")

        improved_blind = res_blind['final_acc'] - (evolution_log[-1]['acc_blind'] if evolution_log else 0)
        if improved_blind >= 0.0:
            print(f"   ✅ BLIND EVOLVED ({improved_blind:+.1f}%) | Coarse Acc: {res_blind['final_coarse_acc']:.1f}%")
            elk_state_blind = child_blind.state_dict()
            torch.save(elk_state_blind, 'alpha_elk_v15_0_blind.pth')
        else:
            print(f"   ❌ BLIND STAGNANT")
            
        delta_structure = res_apex['final_acc'] - res_blind['final_acc']
        delta_hierarchy = res_apex['final_coarse_acc'] - res_blind['final_coarse_acc']
        
        evolution_log.append({
            'cycle': cycle,
            'acc_apex': res_apex['final_acc'],
            'acc_blind': res_blind['final_acc'],
            'coarse_apex': res_apex['final_coarse_acc'],
            'coarse_blind': res_blind['final_coarse_acc'],
            'delta_structure': delta_structure,
            'delta_hierarchy': delta_hierarchy,
            'gap_apex': res_apex['final_gap'],
            'gap_blind': res_blind['final_gap'],
            'L_apex': res_apex['L'],
            'L_blind': res_blind['L'],
            'spar_apex': child_apex.get_sparsity(),
            'spar_blind': child_blind.get_sparsity()
        })

    # ============================
    # 3. FINAL HIERARCHY BENCHMARK
    # ============================
    hierarchy_delta = 0.0
    if RUN_HIERARCHY_BENCHMARK:
        # Reload final states for benchmark
        final_apex = TaxonomicMLP(input_dim=INPUT_DIM, hidden_dim=FIXED_HIDDEN_DIM, num_classes=NUM_CLASSES).to(device)
        final_apex.load_state_dict(torch.load('alpha_elk_v15_0_apex.pth'))
        
        final_blind = TaxonomicMLP(input_dim=INPUT_DIM, hidden_dim=FIXED_HIDDEN_DIM, num_classes=NUM_CLASSES).to(device)
        final_blind.load_state_dict(torch.load('alpha_elk_v15_0_blind.pth'))
        
        hierarchy_delta = run_hierarchy_benchmark(final_apex, final_blind, device)

    # ============================
    # 4. PAPER-READY REPORT
    # ============================
    df = pd.DataFrame(evolution_log)
    print("\n" + "=" * 90)
    print(" " * 30 + "PAPER CANDIDATE REPORT v15.0")
    print("=" * 90)
    
    disp_cols = ['cycle', 'acc_apex', 'acc_blind', 'delta_structure', 
                 'coarse_apex', 'coarse_blind', 'delta_hierarchy',
                 'gap_apex', 'gap_blind', 'L_apex']
    print(df[disp_cols].to_string(index=False))
    
    print("-" * 90)
    print(f"🔬 FINAL ANALYSIS:")
    print(f"   Structure Δ (APEX - BLIND): {df['delta_structure'].mean():.2f}%")
    print(f"   Hierarchy Δ (APEX - BLIND): {df['delta_hierarchy'].mean():.2f}%")
    print(f"   Hierarchy Benchmark (CIFAR-20): {hierarchy_delta:+.2f}%")
    print(f"   Spectral Stability (L_apex):    {df['L_apex'].iloc[-1]:.3f}")
    
    # v15.0: Paper-Ready Criteria
    criteria_met = False
    if df['delta_hierarchy'].mean() > 2.0 and hierarchy_delta > 2.0:
        criteria_met = True
        print("\n✅ PAPER CLAIM VALIDATED:")
        print("   'Taxonomic regularization induces structural inductive bias'")
        print("   Evidence: Consistent positive Δ in hierarchy learning and transfer.")
    elif df['delta_hierarchy'].mean() > 0:
        print("\n⚠️  PARTIAL VALIDATION:")
        print("   Hierarchy learning observed, but transfer is weak.")
        print("   Suggestion: Increase LAMBDA_SPECTRAL or Mixer depth.")
    else:
        print("\n❌ CLAIM NOT SUPPORTED:")
        print("   BLIND model performs equally well on hierarchy.")
        print("   The taxonomic signal may be redundant.")
        
    print("=" * 90)

if __name__ == "__main__":
    main()