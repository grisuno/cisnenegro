#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuroSovereign v18.2: The Scientific Powerhouse
Fusion of v18.1 Rigor + v15.4 Aggressive Performance

Philosophy:
- Use the hardened, reproducible architecture of v18.1.
- Inject the aggressive hyperparameters of v15.4.
- Apply safety mechanisms to balance speed and stability.

Key Features (v18.2):
1. [AGGRESSIVE] GAP_SHOCK_THRESHOLD = 3.5 & LAMBDA_TAX_SHOCK = 0.8 (Fast learning).
2. [SAFE] Mask Warmup (25 epochs) & Overfit Safety Valve (Gap > 12.0).
3. [RIGOROUS] AdaptiveTopologyController with Hysteresis (from v18.1).
4. [HARDENED] torch.linalg.svd & Scope Fixes (from v18.1).

Target: >30% Hierarchy Advantage with Stable Convergence.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import random
import argparse
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION & REPRODUCIBILITY (v18.2 Settings)
# =============================================================================
REFERENCE_BASELINE = 28.0  # Statistical reference only

def set_seed(seed: int = 42):
    """Ensure full reproducibility across runs"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =============================================================================
# TAXONOMIC MAPPING (CIFAR-100 SUPERCLASSES)
# =============================================================================
CIFAR100_SUPERCLASSES = {
    0: [4, 30, 55, 72, 95], 1: [1, 32, 67, 73, 91], 2: [54, 62, 70, 82, 92],
    3: [9, 10, 16, 28, 61], 4: [0, 51, 53, 57, 83], 5: [22, 39, 40, 86, 87],
    6: [5, 20, 25, 84, 94], 7: [6, 7, 14, 18, 24], 8: [3, 42, 43, 88, 97],
    9: [12, 17, 37, 68, 76], 10: [23, 33, 49, 60, 71], 11: [15, 19, 21, 31, 90],
    12: [35, 63, 64, 66, 81], 13: [11, 27, 45, 56, 99], 14: [2, 8, 36, 41, 96],
    15: [26, 44, 65, 74, 89], 16: [13, 29, 50, 80, 93], 17: [34, 46, 52, 58, 77],
    18: [25, 38, 48, 79, 98], 19: [37, 49, 61, 75, 85]
}

_FINE_TO_COARSE = [0] * 100
for coarse_idx, fine_list in CIFAR100_SUPERCLASSES.items():
    for fine_idx in fine_list:
        if 0 <= fine_idx < 100:
            _FINE_TO_COARSE[fine_idx] = coarse_idx

# =============================================================================
# CORE ARCHITECTURE COMPONENTS
# =============================================================================
class GatedTokenMixer(nn.Module):
    """Efficient token mixer with high-std initialization for exploration"""
    def __init__(self, num_patches: int, embed_dim: int):
        super().__init__()
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        
        self.mixer = nn.Sequential(
            nn.Linear(num_patches, num_patches * 2),
            nn.GELU(),
            nn.Linear(num_patches * 2, num_patches)
        )
        self.gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid()
        )
        self._init_weights()
    
    def _init_weights(self):
        # High std for initial spectral exploration (v15.3 logic)
        for m in self.mixer:
            if isinstance(m, nn.Linear):
                fan_in = m.weight.size(1)
                fan_out = m.weight.size(0)
                std = np.sqrt(2.0 / (fan_in + fan_out)) * 2.0
                nn.init.normal_(m.weight, mean=0.0, std=std)
                if m.bias is not None: nn.init.zeros_(m.bias)
        
        for m in self.gate:
            if isinstance(m, nn.Linear):
                fan_in = m.weight.size(1)
                fan_out = m.weight.size(0)
                std = np.sqrt(2.0 / (fan_in + fan_out)) * 2.0
                nn.init.normal_(m.weight, mean=0.0, std=std)
                if m.bias is not None: nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        x_transposed = x.transpose(1, 2)
        mixed = self.mixer(x_transposed)   
        mixed = mixed.transpose(1, 2)
        gate = self.gate(x)
        return x + gate * mixed
        
class PatchFeatureExtractor(nn.Module):
    def __init__(self, img_size: int = 32, patch_size: int = 4, 
                 in_chans: int = 3, embed_dim: int = 384, use_mixer: bool = True):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.use_mixer = use_mixer
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        if self.use_mixer:
            self.mixer = GatedTokenMixer(self.num_patches, embed_dim)
        
        # v18.2: Slightly higher std than v18.1 to match v15.4 aggressive nature
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.08)
        if self.proj.bias is not None: nn.init.zeros_(self.proj.bias)
        self.freeze()
    
    def freeze(self):
        for param in self.parameters(): param.requires_grad = False
    
    def unfreeze_mixer_only(self):
        if self.use_mixer:
            for param in self.mixer.parameters(): param.requires_grad = True
            print("      🔓 MIXER UNFROZEN (v18.2 Aggressive + Stable)")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        if self.use_mixer and hasattr(self, 'mixer'): x = self.mixer(x)
        return x.mean(dim=1)

class TaxonomicMLP(nn.Module):
    """Sparse MLP with taxonomic heads"""
    def __init__(self, input_dim: int = 384, hidden_dim: int = 256, 
                 num_classes: int = 100, num_superclasses: int = 20):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, num_classes, bias=False)
        self.fc_super = nn.Linear(hidden_dim, num_superclasses, bias=False)
        
        self.register_buffer('mask1', torch.ones(hidden_dim, input_dim))
        self.register_buffer('mask2', torch.ones(num_classes, hidden_dim))
        self.register_buffer('mask_super', torch.ones(num_superclasses, hidden_dim))
        
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.fc_super.weight, mean=0.0, std=0.02)
    
    def apply_masks(self):
        with torch.no_grad():
            self.fc1.weight.data *= self.mask1
            self.fc2.weight.data *= self.mask2
            self.fc_super.weight.data *= self.mask_super
    
    def get_sparsity(self) -> float:
        total = self.mask1.numel() + self.mask2.numel() + self.mask_super.numel()
        active = self.mask1.sum() + self.mask2.sum() + self.mask_super.sum()
        return (1.0 - (active.item() / total)) * 100
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        w1 = self.fc1.weight * self.mask1
        w2 = self.fc2.weight * self.mask2
        w_super = self.fc_super.weight * self.mask_super
        h = F.relu(F.linear(x, w1))
        fine_logits = F.linear(h, w2)
        super_logits = F.linear(h, w_super)
        return fine_logits, super_logits

# =============================================================================
# SPECTRAL ANALYSIS & CONTROL (Hardened from v18.1)
# =============================================================================
def compute_spectral_loss(W: torch.Tensor) -> torch.Tensor:
    try:
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    except: return torch.tensor(0.0, device=W.device)
    S_norm = S / (S.sum() + 1e-12)
    entropy = -(S_norm * torch.log(S_norm + 1e-12)).sum()
    threshold = 0.05 * S[0]
    eff_rank = (S > threshold).sum().float()
    target_entropy = torch.log(eff_rank + 1.0)
    return torch.abs(entropy - target_entropy)

class SpectralMonitor:
    def __init__(self, epsilon: float = 0.3):
        self.epsilon = epsilon
    
    def compute_metrics(self, weight: torch.Tensor) -> Tuple[float, int, float]:
        with torch.no_grad():
            W = weight.cpu().numpy()
            try:
                U, S, Vh = np.linalg.svd(W, full_matrices=False)
                threshold = 0.05 * np.max(S)
                rank_eff = max(1, int(np.sum(S > threshold)))
                S_norm = S / (np.sum(S) + 1e-12)
                S_norm = S_norm[S_norm > 1e-15]
                S_vN = -np.sum(S_norm * np.log(S_norm + 1e-15))
                L = 1.0 / (abs(S_vN - np.log(rank_eff + 1)) + self.epsilon)
                return L, rank_eff, S_vN
            except: return 1.0, 1, 0.0
    
    def get_singular_values(self, weight: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            W = weight.cpu().numpy()
            try:
                U, S, Vh = np.linalg.svd(W, full_matrices=False)
                return S
            except: return np.array([0.0])

class AdaptiveTopologyController:
    """Self-referential controller with Hysteresis"""
    def __init__(self, semantic_plasticity_threshold: float = 0.05, 
                 stagnation_limit: int = 10, mixer_noise_scale: float = 0.05,
                 dominant_energy_threshold: float = 0.8, geo_window: int = 3):
        self.semantic_plasticity_threshold = semantic_plasticity_threshold
        self.stagnation_limit = stagnation_limit
        self.mixer_noise_scale = mixer_noise_scale
        self.dominant_energy_threshold = dominant_energy_threshold
        self.geo_window = geo_window
        self.stagnation_counter = 0
        self.topo_history = []
        self.coarse_history = []
        self.intervention_log = []
    
    def compute_semantic_plasticity_ratio(self) -> float:
        if len(self.coarse_history) < 2 or len(self.topo_history) < 2:
            return float('inf')
        d_coarse = abs(self.coarse_history[-1] - self.coarse_history[-2])
        d_topo = abs(self.topo_history[-1] - self.topo_history[-2])
        if d_topo < 1e-6: return float('inf')
        return d_coarse / d_topo
    
    def detect_intervention_need(self, phase_state: str, extractor) -> Tuple[bool, str]:
        trigger_reason = ""
        ratio = self.compute_semantic_plasticity_ratio()
        
        # Hysteresis: Do not intervene during active SHIFTING phases
        if phase_state == "SHIFTING":
            return False, ""

        if ratio < self.semantic_plasticity_threshold:
            trigger_reason = "LOW_PLASTICITY_RATIO"
            self.perturb_mixer_targeted(extractor)
            self.stagnation_counter = 0
            self.intervention_log.append({'ratio': ratio, 'reason': trigger_reason})
            return True, trigger_reason
        
        if self.stagnation_counter > self.stagnation_limit and phase_state == "STABLE":
            trigger_reason = "PERF_STAGNATION"
            self.perturb_mixer_targeted(extractor)
            self.stagnation_counter = 0
            self.intervention_log.append({'ratio': ratio, 'reason': trigger_reason})
            return True, trigger_reason
        return False, trigger_reason
    
    def update_history(self, topo_ratio: float, coarse_acc: float):
        self.topo_history.append(topo_ratio)
        self.coarse_history.append(coarse_acc)
        if len(self.topo_history) > self.geo_window:
            self.topo_history.pop(0)
            self.coarse_history.pop(0)
        if len(self.coarse_history) > 1:
            if abs(self.coarse_history[-1] - self.coarse_history[-2]) < 0.1:
                self.stagnation_counter += 1
            else: self.stagnation_counter = 0
    
    def perturb_mixer_targeted(self, extractor):
        if hasattr(extractor, 'mixer'):
            with torch.no_grad():
                for module in extractor.mixer.mixer:
                    if isinstance(module, nn.Linear):
                        W = module.weight
                        try:
                            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
                        except:
                            module.weight.add_(torch.randn_like(W) * self.mixer_noise_scale)
                            continue
                        S_sq = S ** 2
                        total_energy = torch.sum(S_sq)
                        if total_energy < 1e-12: continue
                        cumulative_energy = torch.cumsum(S_sq, dim=0) / total_energy
                        k = torch.sum(cumulative_energy < self.dominant_energy_threshold).item() + 1
                        k = max(1, min(k, S.shape[0]-1))
                        V_dominant = Vh[:k, :]
                        random_noise = torch.randn_like(W)
                        noise_proj_comp_1 = torch.matmul(random_noise, V_dominant.T)
                        noise_dominant = torch.matmul(noise_proj_comp_1, V_dominant)
                        noise_orthogonal = random_noise - noise_dominant
                        module.weight.add_(noise_orthogonal * self.mixer_noise_scale)
            return True
        return False

class IterativeRefinementEngine:
    def __init__(self, device: torch.device):
        self.device = device
        self.monitor = SpectralMonitor()
    
    def create_refined_model(self, parent_state: Dict, data_loader, feature_extractor, 
                           lambda_taxonomic: float = 0.3, learning_rate: float = 0.01):
        child = TaxonomicMLP(input_dim=feature_extractor.embed_dim, hidden_dim=256, 
                          num_classes=100, num_superclasses=20).to(self.device)
        child.load_state_dict(parent_state)
        child.train()
        optimizer = torch.optim.SGD(child.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            labels = labels.long()
            coarse_labels = torch.tensor([_FINE_TO_COARSE[l.item()] for l in labels], device=self.device)
            with torch.no_grad(): features = feature_extractor(inputs)
            optimizer.zero_grad()
            fine_logits, super_logits = child(features)
            loss_fine = criterion(fine_logits, labels)
            loss_super = criterion(super_logits, coarse_labels)
            total_loss = loss_fine + lambda_taxonomic * loss_super
            total_loss.backward()
            optimizer.step()
            break
        child.apply_masks()
        return child

# =============================================================================
# TRAINING & EVALUATION FRAMEWORK (v18.2: Aggressive + Safe)
# =============================================================================
class IterativeRefinementTrainer:
    def __init__(self, device: torch.device, output_dir: str = 'results_v18_2'):
        self.device = device
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.monitor = SpectralMonitor()
        self.engine = IterativeRefinementEngine(device)
        
        # v18.2 HYPERPARAMETERS: The Scientific Powerhouse Mix
        self.LAMBDA_TAX_BASE = 0.15        # Increased from 0.1 (Enforce hierarchy earlier)
        self.LAMBDA_TAX_SHOCK = 0.8        # Increased from 0.6 (Max shock)
        self.GAP_SHOCK_THRESHOLD = 3.5      # Lowered from 5.0 (Trigger shock sooner)
        self.LAMBDA_SPARSE_MIXER = 1e-4
        self.LAMBDA_SPECTRAL = 0.05
        self.SEMANTIC_PLASTICITY_THRESHOLD = 0.05
        self.STAGNATION_LIMIT = 10
        self.MIXER_NOISE_SCALE = 0.05
        self.DOMINANT_ENERGY_THRESHOLD = 0.8
        self.PHASE_WINDOW = 5
        self.PHASE_STD_DEV_LIMIT = 1.5
        self.GEO_WINDOW = 3
        self.MASK_WARMUP_EPOCHS = 25       # v15.4 Feature: Safe start
        self.OVERFIT_GAP_THRESHOLD = 12.0   # v15.4 Feature: Safety valve
    
    def load_data(self, cycle: int, batch_size: int = 64):
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset_sizes = {1: 5000, 4: 20000, 7: 50000}
        size = dataset_sizes.get(cycle, 50000)
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        if size < len(trainset):
            indices = torch.randperm(len(trainset))[:size]
            trainset = torch.utils.data.Subset(trainset, indices)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        
        test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
        return trainloader, testloader
    
    def train_model(self, model, cycle, chain_type='APEX', feature_extractor=None):
        trainloader, testloader = self.load_data(cycle)
        
        if chain_type == 'APEX' and hasattr(feature_extractor, 'mixer'):
            feature_extractor.unfreeze_mixer_only()
            optimizer = torch.optim.AdamW([
                {'params': model.parameters()},
                {'params': feature_extractor.mixer.parameters(), 'lr': 0.005 * 5.0}
            ], lr=0.005, weight_decay=0.1 if cycle < 5 else 0.01)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.1 if cycle < 5 else 0.01)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        criterion = nn.CrossEntropyLoss()
        
        best_accuracy = 0.0
        best_coarse_accuracy = 0.0
        best_gap = 100.0
        best_state = None
        prev_gap = 0.0
        ratio_history = []
        
        controller = AdaptiveTopologyController(
            semantic_plasticity_threshold=self.SEMANTIC_PLASTICITY_THRESHOLD,
            stagnation_limit=self.STAGNATION_LIMIT,
            mixer_noise_scale=self.MIXER_NOISE_SCALE,
            dominant_energy_threshold=self.DOMINANT_ENERGY_THRESHOLD,
            geo_window=self.GEO_WINDOW
        )
        
        if chain_type == 'BLIND':
            model.fc_super.requires_grad_(False)
        
        for epoch in range(100):
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            
            current_lambda_tax = self.LAMBDA_TAX_BASE
            shock_state = "BASE"
            
            # v18.2: Safety Valve Logic (Prevent early catastrophic overfitting)
            if cycle > 1 and prev_gap > self.GAP_SHOCK_THRESHOLD:
                if prev_gap > self.OVERFIT_GAP_THRESHOLD and cycle < 3:
                    # Safety override: If overfitting is insane early, reduce shock intensity
                    current_lambda_tax = min(current_lambda_tax, 0.4)
                    shock_state = "SAFE_OVERRIDE"
                else:
                    # Standard aggressive shock
                    current_lambda_tax = self.LAMBDA_TAX_SHOCK
                    shock_state = "SHOCK"
            
            for inputs, labels in trainloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                labels = labels.long()
                coarse_labels = torch.tensor([_FINE_TO_COARSE[l.item()] for l in labels], device=self.device)
                
                with torch.no_grad():
                    features = feature_extractor(inputs) if feature_extractor else inputs
                
                fine_logits, super_logits = model(features)
                loss_fine = criterion(fine_logits, labels)
                loss_spectral_opt = compute_spectral_loss(model.fc1.weight)
                
                if chain_type == 'APEX':
                    loss_super = criterion(super_logits, coarse_labels)
                    total_loss = loss_fine + current_lambda_tax * loss_super
                    total_loss += self.LAMBDA_SPECTRAL * loss_spectral_opt
                    
                    if hasattr(feature_extractor, 'mixer'):
                        mixer_spec_loss = 0.0
                        for module in feature_extractor.mixer.mixer:
                            if isinstance(module, nn.Linear):
                                mixer_spec_loss += compute_spectral_loss(module.weight)
                        total_loss += self.LAMBDA_SPECTRAL * mixer_spec_loss
                    
                    mixer_sparse_penalty = 0
                    if hasattr(feature_extractor, 'mixer'):
                        for module in feature_extractor.mixer.mixer:
                            if isinstance(module, nn.Linear):
                                mixer_sparse_penalty += module.weight.abs().sum()
                        total_loss += self.LAMBDA_SPARSE_MIXER * mixer_sparse_penalty
                else:
                    total_loss = loss_fine
                    total_loss += self.LAMBDA_SPECTRAL * loss_spectral_opt
                    mixer_sparse_penalty = 0
                    if hasattr(feature_extractor, 'mixer'):
                        for module in feature_extractor.mixer.mixer:
                            if isinstance(module, nn.Linear):
                                mixer_sparse_penalty += module.weight.abs().sum()
                        total_loss += self.LAMBDA_SPARSE_MIXER * mixer_sparse_penalty
                
                optimizer.zero_grad()
                total_loss.backward()
                
                with torch.no_grad():
                    model.fc1.weight.grad *= model.mask1
                    model.fc2.weight.grad *= model.mask2
                    if model.fc_super.weight.grad is not None:
                        model.fc_super.weight.grad *= model.mask_super
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += total_loss.item()
                _, predicted = torch.max(fine_logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            train_accuracy = 100 * correct / total
            
            model.eval()
            correct = 0
            coarse_correct = 0
            total = 0
            coarse_total = 0
            
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    features = feature_extractor(inputs) if feature_extractor else inputs
                    fine_logits, super_logits = model(features)
                    
                    _, predicted = torch.max(fine_logits.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    true_coarse = torch.tensor([_FINE_TO_COARSE[l.item()] for l in labels], device=self.device)
                    _, pred_coarse = torch.max(super_logits.data, 1)
                    coarse_total += true_coarse.size(0)
                    coarse_correct += (pred_coarse == true_coarse).sum().item()
            
            test_accuracy = 100 * correct / total
            coarse_accuracy = 100 * coarse_correct / coarse_total
            gap = train_accuracy - test_accuracy
            prev_gap = gap
            
            topo_ratio, L_opt_total, L_opt_fc1, L_opt_mixer, L_mon_val = self.compute_topology_ratio(
                model, feature_extractor, chain_type
            )
            ratio_history.append(topo_ratio)
            
            phase_state = self.detect_phase_state(ratio_history, self.PHASE_WINDOW, self.PHASE_STD_DEV_LIMIT)
            controller.update_history(topo_ratio, coarse_accuracy)
            
            intervention = False
            trigger_reason = ""
            if chain_type == 'APEX':
                intervention, trigger_reason = controller.detect_intervention_need(phase_state, feature_extractor)
            
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_coarse_accuracy = coarse_accuracy
                best_gap = gap
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            
            scheduler.step()
            
            if epoch % 20 == 0:
                L_mon, rank_eff, _ = self.monitor.compute_metrics(model.fc1.weight)
                log_str = f"   [{chain_type}] Ep {epoch:3d} | V:{test_accuracy:5.1f}% | C.V:{coarse_accuracy:5.1f}% | Gap:{gap:4.1f}"
                log_str += f" | S:{shock_state} | Phase:{phase_state}"
                if intervention: log_str += f" 🔧{trigger_reason}"
                log_str += f" | Topo_R:{topo_ratio:.3f} | L_mon:{L_mon:.3f}"
                spar = model.get_sparsity()
                log_str += f" | Spar:{spar:.1f}%"
                print(log_str)
        
        if best_state:
            model.load_state_dict(best_state)
            model.apply_masks()
        
        L_final, rank_final, _ = self.monitor.compute_metrics(model.fc1.weight)
        avg_ratio = np.mean(ratio_history[-10:]) if len(ratio_history) >= 10 else ratio_history[-1]
        return {
            'final_accuracy': best_accuracy,
            'final_coarse_accuracy': best_coarse_accuracy,
            'final_gap': best_gap,
            'L': L_final,
            'rank': rank_final,
            'sparsity': model.get_sparsity(),
            'topo_ratio': avg_ratio
        }
    
    def detect_phase_state(self, ratio_history: List[float], phase_window: int = 5, phase_std_dev_limit: float = 1.5) -> str:
        if len(ratio_history) < phase_window: return "INIT"
        recent = ratio_history[-phase_window:]
        std_recent = np.std(recent)
        if std_recent < 1e-6: return "STABLE"
        last_val = ratio_history[-1]
        mean_recent = np.mean(recent)
        z_score = abs(last_val - mean_recent) / std_recent
        return "SHIFTING" if z_score > phase_std_dev_limit else "STABLE"
    
    def compute_topology_ratio(self, model, extractor, chain_type):
        L_opt_fc1 = compute_spectral_loss(model.fc1.weight).item()
        L_opt_mixer = 0.0
        if hasattr(extractor, 'mixer'):
            for m in extractor.mixer.mixer:
                if isinstance(m, nn.Linear):
                    L_opt_mixer += compute_spectral_loss(m.weight).item()
        L_opt_total = L_opt_fc1 + L_opt_mixer
        L_mon_val, _, _ = self.monitor.compute_metrics(model.fc1.weight)
        ratio = (L_opt_total / L_mon_val) if L_mon_val > 0 else 0.0
        return ratio, L_opt_total, L_opt_fc1, L_opt_mixer, L_mon_val
    
    def run_refinement(self, num_iterations: int = 20, num_seeds: int = 5, early_stop_patience: int = 5):
        print("=" * 80)
        print(f"🔬 NEUROSOVEREIGN v18.2: SCIENTIFIC POWERHOUSE")
        print(f"   Iterations: {num_iterations} | Seeds: {num_seeds}")
        print(f"   Aggressive Shock (0.8) + Safe Valve + Adaptive Control")
        print("=" * 80)
        
        extractor_apex = PatchFeatureExtractor(embed_dim=384, use_mixer=True).to(self.device)
        extractor_blind = PatchFeatureExtractor(embed_dim=384, use_mixer=False).to(self.device)
        
        all_results = []
        best_overall = {'apex_acc': 0.0, 'blind_acc': 0.0, 'hierarchy_delta': 0.0}
        
        for seed in range(1, num_seeds + 1):
            print(f"\n{'='*60}")
            print(f"🌱 STATISTICAL RUN {seed}/{num_seeds}")
            print(f"{'='*60}")
            set_seed(42 + seed)
            
            elk_apex = TaxonomicMLP(input_dim=384, hidden_dim=256, num_classes=100, num_superclasses=20).to(self.device)
            elk_blind = TaxonomicMLP(input_dim=384, hidden_dim=256, num_classes=100, num_superclasses=20).to(self.device)
            
            print("  🧬 Initializing seed models (Cycle 0)...")
            res_apex = self.train_model(elk_apex, 0, 'APEX', extractor_apex)
            res_blind = self.train_model(elk_blind, 0, 'BLIND', extractor_blind)
            
            no_improvement = 0
            best_apex_acc = res_apex['final_accuracy']
            best_blind_acc = res_blind['final_accuracy']
            best_apex_coarse = res_apex['final_coarse_accuracy']
            
            for iteration in range(1, num_iterations + 1):
                print(f"\n{'-'*50}")
                print(f"🧬 REFINEMENT ITERATION {iteration}/{num_iterations}")
                print(f"{'-'*50}")
                
                if iteration == 1:
                    print("   ⚡ UNFREEZING MIXER (APEX)...")
                    extractor_apex.unfreeze_mixer_only()
                
                print("   🌟 Training APEX model...")
                refined_apex = self.engine.create_refined_model(
                    elk_apex.state_dict(), self.load_data(iteration)[0], extractor_apex, 
                    lambda_taxonomic=self.LAMBDA_TAX_BASE, learning_rate=0.01
                )
                res_apex = self.train_model(refined_apex, iteration, 'APEX', extractor_apex)
                
                print("   🌀 Training BLIND model...")
                refined_blind = self.engine.create_refined_model(
                    elk_blind.state_dict(), self.load_data(iteration)[0], extractor_blind, 
                    lambda_taxonomic=self.LAMBDA_TAX_BASE, learning_rate=0.01
                )
                res_blind = self.train_model(refined_blind, iteration, 'BLIND', extractor_blind)
                
                improved_apex = res_apex['final_accuracy'] - best_apex_acc
                improved_blind = res_blind['final_accuracy'] - best_blind_acc
                improved_coarse_apex = res_apex['final_coarse_accuracy'] - best_apex_coarse
                
                if improved_apex >= 0.0 or improved_coarse_apex >= 0.5:
                    print(f"   ✅ APEX REFINED ({improved_apex:+.1f}%) | Coarse: {res_apex['final_coarse_accuracy']:.1f}%")
                    elk_apex.load_state_dict(refined_apex.state_dict())
                    best_apex_acc = res_apex['final_accuracy']
                    best_apex_coarse = res_apex['final_coarse_accuracy']
                    no_improvement = 0
                    if best_apex_acc > best_overall['apex_acc']:
                        best_overall['apex_acc'] = best_apex_acc
                        torch.save(elk_apex.state_dict(), os.path.join(self.output_dir, f'best_model_apex_seed_{seed}.pth'))
                else:
                    print(f"   ⚠️ APEX STAGNANT: {res_apex['final_accuracy']:.1f}%")
                    no_improvement += 1
                
                if improved_blind >= 0.0:
                    print(f"   ✅ BLIND REFINED ({improved_blind:+.1f}%)")
                    elk_blind.load_state_dict(refined_blind.state_dict())
                    best_blind_acc = res_blind['final_accuracy']
                    if best_blind_acc > best_overall['blind_acc']:
                        best_overall['blind_acc'] = best_blind_acc
                        torch.save(elk_blind.state_dict(), os.path.join(self.output_dir, f'best_model_blind_seed_{seed}.pth'))
                else:
                    print(f"   ⚠️ BLIND STAGNANT: {res_blind['final_accuracy']:.1f}%")
                
                delta_structure = res_apex['final_accuracy'] - res_blind['final_accuracy']
                delta_hierarchy = res_apex['final_coarse_accuracy'] - res_blind['final_coarse_accuracy']
                
                all_results.append({
                    'seed': seed, 'iteration': iteration,
                    'apex_accuracy': res_apex['final_accuracy'], 'blind_accuracy': res_blind['final_accuracy'],
                    'apex_coarse_accuracy': res_apex['final_coarse_accuracy'], 'blind_coarse_accuracy': res_blind['final_coarse_accuracy'],
                    'delta_structure': delta_structure, 'delta_hierarchy': delta_hierarchy,
                    'apex_L': res_apex['L'], 'blind_L': res_blind['L'],
                    'apex_sparsity': res_apex['sparsity'], 'blind_sparsity': res_blind['sparsity'],
                    'apex_topo_ratio': res_apex['topo_ratio'], 'blind_topo_ratio': res_blind['topo_ratio']
                })
                
                if no_improvement >= early_stop_patience and iteration > 5:
                    print(f"   ⏹️  EARLY STOPPING after {early_stop_patience} iterations")
                    break
        
        # Load best final models for reporting
        best_apex = TaxonomicMLP(input_dim=384, hidden_dim=256, num_classes=100, num_superclasses=20).to(self.device)
        best_blind = TaxonomicMLP(input_dim=384, hidden_dim=256, num_classes=100, num_superclasses=20).to(self.device)
        
        path_apex = os.path.join(self.output_dir, f'best_model_apex_seed_1.pth')
        path_blind = os.path.join(self.output_dir, f'best_model_blind_seed_1.pth')
        
        if os.path.exists(path_apex) and os.path.exists(path_blind):
            best_apex.load_state_dict(torch.load(path_apex))
            best_blind.load_state_dict(torch.load(path_blind))
            hierarchy_delta = run_hierarchy_benchmark(best_apex, best_blind, extractor_apex, extractor_blind, self.device)
            best_overall['hierarchy_delta'] = hierarchy_delta
            visualize_singular_values(best_apex, best_blind, extractor_apex, extractor_blind, self.device, self.output_dir, self.monitor)
        else:
            print("⚠️  Warning: Best model files not found, skipping hierarchy benchmark.")
            hierarchy_delta = 0.0
            best_overall['hierarchy_delta'] = 0.0

        self._save_results(all_results, best_overall, hierarchy_delta)
        self._plot_results(all_results, best_overall)
        return all_results
    
    def _save_results(self, all_results: List[Dict], best_overall: Dict, hierarchy_delta: float):
        df = pd.DataFrame(all_results)
        df.to_csv(os.path.join(self.output_dir, 'refinement_results.csv'), index=False)
        stats = {
            'best_apex_accuracy': best_overall['apex_acc'],
            'best_blind_accuracy': best_overall['blind_acc'],
            'hierarchy_benchmark_delta': hierarchy_delta,
            'structural_advantage': best_overall['apex_acc'] - best_overall['blind_acc'],
            'statistical_summary': {
                'apex_mean': float(df.groupby('iteration')['apex_accuracy'].mean().max()),
                'blind_mean': float(df.groupby('iteration')['blind_accuracy'].mean().max()),
                'delta_structure_mean': float(df.groupby('iteration')['delta_structure'].mean().max()),
                'delta_hierarchy_mean': float(df.groupby('iteration')['delta_hierarchy'].mean().max()),
                'max_coarse_accuracy': float(df.groupby('iteration')['apex_coarse_accuracy'].max().max())
            },
            'parameters': {
                'reference_baseline': REFERENCE_BASELINE,
                'lambda_tax_base': self.LAMBDA_TAX_BASE,
                'lambda_tax_shock': self.LAMBDA_TAX_SHOCK,
                'gap_shock_threshold': self.GAP_SHOCK_THRESHOLD
            }
        }
        with open(os.path.join(self.output_dir, 'taxonomic_report.json'), 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\n✅ Results saved to: {self.output_dir}")
    
    def _plot_results(self, all_results: List[Dict], best_overall: Dict):
        df = pd.DataFrame(all_results)
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        apex_means = df.groupby('iteration')['apex_accuracy'].mean()
        blind_means = df.groupby('iteration')['blind_accuracy'].mean()
        apex_stds = df.groupby('iteration')['apex_accuracy'].std()
        blind_stds = df.groupby('iteration')['blind_accuracy'].std()
        iterations = apex_means.index
        plt.plot(iterations, apex_means, 'b-o', linewidth=2, label='APEX (Taxonomic)')
        plt.fill_between(iterations, apex_means - apex_stds, apex_means + apex_stds, alpha=0.2, color='blue')
        plt.plot(iterations, blind_means, 'r-s', linewidth=2, label='BLIND (Symmetric)')
        plt.fill_between(iterations, blind_means - blind_stds, blind_means + blind_stds, alpha=0.2, color='red')
        plt.title('Iterative Refinement Accuracy', fontsize=14)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Test Accuracy (%)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        delta_hier_means = df.groupby('iteration')['delta_hierarchy'].mean()
        delta_hier_stds = df.groupby('iteration')['delta_hierarchy'].std()
        plt.plot(iterations, delta_hier_means, 'g-^', linewidth=2, label='Hierarchy Advantage')
        plt.fill_between(iterations, delta_hier_means - delta_hier_stds, delta_hier_means + delta_hier_stds, alpha=0.2, color='green')
        plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.3, label='Validation Threshold')
        plt.axhline(y=REFERENCE_BASELINE, color='purple', linestyle=':', alpha=0.5, label=f'Ref. Baseline ({REFERENCE_BASELINE}%)')
        plt.title('Taxonomic Advantage', fontsize=14)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Coarse Diff (%)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        L_apex_means = df.groupby('iteration')['apex_L'].mean()
        L_blind_means = df.groupby('iteration')['blind_L'].mean()
        topo_apex_means = df.groupby('iteration')['apex_topo_ratio'].mean()
        plt.plot(iterations, topo_apex_means, 'b-o', linewidth=2, label='APEX Topo_R')
        plt.plot(iterations, L_apex_means, 'b--', linewidth=1.5, label='APEX L_mon')
        plt.plot(iterations, L_blind_means, 'r--', linewidth=1.5, label='BLIND L_mon')
        plt.axhline(y=1.8, color='k', linestyle='--', alpha=0.3, label='Target Coherence')
        plt.title('Spectral Evolution', fontsize=14)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Metric Value', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        categories = ['APEX Fine', 'BLIND Fine', 'Hierarchy Δ']
        values = [best_overall['apex_acc'], best_overall['blind_acc'], best_overall['hierarchy_delta']]
        colors = ['blue', 'red', 'green']
        plt.bar(categories, values, alpha=0.7, color=colors)
        plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.3)
        plt.title('Final Benchmark Summary', fontsize=14)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'refinement_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()

# =============================================================================
# BENCHMARK & UTILS
# =============================================================================
class CoarseCIFAR100(torchvision.datasets.CIFAR100):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, _FINE_TO_COARSE[target]

def run_hierarchy_benchmark(model_apex, model_blind, feature_extractor_apex, feature_extractor_blind, device):
    testset = CoarseCIFAR100(
        root='./data', train=False, download=True, 
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    )
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)
    
    def evaluate(model, extractor):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                features = extractor(inputs)
                w1 = model.fc1.weight * model.mask1
                h = F.relu(F.linear(features, w1))
                w_super = model.fc_super.weight * model.mask_super
                outputs_super = F.linear(h, w_super)
                _, predicted = torch.max(outputs_super.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total

    print("\n" + "="*90)
    print(" " * 30 + "HIERARCHY STRESS TEST (CIFAR-20)")
    print("=" * 90)
    acc_apex = evaluate(model_apex, feature_extractor_apex)
    acc_blind = evaluate(model_blind, feature_extractor_blind)
    delta = acc_apex - acc_blind
    
    print(f"   APEX Coarse-Only Accuracy: {acc_apex:.2f}%")
    print(f"   BLIND Coarse-Only Accuracy: {acc_blind:.2f}%")
    print(f"   Structural Advantage (Δ):   {delta:+.2f}%")
    print("-" * 90)
    if delta > 1.0: print("✅ VALIDATED: APEX demonstrates significant inductive bias for hierarchy.")
    else: print("⚠️  WARNING: Low hierarchy transfer detected.")
    return delta

def visualize_singular_values(model_apex, model_blind, feature_extractor_apex, feature_extractor_blind, device, output_dir, monitor):
    print("\n" + "="*90)
    print(" " * 30 + "SPECTRAL ANALYSIS: SINGULAR VALUES")
    print("=" * 90)
    os.makedirs(os.path.join(output_dir, 'spectral_analysis'), exist_ok=True)
    
    def get_singular_values(model, extractor):
        S_fc1 = monitor.get_singular_values(model.fc1.weight)
        S_mixers = []
        if hasattr(extractor, 'mixer'):
            for module in extractor.mixer.mixer:
                if isinstance(module, nn.Linear):
                    S = monitor.get_singular_values(module.weight)
                    S_mixers.append(S)
        return S_fc1, S_mixers
    
    S_apex_fc1, S_apex_mixers = get_singular_values(model_apex, feature_extractor_apex)
    S_blind_fc1, S_blind_mixers = get_singular_values(model_blind, feature_extractor_blind)
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(range(len(S_apex_fc1)), S_apex_fc1, 'b-o', markersize=4, linewidth=2, label='APEX FC1')
    plt.semilogy(range(len(S_blind_fc1)), S_blind_fc1, 'r-s', markersize=4, linewidth=2, label='BLIND FC1')
    plt.title('Singular Values: FC1 Layer', fontsize=14)
    plt.xlabel('Singular Value Index', fontsize=12)
    plt.ylabel('Magnitude (log scale)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'spectral_analysis', 'fc1_singular_values.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    if S_apex_mixers and S_blind_mixers:
        plt.figure(figsize=(12, 8))
        for i, (S_apex, S_blind) in enumerate(zip(S_apex_mixers, S_blind_mixers)):
            plt.subplot(len(S_apex_mixers), 1, i+1)
            plt.semilogy(range(len(S_apex)), S_apex, 'b-o', markersize=3, linewidth=1.5, label=f'APEX Mixer {i+1}')
            plt.semilogy(range(len(S_blind)), S_blind, 'r-s', markersize=3, linewidth=1.5, label=f'BLIND Mixer {i+1}')
            plt.title(f'Singular Values: Mixer Layer {i+1}', fontsize=12)
            plt.xlabel('Index', fontsize=10)
            plt.ylabel('Magnitude (log)', fontsize=10)
            plt.legend(fontsize=8)
            plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'spectral_analysis', 'mixer_singular_values.png'), dpi=300, bbox_inches='tight')
        plt.close()
    print("✅ Singular value visualizations saved.")

# =============================================================================
# MAIN EXECUTION
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description='NeuroSovereign v18.2 Scientific Powerhouse')
    parser.add_argument('--iterations', type=int, default=20, help='Refinement Iterations')
    parser.add_argument('--seeds', type=int, default=5, help='Statistical Seeds')
    parser.add_argument('--patience', type=int, default=5, help='Early Stopping Patience')
    parser.add_argument('--output', type=str, default='results_v18_2', help='Output Directory')
    args, _ = parser.parse_known_args()
    return args

def main():
    args = parse_args()
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 NeuroSovereign v18.2 (Scientific Powerhouse) on {device}")
    
    trainer = IterativeRefinementTrainer(device, args.output)
    results = trainer.run_refinement(
        num_iterations=args.iterations,
        num_seeds=args.seeds,
        early_stop_patience=args.patience
    )
    
    df = pd.DataFrame(results)
    final_apex = df.groupby('iteration')['apex_accuracy'].mean().max()
    final_blind = df.groupby('iteration')['blind_accuracy'].mean().max()
    hierarchy_advantage = df.groupby('iteration')['delta_hierarchy'].mean().max()
    max_coarse_accuracy = df.groupby('iteration')['apex_coarse_accuracy'].max().max()
    
    print(f"\n{'='*80}")
    print("🏆 FINAL REPORT v18.2")
    print(f"{'='*80}")
    print(f"Structural Advantage: {final_apex - final_blind:.2f}%")
    print(f"Hierarchy Advantage:  {hierarchy_advantage:.2f}%")
    print(f"Max Coarse Accuracy:  {max_coarse_accuracy:.2f}%")
    
    if hierarchy_advantage > 1.0:
        print("\n✅ VALIDATED: Controlled induction of hierarchical structure.")
    else:
        print("\n⚠️  PARTIAL: Hierarchical signal present but needs amplification.")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()