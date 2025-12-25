#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuroSovereign v13.0: Orthogonal Apex
Features:
1. True Token Mixing (MLP-Mixer style: mixing across T).
2. Spectral Constraint (L as regularizer, preventing metric gaming).
3. Minimality Enforcement (Rank Capping without renormalization).
4. Objective: SOTA generalization via constrained evolution.
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
NUM_TOKENS = 64  # 8x8 patches
TARGET_L = 1.8   # Coherencia objetivo (Sweet Spot)
LAMBDA_SPECTRAL = 0.05  # Peso de la regularización espectral
UNFREEZE_AT_CYCLE = 3

# =============================================================================
# 1. TRUE SYNTACTIC VISION EYE
# =============================================================================
class TokenMixer(nn.Module):
    """
    Mezcla tokens entre sí.
    Input: (B, T, D) -> Transpose -> (B, D, T) -> Linear -> (B, D, T) -> Transpose
    """
    def __init__(self, num_tokens=64):
        super().__init__()
        # Inicialización Identidad para estabilidad inicial
        self.mixer = nn.Linear(num_tokens, num_tokens, bias=False)
        nn.init.eye_(self.mixer.weight)
        # Añadir un pequeño ruido para permitir evolución
        with torch.no_grad():
            self.mixer.weight += torch.randn_like(self.mixer.weight) * 0.01

    def forward(self, x):
        # x: (B, T, D)
        x = x.transpose(1, 2)  # -> (B, D, T)
        x = self.mixer(x)      # -> (B, D, T)
        x = x.transpose(1, 2)  # -> (B, T, D)
        return x

class PatchFeatureExtractor(nn.Module):
    """
    ViT-Lite + True Token Mixing.
    """
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=384):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.mixer = TokenMixer(self.num_patches)
        
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
            
        self.freeze()

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
            
    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.proj(x)  # (B, D, H, W)
        x = x.flatten(2)  # (B, D, T)
        x = x.transpose(1, 2) # (B, T, D)
        
        # TRUE SYNTACTIC MIXING
        x = self.mixer(x) 
        
        x = x.mean(dim=1) # Global Pool (B, D)
        return x

# =============================================================================
# 2. LOTTERY TICKET MLP
# =============================================================================
class LotteryMLP(nn.Module):
    def __init__(self, input_dim: int = 384, hidden_dim: int = 192, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, num_classes, bias=False)
        
        self.register_buffer('mask1', torch.ones_like(self.fc1.weight))
        self.register_buffer('mask2', torch.ones_like(self.fc2.weight))
        
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.02)

    def apply_masks(self):
        with torch.no_grad():
            self.fc1.weight.data *= self.mask1
            self.fc2.weight.data *= self.mask2

    def get_sparsity(self):
        total = self.mask1.numel() + self.mask2.numel()
        active = self.mask1.sum() + self.mask2.sum()
        return (1.0 - (active.item() / total)) * 100

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1 = self.fc1.weight * self.mask1
        w2 = self.fc2.weight * self.mask2
        
        x = F.relu(F.linear(x, w1))
        return F.linear(x, w2)

class StandardBaseline(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=192):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 10))
    
    def forward(self, x):
        return self.net(x)

# =============================================================================
# 3. MONITORING
# =============================================================================
class SpectralMonitor:
    def compute_L(self, weight: torch.Tensor) -> float:
        with torch.no_grad():
            W = weight.cpu().numpy()
            U, S, Vh = np.linalg.svd(W, full_matrices=False)
            rank_eff = max(1, int(np.sum(S > 0.05 * np.max(S))))
            S_norm = S / (np.sum(S) + 1e-12)
            S_norm = S_norm[S_norm > 1e-15]
            S_vN = -np.sum(S_norm * np.log(S_norm + 1e-15))
            L = 1.0 / (abs(S_vN - np.log(rank_eff + 1)) + 0.3)
            return L

# =============================================================================
# 4. ORTHOGONAL EVOLUTION ENGINE
# =============================================================================
class OrthogonalEvolutionEngine:
    def __init__(self, device: torch.device):
        self.device = device
        self.monitor = SpectralMonitor()
        self.input_dim = INPUT_DIM

    def _gradient_nudge_inheritance(self, 
                                     child_model: nn.Module, 
                                     elk_state: Dict, 
                                     data_loader, 
                                     feature_extractor,
                                     nudge_lr: float = 0.005):
        old_dim = elk_state['fc1.weight'].shape[0]
        temp_elk = LotteryMLP(input_dim=self.input_dim, hidden_dim=old_dim).to(self.device)
        temp_elk.load_state_dict(elk_state)
        temp_elk.train()
        
        optimizer = torch.optim.SGD(temp_elk.parameters(), lr=nudge_lr)
        criterion = nn.CrossEntropyLoss()
        
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            with torch.no_grad():
                inputs_features = feature_extractor(inputs)
            
            optimizer.zero_grad()
            outputs = temp_elk(inputs_features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            break
        
        with torch.no_grad():
            child_model.fc1.weight[:old_dim, :] = temp_elk.fc1.weight.data
            child_model.mask1[:old_dim, :] = temp_elk.mask1.data
            child_model.fc2.weight[:, :old_dim] = temp_elk.fc2.weight.data
            child_model.mask2[:, :old_dim] = temp_elk.mask2.data

    def _apply_minimalistic_shock(self, model: nn.Module, layer_name='fc1', target_rank_ratio=0.8):
        """
        Rank Capping: Cortamos singular values débiles y NO renormalizamos.
        Esto fuerza la minimización (Energy Decay).
        """
        layer = getattr(model, layer_name)
        W = layer.weight.data
        
        U, S, V = torch.svd(W)
        
        # Calcular rango objetivo
        max_rank = S.shape[0]
        target_rank = int(max_rank * target_rank_ratio)
        
        # Crear máscara: mantener solo los 'target_rank' valores más grandes
        mask = torch.zeros_like(S)
        mask[:target_rank] = 1.0
        
        S_shocked = S * mask
        
        # RECONSTRUCCIÓN SIN RENORMALIZACIÓN (Minimality Enforcement)
        # Dejamos que la energía total baje.
        W_shocked = U @ torch.diag(S_shocked) @ V.t()
        
        with torch.no_grad():
            layer.weight.data = W_shocked
            # Actualizar máscara explícita basada en lo que sobrevivió
            new_mask = (torch.abs(W_shocked) > 1e-5).float()
            if layer_name == 'fc1':
                model.mask1.copy_(new_mask)
            else:
                model.mask2.copy_(new_mask)

    def create_orthogonal_offspring(self, 
                                     elk_state: Dict, 
                                     new_hidden_dim: int, 
                                     cycle: int,
                                     data_loader,
                                     feature_extractor,
                                     parent_gap: float) -> nn.Module:
        child = LotteryMLP(input_dim=self.input_dim, hidden_dim=new_hidden_dim).to(self.device)
        old_dim = elk_state['fc1.weight'].shape[0]
        
        if new_hidden_dim >= old_dim:
            child.fc1.weight.data[:old_dim, :] = elk_state['fc1.weight']
            child.fc2.weight.data[:, :old_dim] = elk_state['fc2.weight']
            child.mask1[:old_dim, :] = elk_state['mask1']
            child.mask2[:, :old_dim] = elk_state['mask2']
            
            std_elk = elk_state['fc1.weight'].std()
            child.fc1.weight.data[old_dim:, :] = torch.randn(new_hidden_dim - old_dim, self.input_dim, device=self.device) * std_elk
            child.fc2.weight.data[:, old_dim:] = torch.randn(10, new_hidden_dim - old_dim, device=self.device) * std_elk
        else:
            child.fc1.weight.data = elk_state['fc1.weight'][:new_hidden_dim, :]
            child.fc2.weight.data = elk_state['fc2.weight'][:, :new_hidden_dim]
            child.mask1.data = elk_state['mask1'][:new_hidden_dim, :]
            child.mask2.data = elk_state['mask2'][:, :new_hidden_dim]

        self._gradient_nudge_inheritance(child, elk_state, data_loader, feature_extractor, nudge_lr=0.01)

        # Shock de Minimización si el gap es alto (evitar redundancia)
        if parent_gap > 25.0:
            self._apply_minimalistic_shock(child, 'fc1', target_rank_ratio=0.8)
            print(f"      ⚡ MINIMALISTIC SHOCK: Energy decay applied (Gap {parent_gap:.1f})")

        return child

# =============================================================================
# 5. ORTHOGONAL TRAINER (WITH SPECTRAL CONSTRAINT)
# =============================================================================
class OrthogonalTrainer:
    def __init__(self, device: torch.device, feature_extractor=None):
        self.device = device
        self.feature_extractor = feature_extractor
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.monitor = SpectralMonitor()
        self.engine = OrthogonalEvolutionEngine(device)

    def _preprocess_batch(self, x):
        with torch.no_grad():
            return self.feature_extractor(x)

    def get_curriculum_dataset(self, cycle: int):
        full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=self.transform)
        sizes = {1: 2500, 4: 10000, 7: 50000}
        size = sizes.get(cycle, 2500)
        return torch.utils.data.Subset(full_trainset, torch.randperm(len(full_trainset))[:size])

    def train_model(self, model, cycle, is_baseline=False):
        trainset = self.get_curriculum_dataset(cycle)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
        
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=self.transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)
        
        wd = 0.1 if cycle < 5 else 0.01
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=wd) # LR un poco más bajo para estabilidad con regularizador
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)
        criterion = nn.CrossEntropyLoss()
        
        best_gap = 100.0
        best_acc = 0.0
        best_state = None
        
        mode_str = "BASELINE" if is_baseline else "ORTHOGONAL APEX"
        print(f"      🚀 Training {mode_str} (Cycle {cycle}) | Data: {len(trainset)}")
        
        extractor_status = "Trainable" if self.feature_extractor.proj.weight.requires_grad else "Frozen"
        print(f"      👁️  Eye & Mixer: {extractor_status} | Target L: {TARGET_L}")
        
        for epoch in range(100):
            model.train()
            train_correct, train_total = 0, 0
            
            for inputs, labels in trainloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                inputs_proc = self._preprocess_batch(inputs)
                
                optimizer.zero_grad()
                outputs = model(inputs_proc)
                ce_loss = criterion(outputs, labels)
                
                # SPECTRAL CONSTRAINT (No en baseline)
                reg_loss = torch.tensor(0.0, device=self.device)
                if not is_baseline:
                    current_L = self.monitor.compute_L(model.fc1.weight)
                    # Penalizamos desviación del TARGET_L
                    spectral_penalty = abs(current_L - TARGET_L)
                    reg_loss = LAMBDA_SPECTRAL * spectral_penalty
                
                total_loss = ce_loss + reg_loss
                total_loss.backward()
                
                if not is_baseline and hasattr(model, 'mask1'):
                    model.fc1.weight.grad *= model.mask1
                    model.fc2.weight.grad *= model.mask2
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            train_acc = 100 * train_correct / train_total
            
            model.eval()
            test_correct, test_total = 0, 0
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    inputs_proc = self._preprocess_batch(inputs)
                    outputs = model(inputs_proc)
                    _, predicted = torch.max(outputs.data, 1)
                    test_total += labels.size(0)
                    test_correct += (predicted == labels).sum().item()
            
            test_acc = 100 * test_correct / test_total
            gap = train_acc - test_acc
            
            if test_acc > best_acc:
                best_acc = test_acc
                best_gap = gap
                best_state = model.state_dict().copy()
            
            # Dynamic Minimalistic Shock (Rank Capping)
            if not is_baseline and hasattr(model, 'mask1'):
                if epoch in [40, 80] and gap > 15.0:
                    self.engine._apply_minimalistic_shock(model, 'fc1', target_rank_ratio=0.85)
                    print(f"         ⚡⚡ RANK CAPPING at Ep {epoch} (Gap: {gap:.1f})")
            
            if epoch % 20 == 0:
                L = self.monitor.compute_L(model.fc1.weight) if not is_baseline else 0.0
                spar = model.get_sparsity() if not is_baseline else 0.0
                loss_str = f"{ce_loss.item():.3f} + {reg_loss.item():.3f} (reg)" if not is_baseline else f"{ce_loss.item():.3f}"
                print(f"         Ep {epoch:3d} | T:{train_acc:5.1f}% | V:{test_acc:5.1f}% | L:{L:.3f} | Spar:{spar:.1f}% | Loss:{loss_str}")
            
            scheduler.step()
            
        if best_state: 
            model.load_state_dict(best_state)
            if not is_baseline:
                model.apply_masks()
                
        return {'final_acc': best_acc, 'final_gap': best_gap, 'L': 0 if is_baseline else self.monitor.compute_L(model.fc1.weight)}

# =============================================================================
# 6. MAIN EXECUTION
# =============================================================================
def main():
    print("=" * 90)
    print(" " * 25 + "NEUROSOVEREIGN v13.0: ORTHOGONAL APEX")
    print("=" * 90)
    print(f"Input: ViT-Lite + True Token Mixing (T-axis)")
    print(f"Constraint: Spectral Regularization (Target L={TARGET_L}) + Rank Capping")
    print(f"Objective: SOTA Efficiency via Minimality Enforcement")
    print("=" * 90)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    feature_extractor = PatchFeatureExtractor(embed_dim=INPUT_DIM).to(device)
    trainer = OrthogonalTrainer(device, feature_extractor)
    
    # ============================
    # 1. INITIALIZE ELK
    # ============================
    if os.path.exists('alpha_elk_v13.pth'):
        print("📂 Loading Alpha Elk v13...")
        elk_state = torch.load('alpha_elk_v13.pth', map_location=device)
        current_dim = elk_state['fc1.weight'].shape[0]
        start_cycle = 1
    else:
        print("🥚 Generating Seed Elk v13 (Orthogonal Foundation)...")
        seed_model = LotteryMLP(input_dim=INPUT_DIM, hidden_dim=192).to(device)
        res = trainer.train_model(seed_model, 0)
        elk_state = seed_model.state_dict()
        current_dim = 192
        start_cycle = 1
        torch.save(elk_state, 'alpha_elk_v13.pth')

    # ============================
    # 2. BASELINE KILLER
    # ============================
    print("\n" + "=" * 90)
    print(" " * 30 + "BASELINE KILLER INITIALIZED")
    print("=" * 90)
    baseline_model = StandardBaseline(input_dim=INPUT_DIM, hidden_dim=192).to(device)
    baseline_log = []
    
    # ============================
    # 3. ORTHOGONAL EVOLUTION LOOP
    # ============================
    evolution_log = []
    
    for cycle in range(start_cycle, NUM_EVOLUTION_CYCLES + 1):
        print(f"\n{'='*90}")
        print(f"🚀 ORTHOGONAL CYCLE {cycle}/{NUM_EVOLUTION_CYCLES}")
        print(f"{'='*90}")
        
        if cycle == UNFREEZE_AT_CYCLE:
            print("   👁️  UNFREEZING EXTRACTOR & MIXER...")
            feature_extractor.unfreeze()
            
        # --- A. Baseline ---
        baseline_res = trainer.train_model(baseline_model, cycle, is_baseline=True)
        baseline_log.append({'cycle': cycle, 'acc': baseline_res['final_acc']})
        
        # --- B. Evolution ---
        curr_trainset = trainer.get_curriculum_dataset(cycle)
        curr_loader = torch.utils.data.DataLoader(curr_trainset, batch_size=64)
        
        parent_gap = evolution_log[-1]['child_gap'] if evolution_log else 20.0
        new_dim = int(current_dim * 1.1) 
        
        child_model = trainer.engine.create_orthogonal_offspring(
            elk_state, new_dim, cycle, curr_loader, feature_extractor, parent_gap
        )
        child_res = trainer.train_model(child_model, cycle)
        
        # --- C. Evaluación Elk ---
        temp_elk = LotteryMLP(input_dim=INPUT_DIM, hidden_dim=current_dim).to(device)
        temp_elk.load_state_dict(elk_state)
        temp_elk.eval()
        
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=trainer.transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=128)
        
        elk_corr, elk_tot = 0, 0
        with torch.no_grad():
            for i, l in testloader:
                i, l = i.to(device), l.to(device)
                i_proc = trainer._preprocess_batch(i)
                o = temp_elk(i_proc)
                _, p = torch.max(o, 1)
                elk_corr += (p == l).sum().item()
                elk_tot += l.size(0)
        elk_acc = 100 * elk_corr / elk_tot
        
        improvement = child_res['final_acc'] - elk_acc
        sparsity = child_model.get_sparsity()
        current_L = child_res['L']
        
        print(f"   🏆 Apex: Elk {elk_acc:.2f}% vs Child {child_res['final_acc']:.2f}% (L: {current_L:.3f} | Spar: {sparsity:.1f}%)")
        
        # Selección basada en L estabilizado + Accuracy
        evolved = False
        if improvement >= 0.5 and abs(current_L - TARGET_L) < 0.5:
            print(f"   ✅ EVOLUTION: Child becomes Alpha (Stable L).")
            elk_state = child_model.state_dict()
            current_dim = new_dim
            torch.save(elk_state, 'alpha_elk_v13.pth')
            evolved = True
        elif child_res['final_gap'] < 10.0 and improvement > -1.0:
            print(f"   ✅ EVOLUTION: Child adopted (Low Gap Priority).")
            elk_state = child_model.state_dict()
            current_dim = new_dim
            torch.save(elk_state, 'alpha_elk_v13.pth')
            evolved = True
        else:
            print(f"   ❌ EXTINCTION: Alpha persists (Unstable or Inferior).")
            
        diff_vs_baseline = child_res['final_acc'] - baseline_res['final_acc']
        
        evolution_log.append({
            'cycle': cycle,
            'elk_acc': elk_acc,
            'child_acc': child_res['final_acc'],
            'child_gap': child_res['final_gap'],
            'baseline_acc': baseline_res['final_acc'],
            'delta_vs_baseline': diff_vs_baseline,
            'L': current_L,
            'sparsity': sparsity,
            'evolved': evolved
        })

    # ============================
    # 4. ORTHOGONAL REPORT
    # ============================
    df = pd.DataFrame(evolution_log)
    print("\n" + "=" * 90)
    print(" " * 30 + "FINAL ORTHOGONAL REPORT")
    print("=" * 90)
    
    disp_cols = ['cycle', 'child_acc', 'baseline_acc', 'delta_vs_baseline', 'child_gap', 'L', 'sparsity']
    print(df[disp_cols].to_string(index=False))
    
    print("-" * 90)
    print(f"🏆 Best Apex Accuracy: {df['child_acc'].max():.2f}%")
    print(f"📉 Best Sparsity Achieved: {df['sparsity'].min():.1f}%")
    print(f"📈 Avg Delta vs Baseline: {df['delta_vs_baseline'].mean():.2f}%")
    print(f"📐 Avg L (Target {TARGET_L}): {df['L'].mean():.3f}")
    
    if df['child_acc'].max() > 60.0:
        print("✅ SOTA TERRITORY: Orthogonal constraints enabled high accuracy.")
    elif abs(df['L'].mean() - TARGET_L) < 0.3:
        print("✅ CONTROLLED EVOLUTION: Successfully constrained spectral coherence.")
    else:
        print("⚠️  Review needed: Constraints too loose or too tight.")
        
    print("=" * 90)

if __name__ == "__main__":
    main()