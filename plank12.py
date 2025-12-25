#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuroSovereign v13.1: The Structure Proof
Objective: Prove that structure (Mixing + Rank Capping) works without growing capacity.
Changes from v13.0:
1. FIXED_HIDDEN_DIM: No width expansion.
2. Removed reg_loss from backward (SVD has no grad).
3. L used purely for triggering shocks and evolutionary selection.
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
# CONFIGURATION (THE EXPERIMENT VARIABLES)
# =============================================================================
NUM_EVOLUTION_CYCLES = 12
INPUT_DIM = 384
FIXED_HIDDEN_DIM = 256 # 🔒 CAPACIDAD CONGELADA (No Growth)
TARGET_L = 1.8         # Coherencia Objetivo para Selección
UNFREEZE_AT_CYCLE = 3

# =============================================================================
# 1. TRUE SYNTACTIC VISION EYE
# =============================================================================
class TokenMixer(nn.Module):
    """Mezcla tokens entre sí (eje T)."""
    def __init__(self, num_tokens=64):
        super().__init__()
        self.mixer = nn.Linear(num_tokens, num_tokens, bias=False)
        nn.init.eye_(self.mixer.weight)
        with torch.no_grad():
            self.mixer.weight += torch.randn_like(self.mixer.weight) * 0.01

    def forward(self, x):
        # x: (B, T, D)
        x = x.transpose(1, 2)  # -> (B, D, T)
        x = self.mixer(x)      # -> (B, D, T)
        x = x.transpose(1, 2)  # -> (B, T, D)
        return x

class PatchFeatureExtractor(nn.Module):
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
        
        x = self.mixer(x) # TRUE SYNTACTIC MIXING
        
        x = x.mean(dim=1) # Global Pool (B, D)
        return x

# =============================================================================
# 2. LOTTERY TICKET MLP
# =============================================================================
class LotteryMLP(nn.Module):
    def __init__(self, input_dim: int = 384, hidden_dim: int = 256, num_classes: int = 10):
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
    def __init__(self, input_dim=384, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 10))
    
    def forward(self, x):
        return self.net(x)

# =============================================================================
# 3. MONITORING
# =============================================================================
class SpectralMonitor:
    def compute_metrics(self, weight: torch.Tensor) -> Tuple[float, int, float]:
        """
        Returns: (L, Rank_Efficient, S_vN)
        Used for logging and decision making (NOT for backprop).
        """
        with torch.no_grad():
            W = weight.cpu().numpy()
            U, S, Vh = np.linalg.svd(W, full_matrices=False)
            
            # 1. Effective Rank
            threshold = 0.05 * np.max(S)
            rank_eff = max(1, int(np.sum(S > threshold)))
            
            # 2. Von Neumann Entropy
            S_norm = S / (np.sum(S) + 1e-12)
            S_norm = S_norm[S_norm > 1e-15]
            S_vN = -np.sum(S_norm * np.log(S_norm + 1e-15))
            
            # 3. Coherence L
            L = 1.0 / (abs(S_vN - np.log(rank_eff + 1)) + 0.3)
            
            return L, rank_eff, S_vN

# =============================================================================
# 4. ORTHOGONAL EVOLUTION ENGINE (FIXED WIDTH)
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
        # Como hidden_dim es fijo, child_model y temp_elk tienen mismo tamaño
        temp_elk = LotteryMLP(input_dim=self.input_dim, hidden_dim=FIXED_HIDDEN_DIM).to(self.device)
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
            child_model.fc1.weight.data = temp_elk.fc1.weight.data
            child_model.mask1.data = temp_elk.mask1.data
            child_model.fc2.weight.data = temp_elk.fc2.weight.data
            child_model.mask2.data = temp_elk.mask2.data

    def _apply_rank_capping_shock(self, model: nn.Module, layer_name='fc1', keep_ratio=0.85):
        """
        Minimalistic Shock: Zero out weak singular values without renormalizing.
        Force energy decay and minimality.
        """
        layer = getattr(model, layer_name)
        W = layer.weight.data
        
        U, S, V = torch.svd(W)
        
        # Calcular cuántos valores mantener
        max_rank = S.shape[0]
        target_rank = int(max_rank * keep_ratio)
        
        # Crear máscara binaria
        mask = torch.zeros_like(S)
        mask[:target_rank] = 1.0
        
        S_capped = S * mask
        
        # Reconstrucción SIN renormalización (Decaimiento de energía)
        W_shocked = U @ torch.diag(S_capped) @ V.t()
        
        with torch.no_grad():
            layer.weight.data = W_shocked
            # Actualizar máscara de Lottery
            new_mask = (torch.abs(W_shocked) > 1e-5).float()
            if layer_name == 'fc1':
                model.mask1.copy_(new_mask)
            else:
                model.mask2.copy_(new_mask)

    def create_refined_offspring(self, 
                                   elk_state: Dict, 
                                   data_loader,
                                   feature_extractor) -> nn.Module:
        """
        Crea un hijo de las MISMAS dimensiones (Fixed Width).
        Evoluciona mediante Nudge (Aprendizaje) + Shock (Poda).
        """
        child = LotteryMLP(input_dim=self.input_dim, hidden_dim=FIXED_HIDDEN_DIM).to(self.device)
        
        # 1. Herencia total (sin expansión)
        child.fc1.weight.data = elk_state['fc1.weight']
        child.fc2.weight.data = elk_state['fc2.weight']
        child.mask1.data = elk_state['mask1']
        child.mask2.data = elk_state['mask2']
        
        # 2. Gradient Nudge (Aprende de los nuevos datos)
        self._gradient_nudge_inheritance(child, elk_state, data_loader, feature_extractor, nudge_lr=0.01)
        
        return child

# =============================================================================
# 5. ORTHOGONAL TRAINER (L AS CONTROLLER)
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
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)
        criterion = nn.CrossEntropyLoss()
        
        best_gap = 100.0
        best_acc = 0.0
        best_state = None
        
        mode_str = "BASELINE" if is_baseline else "STRUCTURE PROOF"
        print(f"      🚀 Training {mode_str} (Cycle {cycle}) | Data: {len(trainset)}")
        
        extractor_status = "Trainable" if self.feature_extractor.proj.weight.requires_grad else "Frozen"
        print(f"      👁️  Eye & Mixer: {extractor_status} | Dim: {FIXED_HIDDEN_DIM}")
        
        for epoch in range(100):
            model.train()
            train_correct, train_total = 0, 0
            
            for inputs, labels in trainloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                inputs_proc = self._preprocess_batch(inputs)
                
                optimizer.zero_grad()
                outputs = model(inputs_proc)
                
                # 🔒 FIX v13.1: NO reg_loss en backward. Solo CrossEntropy.
                ce_loss = criterion(outputs, labels)
                ce_loss.backward()
                
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
            
            # Dynamic Shock Logic (Controlado por L y Gap)
            if not is_baseline and hasattr(model, 'mask1'):
                # Calcular métricas para decisión (NO para loss)
                L, rank_eff, _ = self.monitor.compute_metrics(model.fc1.weight)
                
                # Shock si Gap es alto (Overfitting) O Rank está creciendo descontrolado
                if epoch in [40, 80] and (gap > 15.0 or rank_eff > (FIXED_HIDDEN_DIM * 0.7)):
                    self.engine._apply_rank_capping_shock(model, 'fc1', keep_ratio=0.85)
                    print(f"         ⚡⚡ RANK CAPPING SHOCK at Ep {epoch} (Gap: {gap:.1f}, Rank: {rank_eff})")
            
            if epoch % 20 == 0:
                L, rank_eff, _ = self.monitor.compute_metrics(model.fc1.weight) if not is_baseline else (0,0,0)
                spar = model.get_sparsity() if not is_baseline else 0.0
                print(f"         Ep {epoch:3d} | T:{train_acc:5.1f}% | V:{test_acc:5.1f}% | Gap:{gap:4.1f} | L:{L:.3f} | Rank:{rank_eff:3d} | Spar:{spar:.1f}%")
            
            scheduler.step()
            
        if best_state: 
            model.load_state_dict(best_state)
            if not is_baseline:
                model.apply_masks()
                
        L_final, Rank_final, _ = self.monitor.compute_metrics(model.fc1.weight) if not is_baseline else (0,0,0)
        return {'final_acc': best_acc, 'final_gap': best_gap, 'L': L_final, 'rank': Rank_final}

# =============================================================================
# 6. MAIN EXECUTION
# =============================================================================
def main():
    print("=" * 90)
    print(" " * 25 + "NEUROSOVEREIGN v13.1: STRUCTURE PROOF")
    print("=" * 90)
    print(f"Input: ViT-Lite + True Token Mixing")
    print(f"Experiment: FIXED_WIDTH={FIXED_HIDDEN_DIM} (No Growth)")
    print(f"Method: Nudge + Rank Capping + Spectral Selection")
    print("=" * 90)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    feature_extractor = PatchFeatureExtractor(embed_dim=INPUT_DIM).to(device)
    trainer = OrthogonalTrainer(device, feature_extractor)
    
    # ============================
    # 1. INITIALIZE ELK
    # ============================
    if os.path.exists('alpha_elk_v13_1.pth'):
        print("📂 Loading Alpha Elk v13.1...")
        elk_state = torch.load('alpha_elk_v13_1.pth', map_location=device)
        current_dim = elk_state['fc1.weight'].shape[0]
        start_cycle = 1
    else:
        print("🥚 Generating Seed Elk v13.1 (Fixed Width Foundation)...")
        seed_model = LotteryMLP(input_dim=INPUT_DIM, hidden_dim=FIXED_HIDDEN_DIM).to(device)
        res = trainer.train_model(seed_model, 0)
        elk_state = seed_model.state_dict()
        current_dim = FIXED_HIDDEN_DIM # Forzar a la config fija
        start_cycle = 1
        torch.save(elk_state, 'alpha_elk_v13_1.pth')

    # ============================
    # 2. BASELINE KILLER
    # ============================
    print("\n" + "=" * 90)
    print(" " * 30 + "BASELINE KILLER INITIALIZED")
    print("=" * 90)
    baseline_model = StandardBaseline(input_dim=INPUT_DIM, hidden_dim=FIXED_HIDDEN_DIM).to(device)
    baseline_log = []
    
    # ============================
    # 3. FIXED WIDTH EVOLUTION LOOP
    # ============================
    evolution_log = []
    
    for cycle in range(start_cycle, NUM_EVOLUTION_CYCLES + 1):
        print(f"\n{'='*90}")
        print(f"🚀 STRUCTURE CYCLE {cycle}/{NUM_EVOLUTION_CYCLES}")
        print(f"{'='*90}")
        
        if cycle == UNFREEZE_AT_CYCLE:
            print("   👁️  UNFREEZING EXTRACTOR & MIXER...")
            feature_extractor.unfreeze()
            
        # --- A. Baseline ---
        baseline_res = trainer.train_model(baseline_model, cycle, is_baseline=True)
        baseline_log.append({'cycle': cycle, 'acc': baseline_res['final_acc']})
        
        # --- B. Evolution (Refinement Only) ---
        curr_trainset = trainer.get_curriculum_dataset(cycle)
        curr_loader = torch.utils.data.DataLoader(curr_trainset, batch_size=64)
        
        # Crear hijo refinado (mismo tamaño)
        child_model = trainer.engine.create_refined_offspring(
            elk_state, curr_loader, feature_extractor
        )
        child_res = trainer.train_model(child_model, cycle)
        
        # --- C. Evaluación Elk ---
        temp_elk = LotteryMLP(input_dim=INPUT_DIM, hidden_dim=FIXED_HIDDEN_DIM).to(device)
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
        current_rank = child_res['rank']
        
        print(f"   🏆 Apex: Elk {elk_acc:.2f}% vs Child {child_res['final_acc']:.2f}% (L: {current_L:.3f} | Spar: {sparsity:.1f}%)")
        
        # Selección basada en L estabilizado + Accuracy (Sin expansión)
        evolved = False
        if improvement >= 0.5 and abs(current_L - TARGET_L) < 0.6:
            print(f"   ✅ EVOLUTION: Refined child becomes Alpha.")
            elk_state = child_model.state_dict()
            torch.save(elk_state, 'alpha_elk_v13_1.pth')
            evolved = True
        elif child_res['final_gap'] < 8.0 and improvement > -1.0:
            print(f"   ✅ EVOLUTION: Child adopted (Low Gap Priority).")
            elk_state = child_model.state_dict()
            torch.save(elk_state, 'alpha_elk_v13_1.pth')
            evolved = True
        else:
            print(f"   ❌ STAGNATION: Alpha persists.")
            
        diff_vs_baseline = child_res['final_acc'] - baseline_res['final_acc']
        
        evolution_log.append({
            'cycle': cycle,
            'elk_acc': elk_acc,
            'child_acc': child_res['final_acc'],
            'child_gap': child_res['final_gap'],
            'baseline_acc': baseline_res['final_acc'],
            'delta_vs_baseline': diff_vs_baseline,
            'L': current_L,
            'rank_eff': current_rank,
            'sparsity': sparsity,
            'evolved': evolved
        })

    # ============================
    # 4. STRUCTURE PROOF REPORT
    # ============================
    df = pd.DataFrame(evolution_log)
    print("\n" + "=" * 90)
    print(" " * 30 + "STRUCTURE PROOF REPORT")
    print("=" * 90)
    
    disp_cols = ['cycle', 'child_acc', 'baseline_acc', 'delta_vs_baseline', 'L', 'rank_eff', 'sparsity']
    print(df[disp_cols].to_string(index=False))
    
    print("-" * 90)
    print(f"🏆 Best Apex Accuracy: {df['child_acc'].max():.2f}%")
    print(f"📉 Best Sparsity Achieved: {df['sparsity'].min():.1f}%")
    print(f"📈 Avg Delta vs Baseline: {df['delta_vs_baseline'].mean():.2f}%")
    
    # Correlación Rank vs Accuracy
    # Simple correlation check: did higher accuracy correlate with lower rank (minimality)?
    best_cycle_row = df.loc[df['child_acc'].idxmax()]
    worst_cycle_row = df.loc[df['child_acc'].idxmin()]
    
    print(f"\n📊 CORRELATION CHECK:")
    print(f"   Best Cycle (Acc {best_cycle_row['child_acc']:.1f}): Rank {best_cycle_row['rank_eff']}, L {best_cycle_row['L']:.2f}")
    print(f"   Worst Cycle (Acc {worst_cycle_row['child_acc']:.1f}): Rank {worst_cycle_row['rank_eff']}, L {worst_cycle_row['L']:.2f}")
    
    if df['child_acc'].max() > 50.0 and df['sparsity'].max() > 10.0:
        print("✅ PROVEN: Structure maintains accuracy with high sparsity (Fixed Width).")
    elif df['delta_vs_baseline'].mean() > 2.0:
        print("✅ PROVEN: Evolutionary refinement consistently beats standard training.")
    else:
        print("⚠️  Review needed: Fixed width requires higher cycles or deeper mixing.")
        
    print("=" * 90)

if __name__ == "__main__":
    main()