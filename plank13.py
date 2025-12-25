#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuroSovereign v13.2: Controlled Isolation (FIXED)
Objective: Isolate "Token Mixing" variable in an evolutionary framework.
Method: Dual Evolutionary Chains (Apex vs Blind Structural Baseline).
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
FIXED_HIDDEN_DIM = 256 # Capacidad Congelada para ambas cadenas
TARGET_L = 1.8
UNFREEZE_AT_CYCLE = 3

# =============================================================================
# 1. MODULAR VISION EYE (CONFIGURABLE)
# =============================================================================
class TokenMixer(nn.Module):
    """Mezcla tokens entre sí (Solo para Apex)."""
    def __init__(self, num_tokens=64):
        super().__init__()
        self.mixer = nn.Linear(num_tokens, num_tokens, bias=False)
        nn.init.eye_(self.mixer.weight)
        with torch.no_grad():
            self.mixer.weight += torch.randn_like(self.mixer.weight) * 0.01

    def forward(self, x):
        x = x.transpose(1, 2) # (B, T, D) -> (B, D, T)
        x = self.mixer(x)
        x = x.transpose(1, 2) # (B, D, T) -> (B, T, D)
        return x

class PatchFeatureExtractor(nn.Module):
    """
    Extractor configurable.
    use_mixer=True -> Apex (Syntactic)
    use_mixer=False -> Blind Structural Baseline
    """
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=384, use_mixer=True):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.use_mixer = use_mixer
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        if self.use_mixer:
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
        x = self.proj(x) # (B, D, H, W)
        x = x.flatten(2) # (B, D, T)
        x = x.transpose(1, 2) # (B, T, D)
        
        if self.use_mixer:
            x = self.mixer(x) # SYNTACTIC MIXING (Solo en Apex)
        
        x = x.mean(dim=1) # Global Pool (B, D)
        return x

# =============================================================================
# 2. LOTTERY TICKET MLP (SHARED BRAIN)
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

# =============================================================================
# 3. MONITORING (FIXED RETURN)
# =============================================================================
class SpectralMonitor:
    def compute_metrics(self, weight: torch.Tensor) -> Tuple[float, int, float]:
        """
        Returns: (L, Rank_Efficient, S_vN)
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
# 4. ORTHOGONAL ENGINE (SHARED LOGIC)
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
        # Nudge genérico, funciona para Apex y Blind
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
        # Shock genérico
        layer = getattr(model, layer_name)
        W = layer.weight.data
        U, S, V = torch.svd(W)
        max_rank = S.shape[0]
        target_rank = int(max_rank * keep_ratio)
        mask = torch.zeros_like(S)
        mask[:target_rank] = 1.0
        S_capped = S * mask
        W_shocked = U @ torch.diag(S_capped) @ V.t()
        
        with torch.no_grad():
            layer.weight.data = W_shocked
            new_mask = (torch.abs(W_shocked) > 1e-5).float()
            if layer_name == 'fc1':
                model.mask1.copy_(new_mask)
            else:
                model.mask2.copy_(new_mask)

    def create_refined_offspring(self, 
                                   elk_state: Dict, 
                                   data_loader,
                                   feature_extractor) -> nn.Module:
        # Creación de hijo genérica
        child = LotteryMLP(input_dim=self.input_dim, hidden_dim=FIXED_HIDDEN_DIM).to(self.device)
        child.fc1.weight.data = elk_state['fc1.weight']
        child.fc2.weight.data = elk_state['fc2.weight']
        child.mask1.data = elk_state['mask1']
        child.mask2.data = elk_state['mask2']
        
        self._gradient_nudge_inheritance(child, elk_state, data_loader, feature_extractor, nudge_lr=0.01)
        return child

# =============================================================================
# 5. DUAL TRAINER (APEX VS BLIND)
# =============================================================================
class DualTrainer:
    def __init__(self, device: torch.device, extractor_apex=None, extractor_blind=None):
        self.device = device
        self.extractor_apex = extractor_apex
        self.extractor_blind = extractor_blind
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.monitor = SpectralMonitor()
        self.engine = OrthogonalEvolutionEngine(device)

    def _preprocess_batch(self, x, extractor):
        with torch.no_grad():
            return extractor(x)

    def get_curriculum_dataset(self, cycle: int):
        full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=self.transform)
        sizes = {1: 2500, 4: 10000, 7: 50000}
        size = sizes.get(cycle, 2500)
        return torch.utils.data.Subset(full_trainset, torch.randperm(len(full_trainset))[:size])

    def train_single_chain(self, model, cycle, chain_type='APEX'):
        """Entrena una cadena específica (Apex o Blind)."""
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
        
        # Seleccionar extractor correcto
        extractor = self.extractor_apex if chain_type == 'APEX' else self.extractor_blind
        
        for epoch in range(100):
            model.train()
            train_correct, train_total = 0, 0
            
            for inputs, labels in trainloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                inputs_proc = self._preprocess_batch(inputs, extractor)
                
                optimizer.zero_grad()
                outputs = model(inputs_proc)
                ce_loss = criterion(outputs, labels)
                ce_loss.backward()
                
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
                    inputs_proc = self._preprocess_batch(inputs, extractor)
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
            
            # Dynamic Shock (Igual para ambas cadenas)
            # L, rank_eff, _ = self.monitor.compute_metrics(model.fc1.weight) # Fixed unpack
            L, rank_eff, _ = self.monitor.compute_metrics(model.fc1.weight)
            if epoch in [40, 80] and (gap > 15.0 or rank_eff > (FIXED_HIDDEN_DIM * 0.7)):
                self.engine._apply_rank_capping_shock(model, 'fc1', keep_ratio=0.85)
            
            if epoch % 20 == 0:
                spar = model.get_sparsity()
                print(f"         [{chain_type}] Ep {epoch:3d} | V:{test_acc:5.1f}% | Gap:{gap:4.1f} | L:{L:.3f} | Spar:{spar:.1f}%")
            
            scheduler.step()
            
        if best_state: 
            model.load_state_dict(best_state)
            model.apply_masks()
                
        L_final, Rank_final, _ = self.monitor.compute_metrics(model.fc1.weight)
        return {'final_acc': best_acc, 'final_gap': best_gap, 'L': L_final, 'rank': Rank_final}

# =============================================================================
# 6. MAIN EXECUTION (DUAL ISOLATION)
# =============================================================================
def main():
    print("=" * 90)
    print(" " * 20 + "NEUROSOVEREIGN v13.2: CONTROLLED ISOLATION")
    print("=" * 90)
    print(f"Experiment: A/B Testing of Token Mixing under Evolutionary Pressure.")
    print(f"Chain A: APEX (ViT-Lite + Mixer)")
    print(f"Chain B: BLIND (ViT-Lite + No Mixer) <- STRUCTURAL BASELINE")
    print(f"Control: Both use same Rank Capping, Nudges, and Selection Logic.")
    print("=" * 90)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Inicializar Extractores (A y B)
    extractor_apex = PatchFeatureExtractor(embed_dim=INPUT_DIM, use_mixer=True).to(device)
    extractor_blind = PatchFeatureExtractor(embed_dim=INPUT_DIM, use_mixer=False).to(device)
    
    trainer = DualTrainer(device, extractor_apex, extractor_blind)
    
    # ============================
    # 1. INITIALIZE DUAL ELKS
    # ============================
    if os.path.exists('alpha_elk_v13_2_apex.pth'):
        print("📂 Loading Dual Alphas v13.2...")
        elk_state_apex = torch.load('alpha_elk_v13_2_apex.pth', map_location=device)
        elk_state_blind = torch.load('alpha_elk_v13_2_blind.pth', map_location=device)
        start_cycle = 1
    else:
        print("🥚 Generating Seed Elks (Apex & Blind)...")
        seed_apex = LotteryMLP(input_dim=INPUT_DIM, hidden_dim=FIXED_HIDDEN_DIM).to(device)
        seed_blind = LotteryMLP(input_dim=INPUT_DIM, hidden_dim=FIXED_HIDDEN_DIM).to(device)
        
        # Entrenamiento inicial rápido para sembrar
        print("   - Seeding Apex...")
        trainer.train_single_chain(seed_apex, 0, 'APEX')
        print("   - Seeding Blind...")
        trainer.train_single_chain(seed_blind, 0, 'BLIND')
        
        elk_state_apex = seed_apex.state_dict()
        elk_state_blind = seed_blind.state_dict()
        start_cycle = 1
        torch.save(elk_state_apex, 'alpha_elk_v13_2_apex.pth')
        torch.save(elk_state_blind, 'alpha_elk_v13_2_blind.pth')

    # ============================
    # 2. DUAL EVOLUTION LOOP
    # ============================
    evolution_log = []
    
    for cycle in range(start_cycle, NUM_EVOLUTION_CYCLES + 1):
        print(f"\n{'='*90}")
        print(f"🔬 DUAL ISOLATION CYCLE {cycle}/{NUM_EVOLUTION_CYCLES}")
        print(f"{'='*90}")
        
        if cycle == UNFREEZE_AT_CYCLE:
            print("   👁️  UNFREEZING BOTH EXTRACTORS (FAIR EVOLUTION)...")
            extractor_apex.unfreeze()
            extractor_blind.unfreeze()
            
        # --- DATA PREP ---
        curr_trainset = trainer.get_curriculum_dataset(cycle)
        curr_loader = torch.utils.data.DataLoader(curr_trainset, batch_size=64)
        
        # --- EVOLUCIÓN A: APEX ---
        parent_gap_apex = evolution_log[-1]['gap_apex'] if evolution_log else 20.0
        child_apex = trainer.engine.create_refined_offspring(elk_state_apex, curr_loader, extractor_apex)
        res_apex = trainer.train_single_chain(child_apex, cycle, 'APEX')
        
        # --- EVOLUCIÓN B: BLIND BASELINE ---
        parent_gap_blind = evolution_log[-1]['gap_blind'] if evolution_log else 20.0
        child_blind = trainer.engine.create_refined_offspring(elk_state_blind, curr_loader, extractor_blind)
        res_blind = trainer.train_single_chain(child_blind, cycle, 'BLIND')
        
        # --- SELECCIÓN NATURAL (INDEPENDIENTE) ---
        # Cada cadena evoluciona según sus propios méritos
        
        # Apex Selection
        improved_apex = res_apex['final_acc'] - (evolution_log[-1]['acc_apex'] if evolution_log else 0)
        if improved_apex >= 0.5 and abs(res_apex['L'] - TARGET_L) < 0.6:
            print(f"   ✅ APEX EVOLVED ({improved_apex:+.1f})")
            elk_state_apex = child_apex.state_dict()
            torch.save(elk_state_apex, 'alpha_elk_v13_2_apex.pth')
        else:
            print(f"   ❌ APEX STAGNANT")

        # Blind Selection
        improved_blind = res_blind['final_acc'] - (evolution_log[-1]['acc_blind'] if evolution_log else 0)
        if improved_blind >= 0.5 and abs(res_blind['L'] - TARGET_L) < 0.6:
            print(f"   ✅ BLIND EVOLVED ({improved_blind:+.1f})")
            elk_state_blind = child_blind.state_dict()
            torch.save(elk_state_blind, 'alpha_elk_v13_2_blind.pth')
        else:
            print(f"   ❌ BLIND STAGNANT")
            
        delta_structure = res_apex['final_acc'] - res_blind['final_acc']
        
        evolution_log.append({
            'cycle': cycle,
            'acc_apex': res_apex['final_acc'],
            'acc_blind': res_blind['final_acc'],
            'delta_structure': delta_structure,
            'gap_apex': res_apex['final_gap'],
            'gap_blind': res_blind['final_gap'],
            'L_apex': res_apex['L'],
            'L_blind': res_blind['L'],
            'spar_apex': child_apex.get_sparsity(),
            'spar_blind': child_blind.get_sparsity()
        })

    # ============================
    # 3. ISOLATION REPORT
    # ============================
    df = pd.DataFrame(evolution_log)
    print("\n" + "=" * 90)
    print(" " * 30 + "CONTROLLED ISOLATION REPORT")
    print("=" * 90)
    
    disp_cols = ['cycle', 'acc_apex', 'acc_blind', 'delta_structure', 'gap_apex', 'gap_blind', 'L_apex', 'L_blind']
    print(df[disp_cols].to_string(index=False))
    
    print("-" * 90)
    print(f"🔬 FINAL COMPARISON:")
    print(f"   Apex Best:   {df['acc_apex'].max():.2f}%")
    print(f"   Blind Best:  {df['acc_blind'].max():.2f}%")
    print(f"   Structure Δ: {df['delta_structure'].mean():.2f}% (Avg)")
    
    if df['delta_structure'].mean() > 2.0:
        print("\n✅ STRUCTURAL VALIDATED:")
        print("   Token Mixing consistently outperforms Structural Baseline")
        print("   while maintaining fixed capacity and same evolutionary pressure.")
    elif df['delta_structure'].mean() > 0.5:
        print("\n✅ TREND VALIDATED:")
        print("   Token Mixing shows slight advantage over baseline.")
    else:
        print("\n⚠️  NO STRUCTURAL ADVANTAGE:")
        print("   Baseline performs equally. The mixing adds no value in this setup.")
        
    print("=" * 90)

if __name__ == "__main__":
    main()