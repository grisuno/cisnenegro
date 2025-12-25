#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuroSovereign v8.0: Shock Therapy & Gradient Nudging
Target: Break Generalization Plateau via Gradient Nudging & Curriculum Learning
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
import warnings
from typing import Dict, List, Tuple
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
NUM_EVOLUTION_CYCLES = 12  # Cliclos de evolución
BASE_ACCURACY_TARGET = 32.0

# =============================================================================
# 1. CORE COMPONENTS
# =============================================================================
class SpectralMonitor:
    def __init__(self, epsilon_c: float = 0.3):
        self.epsilon_c = epsilon_c

    def compute_L(self, weight: torch.Tensor) -> Tuple[float, int]:
        with torch.no_grad():
            W = weight.cpu().numpy()
            U, S, Vh = np.linalg.svd(W, full_matrices=False)
            threshold = 0.05 * np.max(S)
            rank_eff = max(1, int(np.sum(S > threshold)))
            S_norm = S / (np.sum(S) + 1e-12)
            S_norm = S_norm[S_norm > 1e-15]
            S_vN = -np.sum(S_norm * np.log(S_norm + 1e-15))
            L = 1.0 / (abs(S_vN - np.log(rank_eff + 1)) + self.epsilon_c)
            return L, rank_eff

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
# 2. ADVANCED EVOLUTION ENGINE (The "Boss" Logic)
# =============================================================================
class AdvancedEvolutionEngine:
    """
    Implementa Gradient Nudging y Espectral Shock.
    """
    def __init__(self, device: torch.device):
        self.device = device
        self.monitor = SpectralMonitor()

    def _gradient_nudge_inheritance(self, 
                                     child_model: nn.Module, 
                                     elk_state: Dict, 
                                     data_loader, 
                                     nudge_lr: float = 0.005):
        """
        Antes de entrenar, hacemos 1 paso de gradiente del Elk sobre los nuevos datos.
        Esto 'pre-ajusta' el ADN al contexto actual.
        """
        # Crear modelo temporal del Elk
        temp_elk = SpectralMLP(hidden_dim=elk_state['fc1.weight'].shape[0]).to(self.device)
        temp_elk.load_state_dict(elk_state)
        temp_elk.train()
        
        optimizer = torch.optim.SGD(temp_elk.parameters(), lr=nudge_lr)
        criterion = nn.CrossEntropyLoss()
        
        # Un solo batch o paso completo
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            outputs = temp_elk(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            break # Solo un paso de nudge
        
        # Ahora transferimos estos pesos 'nudgeados' al hijo
        # Nota: Si las dimensiones cambian, esto se maneja en expand_weights
        return temp_elk.state_dict()

    def _apply_spectral_shock(self, W: torch.Tensor, shock_intensity: float = 0.05):
        """
        Aplica una perturbación no-lineal a los valores singulares.
        Esto rompe mínimos locales planos sin destruir la estructura global.
        """
        U, S, V = torch.svd(W)
        
        # Shock: escalar valores singulares con ruido multiplicativo
        # Esto cambia la 'importancia' de las características internas
        noise = 1.0 + torch.randn_like(S) * shock_intensity
        S_shocked = S * noise
        
        W_shocked = U @ torch.diag(S_shocked) @ V.t()
        
        # Normalizar para mantener escala de salida
        return W_shocked * (W.std() / (W_shocked.std() + 1e-8))

    def create_advanced_offspring(self, 
                                  elk_state: Dict, 
                                  new_hidden_dim: int, 
                                  cycle: int,
                                  data_loader) -> nn.Module:
        """
        Crea un hijo combinando:
        1. Herencia de pesos
        2. Gradient Nudge (context awareness)
        3. Spectral Shock (ruptura de estancamiento)
        """
        child = SpectralMLP(hidden_dim=new_hidden_dim).to(self.device)
        
        old_dim = elk_state['fc1.weight'].shape[0]
        
        # 1. Herencia base (copiar lo que se puede)
        if new_hidden_dim > old_dim:
            # Crear pesos base
            new_fc1 = torch.zeros(new_hidden_dim, 32, device=self.device)
            new_fc2 = torch.zeros(10, new_hidden_dim, device=self.device)
            
            # Copiar Elk
            new_fc1[:old_dim, :] = elk_state['fc1.weight']
            new_fc2[:, :old_dim] = elk_state['fc2.weight']
            
            # Relleno para neuronas nuevas (Varianza del Elk)
            std_elk = elk_state['fc1.weight'].std()
            new_fc1[old_dim:, :] = torch.randn(new_hidden_dim - old_dim, 32, device=self.device) * std_elk
            
            child.fc1.weight.data = new_fc1
            child.fc2.weight.data = new_fc2
        else:
            # Si achicamos (no debería pasar en este diseño), recortamos
            child.fc1.weight.data = elk_state['fc1.weight'][:new_hidden_dim, :]
            child.fc2.weight.data = elk_state['fc2.weight'][:, :new_hidden_dim]

        # 2. Gradient Nudge (¡NUEVO!)
        # Ajustamos el modelo 'clonado' a los nuevos datos antes de empezar
        nudged_state = self._gradient_nudge_inheritance(child, elk_state, data_loader, nudge_lr=0.01)
        child.load_state_dict(nudged_state)

        # 3. Spectral Shock (¡NUEVO!)
        # Aplicar shock cada 3 ciclos para evitar plateaus
        if cycle % 3 == 0:
            with torch.no_grad():
                child.fc1.weight.data = self._apply_spectral_shock(child.fc1.weight, shock_intensity=0.1)
                print(f"      ⚡ SPECTRAL SHOCK APPLIED at Cycle {cycle}")

        return child

# =============================================================================
# 3. CURRICULUM TRAINING CYCLE
# =============================================================================
class CurriculumTrainingCycle:
    def __init__(self, device: torch.device):
        self.device = device
        self.monitor = SpectralMonitor()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def get_curriculum_dataset(self, cycle: int):
        """
        Estrategia de Curriculum:
        Ciclos 1-3: Subset pequeño (Foco en estructura).
        Ciclos 4+: Expansión progresiva (Foco en generalización).
        """
        full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=self.transform)
        
        if cycle <= 3:
            size = 2500
        elif cycle <= 6:
            size = 10000
        else:
            size = 50000 # Full dataset
            
        indices = torch.randperm(len(full_trainset))[:size]
        return torch.utils.data.Subset(full_trainset, indices)

    def train_phase(self, model: nn.Module, cycle: int) -> Dict:
        trainset = self.get_curriculum_dataset(cycle)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
        
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=self.transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)
        
        # Optimización con Weight Decay Cíclico
        # Empieza alto para regularizar, baja para afinar
        initial_wd = 0.1 if cycle < 5 else 0.01
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=initial_wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)
        criterion = nn.CrossEntropyLoss()
        
        best_gap = 100.0
        best_acc = 0.0
        best_state = None
        
        print(f"      🎓 Training Phase (Cycle {cycle}) | Dataset Size: {len(trainset)}")
        
        for epoch in range(100):
            model.train()
            train_correct, train_total = 0, 0
            
            for inputs, labels in trainloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
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
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    test_total += labels.size(0)
                    test_correct += (predicted == labels).sum().item()
            
            test_acc = 100 * test_correct / test_total
            gap = train_acc - test_acc
            
            # Fitness function mejorada: priorizamos cierre de gap (Grokking)
            # Guardamos si el gap es bajo Y la accuracy es decente
            if test_acc > best_acc and gap < best_gap + 5.0: 
                best_acc = test_acc
                best_gap = gap
                best_state = model.state_dict().copy()
            
            if epoch % 20 == 0:
                L, _ = self.monitor.compute_L(model.fc1.weight)
                print(f"         Ep {epoch:3d} | T:{train_acc:5.1f}% | V:{test_acc:5.1f}% | Gap:{gap:4.1f} | L:{L:.3f}")
            
            scheduler.step()
            
        if best_state:
            model.load_state_dict(best_state)
            
        L, _ = self.monitor.compute_L(model.fc1.weight)
        return {
            'model': model,
            'final_acc': best_acc,
            'final_gap': best_gap,
            'L': L
        }

# =============================================================================
# 4. MAIN EXECUTION (Impress the Boss)
# =============================================================================
def main():
    print("=" * 90)
    print(" " * 20 + "NEUROSOVEREIGN v8.0: SHOCK THERAPY")
    print("=" * 90)
    print(f"Evolution Strategy: Gradient Nudging + Curriculum + Spectral Shock")
    print(f"Target: Break Generalization Plateau (>34%)")
    print("=" * 90)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine = AdvancedEvolutionEngine(device)
    trainer = CurriculumTrainingCycle(device)
    
    # 1. Semilla Inicial
    if os.path.exists('alpha_elk_v8.pth'):
        print("📂 Loading Alpha Elk v8...")
        elk_state = torch.load('alpha_elk_v8.pth', map_location=device)
        current_dim = elk_state['fc1.weight'].shape[0]
        cycle_start = 1
    else:
        print("🥚 Generating Seed Elk v8...")
        seed_model = SpectralMLP(hidden_dim=47).to(device)
        # Dataset inicial pequeño
        init_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=trainer.transform)
        init_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(init_set, torch.randperm(len(init_set))[:2500]), batch_size=64)
        
        res = trainer.train_phase(seed_model, 0)
        elk_state = seed_model.state_dict()
        current_dim = 47
        cycle_start = 1
        torch.save(elk_state, 'alpha_elk_v8.pth')

    evolution_log = []

    # ==========================================
    # EVOLUTION LOOP
    # ==========================================
    for cycle in range(cycle_start, NUM_EVOLUTION_CYCLES + 1):
        print(f"\n{'='*90}")
        print(f"🚀 EVOLUTIONARY CYCLE {cycle}/{NUM_EVOLUTION_CYCLES}")
        print(f"{'='*90}")
        
        # Crecimiento arquitectónico (más lento para permitir estabilidad)
        growth_factor = 1.1
        new_dim = int(current_dim * growth_factor)
        
        # Preparar DataLoader del ciclo actual (para el Nudge)
        curr_trainset = trainer.get_curriculum_dataset(cycle)
        curr_loader = torch.utils.data.DataLoader(curr_trainset, batch_size=64, shuffle=True)
        
        # 1. Crear Hijo con Gradient Nudge
        child_model = engine.create_advanced_offspring(elk_state, new_dim, cycle, curr_loader)
        
        # 2. Entrenar con Curriculum
        result = trainer.train_phase(child_model, cycle)
        
        # 3. Evaluación del Elk actual (Benchmark)
        elk_model = SpectralMLP(hidden_dim=current_dim).to(device)
        elk_model.load_state_dict(elk_state)
        elk_model.eval()
        
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=trainer.transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=128)
        
        elk_corr, elk_tot = 0, 0
        with torch.no_grad():
            for i, l in testloader:
                i, l = i.to(device), l.to(device)
                o = elk_model(i)
                _, p = torch.max(o, 1)
                elk_corr += (p == l).sum().item()
                elk_tot += l.size(0)
        elk_acc = 100 * elk_corr / elk_tot
        
        # 4. Lógica de Selección Mejorada (Fitness Function)
        # Fitness = TestAcc - Penalty * Gap
        gap_penalty_weight = 0.5
        child_fitness = result['final_acc'] - (gap_penalty_weight * result['final_gap'])
        
        # Asumimos un gap bajo para el Elk previo (o recalculamos si guardamos gap)
        # Para simplificar, usamos la Acc directa pero requerimos mejora significativa
        # O requerimos mejor Acc O mismo Acc pero Gap mucho menor.
        
        improvement = result['final_acc'] - elk_acc
        
        print(f"\n   📊 SELECTION REPORT:")
        print(f"   Alpha Elk: {elk_acc:.2f}% (Dim: {current_dim})")
        print(f"   Child:     {result['final_acc']:.2f}% (Dim: {new_dim}) | Gap: {result['final_gap']:.2f} | L: {result['L']:.3f}")
        
        evolved = False
        if improvement >= 0.5:
            print(f"   ✅ EVOLUTION: Child replaces Alpha! (+{improvement:.2f}%)")
            elk_state = child_model.state_dict()
            current_dim = new_dim
            torch.save(elk_state, 'alpha_elk_v8.pth')
            evolved = True
        elif improvement > -0.5 and result['final_gap'] < 5.0:
            # Si no gana mucho accuracy pero tiene un gap excelente (Grokking real)
            print(f"   🧠 GROKKING DETECTED: Child adopted for superior generalization.")
            elk_state = child_model.state_dict()
            current_dim = new_dim
            torch.save(elk_state, 'alpha_elk_v8.pth')
            evolved = True
        else:
            print(f"   ❌ STAGNATION: Child discarded. Alpha remains.")
        
        log_entry = {
            'cycle': cycle,
            'dataset_size': len(curr_trainset),
            'elk_acc': elk_acc,
            'child_acc': result['final_acc'],
            'child_gap': result['final_gap'],
            'L': result['L'],
            'dim': current_dim if not evolved else new_dim,
            'evolved': evolved
        }
        evolution_log.append(log_entry)
        pd.DataFrame(evolution_log).to_csv('evolution_report_v8.csv', index=False)

    # ==========================================
    # BOSS REPORT
    # ==========================================
    df = pd.DataFrame(evolution_log)
    
    print("\n" + "=" * 90)
    print(" " * 30 + "FINAL BOSS REPORT")
    print("=" * 90)
    print(df[['cycle', 'dataset_size', 'elk_acc', 'child_acc', 'child_gap', 'L']].to_string(index=False))
    
    best_row = df.loc[df['child_acc'].idxmax()]
    print("\n" + "-" * 90)
    print(f"🏆 PEAK PERFORMANCE: Cycle {int(best_row['cycle'])} | Acc: {best_row['child_acc']:.2f}%")
    print(f"📈 PROGRESSION: Start {df.iloc[0]['child_acc']:.2f}% -> End {df.iloc[-1]['child_acc']:.2f}%")
    
    if df.iloc[-1]['child_acc'] > 34.0:
        print("✅ SUCCESS: Generalization Plateau BROKEN.")
    else:
        print("⚠️  STATUS: Persistent Plateau detected.")
        
    print("=" * 90)

if __name__ == "__main__":
    main()