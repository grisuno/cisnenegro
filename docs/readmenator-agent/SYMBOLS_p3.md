# Symbols (page 3 of 3)
Previous: [SYMBOLS_p2.md](SYMBOLS_p2.md)

| Symbol | Kind | File:Line | Signature |
|--------|------|-----------|-----------|
| `__init__` | method | `plank6.py:106` | `def __init__(self, device)` |
| `__init__` | method | `plank6.py:200` | `def __init__(self, device)` |
| `_apply_spectral_refinement` | method | `plank6.py:146` | `def _apply_spectral_refinement(self, W)` |
| `_guided_elk_mutation` | method | `plank6.py:110` | `def _guided_elk_mutation(self, old_weight, target_shape, noise_scale, refinement_steps)` |
| `compute_L` | method | `plank6.py:34` | `def compute_L(self, weight)` |
| `create_offspring_from_elk` | method | `plank6.py:158` | `def create_offspring_from_elk(self, elk_state, new_hidden_dim, generation)` |
| `detect_grokking` | method | `plank6.py:87` | `def detect_grokking(self)` |
| `forward` | method | `plank6.py:70` | `def forward(self, x)` |
| `main` | method | `plank6.py:293` | `def main()` |
| `reduce_input` | method | `plank6.py:63` | `def reduce_input(self, x)` |
| `train_phase` | method | `plank6.py:204` | `def train_phase(self, model, cycle_id)` |
| `update` | method | `plank6.py:84` | `def update(self, train_acc, test_acc, epoch)` |
| `AdvancedEvolutionEngine` | class | `plank7.py:68` | `class AdvancedEvolutionEngine` |
| `CurriculumTrainingCycle` | class | `plank7.py:177` | `class CurriculumTrainingCycle` |
| `SpectralMLP` | class | `plank7.py:45` | `class SpectralMLP(Module)` |
| `SpectralMonitor` | class | `plank7.py:29` | `class SpectralMonitor` |
| `__init__` | method | `plank7.py:30` | `def __init__(self, epsilon_c)` |
| `__init__` | method | `plank7.py:46` | `def __init__(self, input_dim, hidden_dim, num_classes)` |
| `__init__` | method | `plank7.py:72` | `def __init__(self, device)` |
| `__init__` | method | `plank7.py:178` | `def __init__(self, device)` |
| `_apply_spectral_shock` | method | `plank7.py:107` | `def _apply_spectral_shock(self, W, shock_intensity)` |
| `_gradient_nudge_inheritance` | method | `plank7.py:76` | `def _gradient_nudge_inheritance(self, child_model, elk_state, data_loader, nudge_lr)` |
| `compute_L` | method | `plank7.py:33` | `def compute_L(self, weight)` |
| `create_advanced_offspring` | method | `plank7.py:124` | `def create_advanced_offspring(self, elk_state, new_hidden_dim, cycle, data_loader)` |
| `forward` | method | `plank7.py:60` | `def forward(self, x)` |
| `get_curriculum_dataset` | method | `plank7.py:186` | `def get_curriculum_dataset(self, cycle)` |
| `main` | method | `plank7.py:283` | `def main()` |
| `reduce_input` | method | `plank7.py:54` | `def reduce_input(self, x)` |
| `train_phase` | method | `plank7.py:204` | `def train_phase(self, model, cycle)` |
| `ApexEvolutionEngine` | class | `plank8.py:136` | `class ApexEvolutionEngine` |
| `ApexTrainer` | class | `plank8.py:235` | `class ApexTrainer` |
| `LotteryMLP` | class | `plank8.py:78` | `class LotteryMLP(Module)` |
| `PatchFeatureExtractor` | class | `plank8.py:34` | `class PatchFeatureExtractor(Module)` |
| `SpectralMonitor` | class | `plank8.py:121` | `class SpectralMonitor` |
| `StandardBaseline` | class | `plank8.py:109` | `class StandardBaseline(Module)` |
| `__init__` | method | `plank8.py:39` | `def __init__(self, img_size, patch_size, in_chans, embed_dim)` |
| `__init__` | method | `plank8.py:79` | `def __init__(self, input_dim, hidden_dim, num_classes)` |
| `__init__` | method | `plank8.py:111` | `def __init__(self, input_dim, hidden_dim)` |
| `__init__` | method | `plank8.py:137` | `def __init__(self, device)` |
| `__init__` | method | `plank8.py:236` | `def __init__(self, device, feature_extractor)` |
| `_apply_dynamic_spectral_shock` | method | `plank8.py:176` | `def _apply_dynamic_spectral_shock(self, model, layer_name)` |
| `_gradient_nudge_inheritance` | method | `plank8.py:142` | `def _gradient_nudge_inheritance(self, child_model, elk_state, data_loader, feature_extractor, nudge_lr)` |
| `_preprocess_batch` | method | `plank8.py:243` | `def _preprocess_batch(self, x)` |
| `apply_masks` | method | `plank8.py:92` | `def apply_masks(self)` |
| `compute_L` | method | `plank8.py:122` | `def compute_L(self, weight)` |
| `create_apex_offspring` | method | `plank8.py:199` | `def create_apex_offspring(self, elk_state, new_hidden_dim, cycle, data_loader, feature_extractor, parent_gap)` |
| `forward` | method | `plank8.py:65` | `def forward(self, x)` |
| `forward` | method | `plank8.py:102` | `def forward(self, x)` |
| `forward` | method | `plank8.py:115` | `def forward(self, x)` |
| `freeze` | method | `plank8.py:57` | `def freeze(self)` |
| `get_curriculum_dataset` | method | `plank8.py:247` | `def get_curriculum_dataset(self, cycle)` |
| `get_sparsity` | method | `plank8.py:97` | `def get_sparsity(self)` |
| `main` | method | `plank8.py:342` | `def main()` |
| `train_model` | method | `plank8.py:253` | `def train_model(self, model, cycle, is_baseline)` |
| `unfreeze` | method | `plank8.py:61` | `def unfreeze(self)` |
| `GAT_Baseline` | class | `resmav2_1.py:182` | `class GAT_Baseline(Module)` |
| `OptimizedE8Layer` | class | `resmav2_1.py:23` | `class OptimizedE8Layer(Module)` |
| `RESMAv2Deep` | class | `resmav2_1.py:134` | `class RESMAv2Deep(Module)` |
| `RESMAv2Fast` | class | `resmav2_1.py:48` | `class RESMAv2Fast(Module)` |
| `RESMAv2Standard` | class | `resmav2_1.py:86` | `class RESMAv2Standard(Module)` |
| `__init__` | method | `resmav2_1.py:25` | `def __init__(self, in_features, out_features, edge_index, num_nodes)` |
| `__init__` | method | `resmav2_1.py:50` | `def __init__(self, input_dim, hidden_dim, edge_index, num_nodes, dropout)` |
| `__init__` | method | `resmav2_1.py:88` | `def __init__(self, input_dim, hidden_dim, edge_index, num_nodes, dropout)` |
| `__init__` | method | `resmav2_1.py:136` | `def __init__(self, input_dim, hidden_dim, edge_index, num_nodes, dropout)` |
| `__init__` | method | `resmav2_1.py:184` | `def __init__(self, input_dim, hidden_dim, dropout)` |
| `cross_validate_model` | method | `resmav2_1.py:321` | `def cross_validate_model(model_class, X, y, edge_index, num_nodes, n_splits, seed, name)` |
| `forward` | method | `resmav2_1.py:40` | `def forward(self, x)` |
| `forward` | method | `resmav2_1.py:73` | `def forward(self, x, edge_index)` |
| `forward` | method | `resmav2_1.py:113` | `def forward(self, x, edge_index)` |
| `forward` | method | `resmav2_1.py:168` | `def forward(self, x, edge_index)` |
| `forward` | method | `resmav2_1.py:194` | `def forward(self, x, edge_index)` |
| `load_elliptic_data` | method | `resmav2_1.py:207` | `def load_elliptic_data()` |
| `train_and_evaluate` | method | `resmav2_1.py:264` | `def train_and_evaluate(model, X, y, edge_index, train_idx, val_idx, epochs, lr, name, fold)` |

