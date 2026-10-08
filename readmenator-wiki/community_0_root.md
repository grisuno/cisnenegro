# root

*Community 0 | 37 files | cohesion 1.00*

## Definition

This community groups 37 file(s) rooted at `root` with dominant language py (cohesion 1.00). Central symbols: `AdaptiveTopologyController`, `AdvancedEvolutionEngine`, `ApexEvolutionEngine`, `ApexTrainer`, `BlackMirrorMonitor`, `CoarseCIFAR100`, `CurriculumTrainingCycle`, `DualTrainer`. Core file: `apex28.py` (47 symbols). Documented purpose: NeuroSovereign v14.0: Hierarchical Apex (FIXED) Objective: Prove Structural Necessity on CIFAR-100 (Hierarchical Task). Features: 1. Dataset Upgrade: CIFAR-10 -.

## Files

### `.` (37 files)

| File | Language | Layer | Symbols | Doc |
|------|----------|-------|---------|-----|
| `apex14.py` | py | utility | 26 | yes |
| `apex15.py` | py | utility | 22 | yes |
| `apex16.py` | py | utility | 22 | yes |
| `apex17.py` | py | utility | 26 | yes |
| `apex18.py` | py | utility | 26 | yes |
| `apex19.py` | py | utility | 27 | yes |
| `apex20.py` | py | utility | 28 | yes |
| `apex21.py` | py | utility | 32 | yes |
| `apex22.py` | py | utility | 31 | yes |
| `apex23.py` | py | utility | 32 | yes |
| `apex24.py` | py | utility | 32 | yes |
| `apex25.py` | py | utility | 32 | yes |
| `apex26.py` | py | utility | 42 | yes |
| `apex27.py` | py | utility | 40 | yes |
| `apex28.py` | py | utility | 47 | yes |
| `apex29.py` | py | utility | 47 | yes |
| `apex30.py` | py | utility | 47 | yes |
| `apex31.py` | py | utility | 47 | yes |
| `apex32.py` | py | utility | 28 | yes |
| `apex33.py` | py | utility | 46 | yes |

*... and 17 more files in this community.*


## Key Symbols

- `GatedTokenMixer` (class, `apex14.py:36`) `class GatedTokenMixer(Module)`
- `__init__` (method, `apex14.py:37`) `def __init__(self, num_tokens, embed_dim)`
- `forward` (method, `apex14.py:54`) `def forward(self, x)`
- `PatchFeatureExtractor` (class, `apex14.py:66`) `class PatchFeatureExtractor(Module)` - Extractor configurable para CIFAR-100.
- `__init__` (method, `apex14.py:72`) `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)`
- `freeze` (method, `apex14.py:89`) `def freeze(self)`
- `unfreeze` (method, `apex14.py:93`) `def unfreeze(self)`
- `forward` (method, `apex14.py:97`) `def forward(self, x)`
- `LotteryMLP` (class, `apex14.py:111`) `class LotteryMLP(Module)`
- `__init__` (method, `apex14.py:112`) `def __init__(self, input_dim, hidden_dim, num_classes)`
- `apply_masks` (method, `apex14.py:123`) `def apply_masks(self)`
- `get_sparsity` (method, `apex14.py:128`) `def get_sparsity(self)`
- `forward` (method, `apex14.py:133`) `def forward(self, x)`
- `SpectralMonitor` (class, `apex14.py:142`) `class SpectralMonitor`
- `compute_metrics` (method, `apex14.py:143`) `def compute_metrics(self, weight)`
- `OrthogonalEvolutionEngine` (class, `apex14.py:158`) `class OrthogonalEvolutionEngine`
- `__init__` (method, `apex14.py:159`) `def __init__(self, device)`
- `_gradient_nudge_inheritance` (method, `apex14.py:164`) `def _gradient_nudge_inheritance(self, child_model, elk_state, data_loader, featu`
- `_apply_rank_capping_shock` (method, `apex14.py:198`) `def _apply_rank_capping_shock(self, model, layer_name, keep_ratio)`
- `create_refined_offspring` (method, `apex14.py:217`) `def create_refined_offspring(self, elk_state, data_loader, feature_extractor)`
- `HierarchicalTrainer` (class, `apex14.py:231`) `class HierarchicalTrainer`
- `__init__` (method, `apex14.py:232`) `def __init__(self, device, extractor_apex, extractor_blind)`
- `_preprocess_batch` (method, `apex14.py:240`) `def _preprocess_batch(self, x, extractor)`
- `get_curriculum_dataset` (method, `apex14.py:244`) `def get_curriculum_dataset(self, cycle)`
- `train_single_chain` (method, `apex14.py:250`) `def train_single_chain(self, model, cycle, chain_type)`
- `main` (method, `apex14.py:336`) `def main()`
- `compute_spectral_loss` (function, `apex15.py:63`) `def compute_spectral_loss(W, target_rank_factor)` - Penaliza la desalineación entre Entropía Espectral y Rango Efectivo.
- `GatedTokenMixer` (class, `apex15.py:88`) `class GatedTokenMixer(Module)`
- `__init__` (method, `apex15.py:89`) `def __init__(self, num_patches, embed_dim)`
- `forward` (method, `apex15.py:98`) `def forward(self, x)`

## Internal vs External Edges

- Internal resolved imports (EXTRACTED): 0
- Cross-boundary resolved imports (EXTRACTED): 0

## Connections

- No cross-community bridges recorded. This community is self-contained.

## Risks

- No scoped security, taint, cycle, or layer risks.

## Open Questions

- Why do 2 file(s) lack file-level docs (e.g. `install.sh`)? What purpose do they serve?
- What would break if the most connected file in root changed?
- Should root be split, given cohesion 1.00?

## Sources

- `apex14.py`
- `apex15.py`
- `apex16.py`
- `apex17.py`
- `apex18.py`
- `apex19.py`
- `apex20.py`
- `apex21.py`
- `apex22.py`
- `apex23.py`
- `apex24.py`
- `apex25.py`
- `apex26.py`
- `apex27.py`
- `apex28.py`
- `apex29.py`
- `apex30.py`
- `apex31.py`
- `apex32.py`
- `apex33.py`
- *... and 17 more*
