# Symbols (page 1 of 3)
Pages: [SYMBOLS.md](SYMBOLS.md), [SYMBOLS_p2.md](SYMBOLS_p2.md), [SYMBOLS_p3.md](SYMBOLS_p3.md)

| Symbol | Kind | File:Line | Signature |
|--------|------|-----------|-----------|
| `GatedTokenMixer` | class | `apex14.py:36` | `class GatedTokenMixer(Module)` |
| `HierarchicalTrainer` | class | `apex14.py:231` | `class HierarchicalTrainer` |
| `LotteryMLP` | class | `apex14.py:111` | `class LotteryMLP(Module)` |
| `OrthogonalEvolutionEngine` | class | `apex14.py:158` | `class OrthogonalEvolutionEngine` |
| `PatchFeatureExtractor` | class | `apex14.py:66` | `class PatchFeatureExtractor(Module)` |
| `SpectralMonitor` | class | `apex14.py:142` | `class SpectralMonitor` |
| `__init__` | method | `apex14.py:37` | `def __init__(self, num_tokens, embed_dim)` |
| `__init__` | method | `apex14.py:72` | `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)` |
| `__init__` | method | `apex14.py:112` | `def __init__(self, input_dim, hidden_dim, num_classes)` |
| `__init__` | method | `apex14.py:159` | `def __init__(self, device)` |
| `__init__` | method | `apex14.py:232` | `def __init__(self, device, extractor_apex, extractor_blind)` |
| `_apply_rank_capping_shock` | method | `apex14.py:198` | `def _apply_rank_capping_shock(self, model, layer_name, keep_ratio)` |
| `_gradient_nudge_inheritance` | method | `apex14.py:164` | `def _gradient_nudge_inheritance(self, child_model, elk_state, data_loader, feature_extractor, nudge_lr)` |
| `_preprocess_batch` | method | `apex14.py:240` | `def _preprocess_batch(self, x, extractor)` |
| `apply_masks` | method | `apex14.py:123` | `def apply_masks(self)` |
| `compute_metrics` | method | `apex14.py:143` | `def compute_metrics(self, weight)` |
| `create_refined_offspring` | method | `apex14.py:217` | `def create_refined_offspring(self, elk_state, data_loader, feature_extractor)` |
| `forward` | method | `apex14.py:54` | `def forward(self, x)` |
| `forward` | method | `apex14.py:97` | `def forward(self, x)` |
| `forward` | method | `apex14.py:133` | `def forward(self, x)` |
| `freeze` | method | `apex14.py:89` | `def freeze(self)` |
| `get_curriculum_dataset` | method | `apex14.py:244` | `def get_curriculum_dataset(self, cycle)` |
| `get_sparsity` | method | `apex14.py:128` | `def get_sparsity(self)` |
| `main` | method | `apex14.py:336` | `def main()` |
| `train_single_chain` | method | `apex14.py:250` | `def train_single_chain(self, model, cycle, chain_type)` |
| `unfreeze` | method | `apex14.py:93` | `def unfreeze(self)` |
| `GatedTokenMixer` | class | `apex15.py:88` | `class GatedTokenMixer(Module)` |
| `PatchFeatureExtractor` | class | `apex15.py:105` | `class PatchFeatureExtractor(Module)` |
| `SpectralMonitor` | class | `apex15.py:179` | `class SpectralMonitor` |
| `TaxonomicMLP` | class | `apex15.py:143` | `class TaxonomicMLP(Module)` |
| `TaxonomicTrainer` | class | `apex15.py:195` | `class TaxonomicTrainer` |
| `__init__` | method | `apex15.py:89` | `def __init__(self, num_patches, embed_dim)` |
| `__init__` | method | `apex15.py:106` | `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)` |
| `__init__` | method | `apex15.py:144` | `def __init__(self, input_dim, hidden_dim, num_classes)` |
| `__init__` | method | `apex15.py:196` | `def __init__(self, device, extractor_apex, extractor_blind)` |
| `_preprocess_batch` | method | `apex15.py:203` | `def _preprocess_batch(self, x, extractor)` |
| `apply_masks` | method | `apex15.py:158` | `def apply_masks(self)` |
| `compute_metrics` | method | `apex15.py:180` | `def compute_metrics(self, weight)` |
| `compute_spectral_loss` | function | `apex15.py:63` | `def compute_spectral_loss(W, target_rank_factor)` |
| `forward` | method | `apex15.py:98` | `def forward(self, x)` |
| `forward` | method | `apex15.py:131` | `def forward(self, x)` |
| `forward` | method | `apex15.py:169` | `def forward(self, x)` |
| `freeze` | method | `apex15.py:121` | `def freeze(self)` |
| `get_curriculum_dataset` | method | `apex15.py:207` | `def get_curriculum_dataset(self, cycle)` |
| `get_sparsity` | method | `apex15.py:164` | `def get_sparsity(self)` |
| `main` | method | `apex15.py:367` | `def main()` |
| `train_single_chain` | method | `apex15.py:213` | `def train_single_chain(self, model, cycle, chain_type)` |
| `unfreeze_mixer_only` | method | `apex15.py:125` | `def unfreeze_mixer_only(self)` |
| `GatedTokenMixer` | class | `apex16.py:82` | `class GatedTokenMixer(Module)` |
| `PatchFeatureExtractor` | class | `apex16.py:99` | `class PatchFeatureExtractor(Module)` |
| `SpectralMonitor` | class | `apex16.py:173` | `class SpectralMonitor` |
| `TaxonomicMLP` | class | `apex16.py:137` | `class TaxonomicMLP(Module)` |
| `TaxonomicTrainer` | class | `apex16.py:189` | `class TaxonomicTrainer` |
| `__init__` | method | `apex16.py:83` | `def __init__(self, num_patches, embed_dim)` |
| `__init__` | method | `apex16.py:100` | `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)` |
| `__init__` | method | `apex16.py:138` | `def __init__(self, input_dim, hidden_dim, num_classes)` |
| `__init__` | method | `apex16.py:190` | `def __init__(self, device, extractor_apex, extractor_blind)` |
| `_preprocess_batch` | method | `apex16.py:197` | `def _preprocess_batch(self, x, extractor)` |
| `apply_masks` | method | `apex16.py:152` | `def apply_masks(self)` |
| `compute_metrics` | method | `apex16.py:174` | `def compute_metrics(self, weight)` |
| `compute_spectral_loss` | function | `apex16.py:63` | `def compute_spectral_loss(W, target_rank_factor)` |
| `forward` | method | `apex16.py:92` | `def forward(self, x)` |
| `forward` | method | `apex16.py:125` | `def forward(self, x)` |
| `forward` | method | `apex16.py:163` | `def forward(self, x)` |
| `freeze` | method | `apex16.py:115` | `def freeze(self)` |
| `get_curriculum_dataset` | method | `apex16.py:201` | `def get_curriculum_dataset(self, cycle)` |
| `get_sparsity` | method | `apex16.py:158` | `def get_sparsity(self)` |
| `main` | method | `apex16.py:377` | `def main()` |
| `train_single_chain` | method | `apex16.py:207` | `def train_single_chain(self, model, cycle, chain_type)` |
| `unfreeze_mixer_only` | method | `apex16.py:119` | `def unfreeze_mixer_only(self)` |
| `CoarseCIFAR100` | class | `apex17.py:368` | `class CoarseCIFAR100(CIFAR100)` |
| `GatedTokenMixer` | class | `apex17.py:81` | `class GatedTokenMixer(Module)` |
| `PatchFeatureExtractor` | class | `apex17.py:98` | `class PatchFeatureExtractor(Module)` |
| `SpectralMonitor` | class | `apex17.py:172` | `class SpectralMonitor` |
| `TaxonomicMLP` | class | `apex17.py:136` | `class TaxonomicMLP(Module)` |
| `TaxonomicTrainer` | class | `apex17.py:190` | `class TaxonomicTrainer` |
| `__getitem__` | method | `apex17.py:373` | `def __getitem__(self, index)` |
| `__init__` | method | `apex17.py:82` | `def __init__(self, num_patches, embed_dim)` |
| `__init__` | method | `apex17.py:99` | `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)` |
| `__init__` | method | `apex17.py:137` | `def __init__(self, input_dim, hidden_dim, num_classes)` |
| `__init__` | method | `apex17.py:191` | `def __init__(self, device, extractor_apex, extractor_blind)` |
| `_preprocess_batch` | method | `apex17.py:198` | `def _preprocess_batch(self, x, extractor)` |
| `apply_masks` | method | `apex17.py:151` | `def apply_masks(self)` |
| `compute_metrics` | method | `apex17.py:173` | `def compute_metrics(self, weight)` |
| `compute_spectral_loss` | function | `apex17.py:65` | `def compute_spectral_loss(W)` |
| `evaluate` | method | `apex17.py:391` | `def evaluate(model, extractor, name)` |
| `forward` | method | `apex17.py:91` | `def forward(self, x)` |
| `forward` | method | `apex17.py:124` | `def forward(self, x)` |
| `forward` | method | `apex17.py:162` | `def forward(self, x)` |
| `freeze` | method | `apex17.py:114` | `def freeze(self)` |
| `get_curriculum_dataset` | method | `apex17.py:202` | `def get_curriculum_dataset(self, cycle)` |
| `get_sparsity` | method | `apex17.py:157` | `def get_sparsity(self)` |
| `main` | method | `apex17.py:428` | `def main()` |
| `run_hierarchy_benchmark` | method | `apex17.py:378` | `def run_hierarchy_benchmark(model_apex, model_blind, device)` |
| `train_single_chain` | method | `apex17.py:208` | `def train_single_chain(self, model, cycle, chain_type)` |
| `unfreeze_mixer_only` | method | `apex17.py:118` | `def unfreeze_mixer_only(self)` |
| `CoarseCIFAR100` | class | `apex18.py:369` | `class CoarseCIFAR100(CIFAR100)` |
| `GatedTokenMixer` | class | `apex18.py:81` | `class GatedTokenMixer(Module)` |
| `PatchFeatureExtractor` | class | `apex18.py:98` | `class PatchFeatureExtractor(Module)` |
| `SpectralMonitor` | class | `apex18.py:172` | `class SpectralMonitor` |
| `TaxonomicMLP` | class | `apex18.py:136` | `class TaxonomicMLP(Module)` |
| `TaxonomicTrainer` | class | `apex18.py:188` | `class TaxonomicTrainer` |
| `__getitem__` | method | `apex18.py:370` | `def __getitem__(self, index)` |
| `__init__` | method | `apex18.py:82` | `def __init__(self, num_patches, embed_dim)` |
| `__init__` | method | `apex18.py:99` | `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)` |
| `__init__` | method | `apex18.py:137` | `def __init__(self, input_dim, hidden_dim, num_classes)` |
| `__init__` | method | `apex18.py:189` | `def __init__(self, device, extractor_apex, extractor_blind)` |
| `_preprocess_batch` | method | `apex18.py:196` | `def _preprocess_batch(self, x, extractor)` |
| `apply_masks` | method | `apex18.py:151` | `def apply_masks(self)` |
| `compute_metrics` | method | `apex18.py:173` | `def compute_metrics(self, weight)` |
| `compute_spectral_loss` | function | `apex18.py:65` | `def compute_spectral_loss(W)` |
| `evaluate` | method | `apex18.py:386` | `def evaluate(model, extractor, name)` |
| `forward` | method | `apex18.py:91` | `def forward(self, x)` |
| `forward` | method | `apex18.py:124` | `def forward(self, x)` |
| `forward` | method | `apex18.py:162` | `def forward(self, x)` |
| `freeze` | method | `apex18.py:114` | `def freeze(self)` |
| `get_curriculum_dataset` | method | `apex18.py:200` | `def get_curriculum_dataset(self, cycle)` |
| `get_sparsity` | method | `apex18.py:157` | `def get_sparsity(self)` |
| `main` | method | `apex18.py:420` | `def main()` |
| `run_hierarchy_benchmark` | method | `apex18.py:374` | `def run_hierarchy_benchmark(model_apex, model_blind, device)` |
| `train_single_chain` | method | `apex18.py:206` | `def train_single_chain(self, model, cycle, chain_type)` |
| `unfreeze_mixer_only` | method | `apex18.py:118` | `def unfreeze_mixer_only(self)` |
| `CoarseCIFAR100` | class | `apex19.py:394` | `class CoarseCIFAR100(CIFAR100)` |
| `GatedTokenMixer` | class | `apex19.py:84` | `class GatedTokenMixer(Module)` |
| `PatchFeatureExtractor` | class | `apex19.py:101` | `class PatchFeatureExtractor(Module)` |
| `SpectralMonitor` | class | `apex19.py:175` | `class SpectralMonitor` |
| `TaxonomicMLP` | class | `apex19.py:139` | `class TaxonomicMLP(Module)` |
| `TaxonomicTrainer` | class | `apex19.py:211` | `class TaxonomicTrainer` |
| `__getitem__` | method | `apex19.py:395` | `def __getitem__(self, index)` |
| `__init__` | method | `apex19.py:85` | `def __init__(self, num_patches, embed_dim)` |
| `__init__` | method | `apex19.py:102` | `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)` |
| `__init__` | method | `apex19.py:140` | `def __init__(self, input_dim, hidden_dim, num_classes)` |
| `__init__` | method | `apex19.py:212` | `def __init__(self, device, extractor_apex, extractor_blind)` |
| `_preprocess_batch` | method | `apex19.py:219` | `def _preprocess_batch(self, x, extractor)` |
| `apply_masks` | method | `apex19.py:154` | `def apply_masks(self)` |
| `compute_metrics` | method | `apex19.py:176` | `def compute_metrics(self, weight)` |
| `compute_spectral_loss` | function | `apex19.py:68` | `def compute_spectral_loss(W)` |
| `compute_topology_ratio` | method | `apex19.py:188` | `def compute_topology_ratio(self, model, extractor, chain_type)` |
| `evaluate` | method | `apex19.py:411` | `def evaluate(model, extractor, name)` |
| `forward` | method | `apex19.py:94` | `def forward(self, x)` |
| `forward` | method | `apex19.py:127` | `def forward(self, x)` |
| `forward` | method | `apex19.py:165` | `def forward(self, x)` |
| `freeze` | method | `apex19.py:117` | `def freeze(self)` |
| `get_curriculum_dataset` | method | `apex19.py:223` | `def get_curriculum_dataset(self, cycle)` |
| `get_sparsity` | method | `apex19.py:160` | `def get_sparsity(self)` |
| `main` | method | `apex19.py:445` | `def main()` |
| `run_hierarchy_benchmark` | method | `apex19.py:399` | `def run_hierarchy_benchmark(model_apex, model_blind, device)` |
| `train_single_chain` | method | `apex19.py:229` | `def train_single_chain(self, model, cycle, chain_type)` |
| `unfreeze_mixer_only` | method | `apex19.py:121` | `def unfreeze_mixer_only(self)` |
| `CoarseCIFAR100` | class | `apex20.py:426` | `class CoarseCIFAR100(CIFAR100)` |
| `GatedTokenMixer` | class | `apex20.py:85` | `class GatedTokenMixer(Module)` |
| `PatchFeatureExtractor` | class | `apex20.py:102` | `class PatchFeatureExtractor(Module)` |
| `SpectralMonitor` | class | `apex20.py:176` | `class SpectralMonitor` |
| `TaxonomicMLP` | class | `apex20.py:140` | `class TaxonomicMLP(Module)` |
| `TaxonomicTrainer` | class | `apex20.py:242` | `class TaxonomicTrainer` |
| `__getitem__` | method | `apex20.py:427` | `def __getitem__(self, index)` |
| `__init__` | method | `apex20.py:86` | `def __init__(self, num_patches, embed_dim)` |
| `__init__` | method | `apex20.py:103` | `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)` |
| `__init__` | method | `apex20.py:141` | `def __init__(self, input_dim, hidden_dim, num_classes)` |
| `__init__` | method | `apex20.py:243` | `def __init__(self, device, extractor_apex, extractor_blind)` |
| `_preprocess_batch` | method | `apex20.py:250` | `def _preprocess_batch(self, x, extractor)` |
| `apply_masks` | method | `apex20.py:155` | `def apply_masks(self)` |
| `compute_metrics` | method | `apex20.py:177` | `def compute_metrics(self, weight)` |
| `compute_spectral_loss` | function | `apex20.py:69` | `def compute_spectral_loss(W)` |
| `compute_topology_ratio` | method | `apex20.py:216` | `def compute_topology_ratio(self, model, extractor, chain_type)` |
| `detect_phase_state` | method | `apex20.py:190` | `def detect_phase_state(self, ratio_history)` |
| `evaluate` | method | `apex20.py:443` | `def evaluate(model, extractor, name)` |
| `forward` | method | `apex20.py:95` | `def forward(self, x)` |
| `forward` | method | `apex20.py:128` | `def forward(self, x)` |
| `forward` | method | `apex20.py:166` | `def forward(self, x)` |
| `freeze` | method | `apex20.py:118` | `def freeze(self)` |
| `get_curriculum_dataset` | method | `apex20.py:254` | `def get_curriculum_dataset(self, cycle)` |
| `get_sparsity` | method | `apex20.py:161` | `def get_sparsity(self)` |
| `main` | method | `apex20.py:490` | `def main()` |
| `run_hierarchy_benchmark` | method | `apex20.py:431` | `def run_hierarchy_benchmark(model_apex, model_blind, device)` |
| `train_single_chain` | method | `apex20.py:260` | `def train_single_chain(self, model, cycle, chain_type)` |
| `unfreeze_mixer_only` | method | `apex20.py:122` | `def unfreeze_mixer_only(self)` |
| `CoarseCIFAR100` | class | `apex21.py:459` | `class CoarseCIFAR100(CIFAR100)` |
| `GatedTokenMixer` | class | `apex21.py:88` | `class GatedTokenMixer(Module)` |
| `PatchFeatureExtractor` | class | `apex21.py:105` | `class PatchFeatureExtractor(Module)` |
| `SpectralMonitor` | class | `apex21.py:179` | `class SpectralMonitor` |
| `TaxonomicMLP` | class | `apex21.py:143` | `class TaxonomicMLP(Module)` |
| `TaxonomicTrainer` | class | `apex21.py:270` | `class TaxonomicTrainer` |
| `TopologyController` | class | `apex21.py:228` | `class TopologyController` |
| `__getitem__` | method | `apex21.py:460` | `def __getitem__(self, index)` |
| `__init__` | method | `apex21.py:89` | `def __init__(self, num_patches, embed_dim)` |
| `__init__` | method | `apex21.py:106` | `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)` |
| `__init__` | method | `apex21.py:144` | `def __init__(self, input_dim, hidden_dim, num_classes)` |
| `__init__` | method | `apex21.py:230` | `def __init__(self)` |
| `__init__` | method | `apex21.py:271` | `def __init__(self, device, extractor_apex, extractor_blind)` |
| `_preprocess_batch` | method | `apex21.py:279` | `def _preprocess_batch(self, x, extractor)` |
| `apply_masks` | method | `apex21.py:158` | `def apply_masks(self)` |
| `check_intervention` | method | `apex21.py:233` | `def check_intervention(self, phase_state, coarse_acc, extractor)` |
| `compute_metrics` | method | `apex21.py:180` | `def compute_metrics(self, weight)` |
| `compute_spectral_loss` | function | `apex21.py:73` | `def compute_spectral_loss(W)` |
| `compute_topology_ratio` | method | `apex21.py:204` | `def compute_topology_ratio(self, model, extractor, chain_type)` |
| `detect_phase_state` | method | `apex21.py:192` | `def detect_phase_state(self, ratio_history)` |
| `evaluate` | method | `apex21.py:476` | `def evaluate(model, extractor, name)` |
| `forward` | method | `apex21.py:98` | `def forward(self, x)` |
| `forward` | method | `apex21.py:131` | `def forward(self, x)` |
| `forward` | method | `apex21.py:169` | `def forward(self, x)` |
| `freeze` | method | `apex21.py:121` | `def freeze(self)` |
| `get_curriculum_dataset` | method | `apex21.py:283` | `def get_curriculum_dataset(self, cycle)` |
| `get_sparsity` | method | `apex21.py:164` | `def get_sparsity(self)` |
| `main` | method | `apex21.py:513` | `def main()` |
| `perturb_mixer` | method | `apex21.py:255` | `def perturb_mixer(self, extractor)` |
| `run_hierarchy_benchmark` | method | `apex21.py:464` | `def run_hierarchy_benchmark(model_apex, model_blind, device)` |
| `train_single_chain` | method | `apex21.py:289` | `def train_single_chain(self, model, cycle, chain_type)` |
| `unfreeze_mixer_only` | method | `apex21.py:125` | `def unfreeze_mixer_only(self)` |
| `CoarseCIFAR100` | class | `apex22.py:459` | `class CoarseCIFAR100(CIFAR100)` |
| `GatedTokenMixer` | class | `apex22.py:88` | `class GatedTokenMixer(Module)` |
| `PatchFeatureExtractor` | class | `apex22.py:105` | `class PatchFeatureExtractor(Module)` |
| `SpectralMonitor` | class | `apex22.py:179` | `class SpectralMonitor` |
| `TaxonomicMLP` | class | `apex22.py:143` | `class TaxonomicMLP(Module)` |
| `TaxonomicTrainer` | class | `apex22.py:273` | `class TaxonomicTrainer` |
| `TopologyController` | class | `apex22.py:204` | `class TopologyController` |
| `__getitem__` | method | `apex22.py:460` | `def __getitem__(self, index)` |
| `__init__` | method | `apex22.py:89` | `def __init__(self, num_patches, embed_dim)` |
| `__init__` | method | `apex22.py:106` | `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)` |
| `__init__` | method | `apex22.py:144` | `def __init__(self, input_dim, hidden_dim, num_classes)` |
| `__init__` | method | `apex22.py:206` | `def __init__(self)` |
| `__init__` | method | `apex22.py:274` | `def __init__(self, device, extractor_apex, extractor_blind)` |
| `_preprocess_batch` | method | `apex22.py:282` | `def _preprocess_batch(self, x, extractor)` |
| `apply_masks` | method | `apex22.py:158` | `def apply_masks(self)` |
| `check_intervention` | method | `apex22.py:209` | `def check_intervention(self, phase_state, coarse_acc, extractor)` |
| `compute_metrics` | method | `apex22.py:180` | `def compute_metrics(self, weight)` |
| `compute_spectral_loss` | function | `apex22.py:73` | `def compute_spectral_loss(W)` |
| `detect_phase_state` | method | `apex22.py:192` | `def detect_phase_state(self, ratio_history)` |
| `evaluate` | method | `apex22.py:476` | `def evaluate(model, extractor, name)` |
| `forward` | method | `apex22.py:98` | `def forward(self, x)` |
| `forward` | method | `apex22.py:131` | `def forward(self, x)` |
| `forward` | method | `apex22.py:169` | `def forward(self, x)` |
| `freeze` | method | `apex22.py:121` | `def freeze(self)` |
| `get_curriculum_dataset` | method | `apex22.py:286` | `def get_curriculum_dataset(self, cycle)` |
| `get_sparsity` | method | `apex22.py:164` | `def get_sparsity(self)` |
| `main` | method | `apex22.py:513` | `def main()` |
| `perturb_mixer_targeted` | method | `apex22.py:226` | `def perturb_mixer_targeted(self, extractor)` |
| `run_hierarchy_benchmark` | method | `apex22.py:464` | `def run_hierarchy_benchmark(model_apex, model_blind, device)` |
| `train_single_chain` | method | `apex22.py:292` | `def train_single_chain(self, model, cycle, chain_type)` |
| `unfreeze_mixer_only` | method | `apex22.py:125` | `def unfreeze_mixer_only(self)` |
| `CoarseCIFAR100` | class | `apex23.py:493` | `class CoarseCIFAR100(CIFAR100)` |
| `GatedTokenMixer` | class | `apex23.py:91` | `class GatedTokenMixer(Module)` |
| `PatchFeatureExtractor` | class | `apex23.py:108` | `class PatchFeatureExtractor(Module)` |
| `SpectralMonitor` | class | `apex23.py:182` | `class SpectralMonitor` |
| `TaxonomicMLP` | class | `apex23.py:146` | `class TaxonomicMLP(Module)` |
| `TaxonomicTrainer` | class | `apex23.py:306` | `class TaxonomicTrainer` |
| `TopologyController` | class | `apex23.py:228` | `class TopologyController` |
| `__getitem__` | method | `apex23.py:494` | `def __getitem__(self, index)` |
| `__init__` | method | `apex23.py:92` | `def __init__(self, num_patches, embed_dim)` |
| `__init__` | method | `apex23.py:109` | `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)` |
| `__init__` | method | `apex23.py:147` | `def __init__(self, input_dim, hidden_dim, num_classes)` |
| `__init__` | method | `apex23.py:229` | `def __init__(self)` |
| `__init__` | method | `apex23.py:307` | `def __init__(self, device, extractor_apex, extractor_blind)` |
| `_preprocess_batch` | method | `apex23.py:315` | `def _preprocess_batch(self, x, extractor)` |
| `apply_masks` | method | `apex23.py:161` | `def apply_masks(self)` |
| `check_intervention` | method | `apex23.py:232` | `def check_intervention(self, phase_state, coarse_acc, extractor)` |
| `compute_metrics` | method | `apex23.py:183` | `def compute_metrics(self, weight)` |
| `compute_spectral_loss` | function | `apex23.py:74` | `def compute_spectral_loss(W)` |
| `compute_topology_ratio` | method | `apex23.py:208` | `def compute_topology_ratio(self, model, extractor, chain_type)` |
| `detect_phase_state` | method | `apex23.py:196` | `def detect_phase_state(self, ratio_history)` |
| `evaluate` | method | `apex23.py:510` | `def evaluate(model, extractor, name)` |
| `forward` | method | `apex23.py:101` | `def forward(self, x)` |
| `forward` | method | `apex23.py:134` | `def forward(self, x)` |
| `forward` | method | `apex23.py:172` | `def forward(self, x)` |
| `freeze` | method | `apex23.py:124` | `def freeze(self)` |
| `get_curriculum_dataset` | method | `apex23.py:319` | `def get_curriculum_dataset(self, cycle)` |
| `get_sparsity` | method | `apex23.py:167` | `def get_sparsity(self)` |
| `main` | method | `apex23.py:544` | `def main()` |
| `perturb_mixer_targeted` | method | `apex23.py:250` | `def perturb_mixer_targeted(self, extractor)` |
| `run_hierarchy_benchmark` | method | `apex23.py:498` | `def run_hierarchy_benchmark(model_apex, model_blind, device)` |
| `train_single_chain` | method | `apex23.py:325` | `def train_single_chain(self, model, cycle, chain_type)` |
| `unfreeze_mixer_only` | method | `apex23.py:128` | `def unfreeze_mixer_only(self)` |
| `CoarseCIFAR100` | class | `apex24.py:527` | `class CoarseCIFAR100(CIFAR100)` |
| `GatedTokenMixer` | class | `apex24.py:92` | `class GatedTokenMixer(Module)` |
| `PatchFeatureExtractor` | class | `apex24.py:109` | `class PatchFeatureExtractor(Module)` |
| `SpectralMonitor` | class | `apex24.py:183` | `class SpectralMonitor` |
| `TaxonomicMLP` | class | `apex24.py:147` | `class TaxonomicMLP(Module)` |
| `TaxonomicTrainer` | class | `apex24.py:339` | `class TaxonomicTrainer` |
| `TopologyController` | class | `apex24.py:229` | `class TopologyController` |
| `__getitem__` | method | `apex24.py:528` | `def __getitem__(self, index)` |
| `__init__` | method | `apex24.py:93` | `def __init__(self, num_patches, embed_dim)` |
| `__init__` | method | `apex24.py:110` | `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)` |
| `__init__` | method | `apex24.py:148` | `def __init__(self, input_dim, hidden_dim, num_classes)` |
| `__init__` | method | `apex24.py:230` | `def __init__(self)` |
| `__init__` | method | `apex24.py:340` | `def __init__(self, device, extractor_apex, extractor_blind)` |
| `_preprocess_batch` | method | `apex24.py:348` | `def _preprocess_batch(self, x, extractor)` |
| `apply_masks` | method | `apex24.py:162` | `def apply_masks(self)` |
| `check_intervention` | method | `apex24.py:235` | `def check_intervention(self, phase_state, coarse_acc, extractor, current_topo_r)` |
| `compute_metrics` | method | `apex24.py:184` | `def compute_metrics(self, weight)` |
| `compute_spectral_loss` | function | `apex24.py:75` | `def compute_spectral_loss(W)` |
| `compute_topology_ratio` | method | `apex24.py:209` | `def compute_topology_ratio(self, model, extractor, chain_type)` |
| `detect_phase_state` | method | `apex24.py:197` | `def detect_phase_state(self, ratio_history)` |
| `evaluate` | method | `apex24.py:544` | `def evaluate(model, extractor, name)` |
| `forward` | method | `apex24.py:102` | `def forward(self, x)` |
| `forward` | method | `apex24.py:135` | `def forward(self, x)` |
| `forward` | method | `apex24.py:173` | `def forward(self, x)` |
| `freeze` | method | `apex24.py:125` | `def freeze(self)` |
| `get_curriculum_dataset` | method | `apex24.py:352` | `def get_curriculum_dataset(self, cycle)` |
| `get_sparsity` | method | `apex24.py:168` | `def get_sparsity(self)` |
| `main` | method | `apex24.py:578` | `def main()` |
| `perturb_mixer_targeted` | method | `apex24.py:284` | `def perturb_mixer_targeted(self, extractor)` |
| `run_hierarchy_benchmark` | method | `apex24.py:532` | `def run_hierarchy_benchmark(model_apex, model_blind, device)` |
| `train_single_chain` | method | `apex24.py:358` | `def train_single_chain(self, model, cycle, chain_type)` |
| `unfreeze_mixer_only` | method | `apex24.py:129` | `def unfreeze_mixer_only(self)` |
| `CoarseCIFAR100` | class | `apex25.py:504` | `class CoarseCIFAR100(CIFAR100)` |
| `GatedTokenMixer` | class | `apex25.py:89` | `class GatedTokenMixer(Module)` |
| `PatchFeatureExtractor` | class | `apex25.py:106` | `class PatchFeatureExtractor(Module)` |
| `SpectralMonitor` | class | `apex25.py:180` | `class SpectralMonitor` |
| `TaxonomicMLP` | class | `apex25.py:144` | `class TaxonomicMLP(Module)` |
| `TaxonomicTrainer` | class | `apex25.py:316` | `class TaxonomicTrainer` |
| `TopologyController` | class | `apex25.py:224` | `class TopologyController` |
| `__getitem__` | method | `apex25.py:505` | `def __getitem__(self, index)` |
| `__init__` | method | `apex25.py:90` | `def __init__(self, num_patches, embed_dim)` |
| `__init__` | method | `apex25.py:107` | `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)` |
| `__init__` | method | `apex25.py:145` | `def __init__(self, input_dim, hidden_dim, num_classes)` |
| `__init__` | method | `apex25.py:225` | `def __init__(self)` |
| `__init__` | method | `apex25.py:317` | `def __init__(self, device, extractor_apex, extractor_blind)` |
| `_preprocess_batch` | method | `apex25.py:325` | `def _preprocess_batch(self, x, extractor)` |
| `apply_masks` | method | `apex25.py:159` | `def apply_masks(self)` |
| `check_intervention` | method | `apex25.py:230` | `def check_intervention(self, phase_state, coarse_acc, extractor, current_topo_r)` |
| `compute_metrics` | method | `apex25.py:181` | `def compute_metrics(self, weight)` |
| `compute_spectral_loss` | function | `apex25.py:75` | `def compute_spectral_loss(W)` |
| `compute_topology_ratio` | method | `apex25.py:206` | `def compute_topology_ratio(self, model, extractor, chain_type)` |
| `detect_phase_state` | method | `apex25.py:194` | `def detect_phase_state(self, ratio_history)` |
| `evaluate` | method | `apex25.py:521` | `def evaluate(model, extractor, name)` |
| `forward` | method | `apex25.py:99` | `def forward(self, x)` |
| `forward` | method | `apex25.py:132` | `def forward(self, x)` |
| `forward` | method | `apex25.py:170` | `def forward(self, x)` |
| `freeze` | method | `apex25.py:122` | `def freeze(self)` |
| `get_curriculum_dataset` | method | `apex25.py:329` | `def get_curriculum_dataset(self, cycle)` |
| `get_sparsity` | method | `apex25.py:165` | `def get_sparsity(self)` |
| `main` | method | `apex25.py:555` | `def main()` |
| `perturb_mixer_targeted` | method | `apex25.py:277` | `def perturb_mixer_targeted(self, extractor)` |
| `run_hierarchy_benchmark` | method | `apex25.py:509` | `def run_hierarchy_benchmark(model_apex, model_blind, device)` |
| `train_single_chain` | method | `apex25.py:335` | `def train_single_chain(self, model, cycle, chain_type)` |
| `unfreeze_mixer_only` | method | `apex25.py:126` | `def unfreeze_mixer_only(self)` |
| `CoarseCIFAR100` | class | `apex26.py:480` | `class CoarseCIFAR100(CIFAR100)` |
| `EvolutionaryEngine` | class | `apex26.py:410` | `class EvolutionaryEngine` |
| `EvolutionaryTrainer` | class | `apex26.py:544` | `class EvolutionaryTrainer` |
| `GatedTokenMixer` | class | `apex26.py:89` | `class GatedTokenMixer(Module)` |
| `PatchFeatureExtractor` | class | `apex26.py:146` | `class PatchFeatureExtractor(Module)` |
| `SpectralMonitor` | class | `apex26.py:281` | `class SpectralMonitor` |
| `TaxonomicMLP` | class | `apex26.py:207` | `class TaxonomicMLP(Module)` |
| `TopologyController` | class | `apex26.py:302` | `class TopologyController` |
| `__getitem__` | method | `apex26.py:485` | `def __getitem__(self, index)` |
| `__init__` | method | `apex26.py:91` | `def __init__(self, num_patches, embed_dim)` |
| `__init__` | method | `apex26.py:148` | `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)` |
| `__init__` | method | `apex26.py:209` | `def __init__(self, input_dim, hidden_dim, num_classes, num_superclasses)` |
| `__init__` | method | `apex26.py:283` | `def __init__(self, epsilon)` |
| `__init__` | method | `apex26.py:304` | `def __init__(self, target_coarse_v, stagnation_limit, mixer_noise_scale, dominant_energy_threshold)` |
| `__init__` | method | `apex26.py:412` | `def __init__(self, device, target_L)` |
| `__init__` | method | `apex26.py:546` | `def __init__(self, device, output_dir)` |
| `_init_weights` | method | `apex26.py:114` | `def _init_weights(self)` |
| `_plot_results` | method | `apex26.py:1081` | `def _plot_results(self, all_results)` |
| `_save_results` | method | `apex26.py:1027` | `def _save_results(self, all_results, best_overall, hierarchy_delta)` |
| `apply_masks` | method | `apex26.py:232` | `def apply_masks(self)` |
| `apply_rank_capping` | method | `apex26.py:417` | `def apply_rank_capping(self, model, layer_name, keep_ratio)` |
| `check_intervention` | method | `apex26.py:332` | `def check_intervention(self, phase_state, coarse_acc, extractor, current_topo_r, geo_window)` |
| `compute_metrics` | method | `apex26.py:286` | `def compute_metrics(self, weight)` |
| `compute_spectral_loss` | method | `apex26.py:265` | `def compute_spectral_loss(W)` |
| `compute_topology_ratio` | method | `apex26.py:837` | `def compute_topology_ratio(self, model, extractor, chain_type)` |
| `create_offspring` | method | `apex26.py:434` | `def create_offspring(self, parent_state, data_loader, feature_extractor, lambda_taxonomic, learning_rate)` |
| `detect_phase_state` | method | `apex26.py:314` | `def detect_phase_state(self, ratio_history, phase_window, phase_std_dev_limit)` |
| `evaluate` | method | `apex26.py:503` | `def evaluate(model, extractor, name)` |
| `forward` | method | `apex26.py:128` | `def forward(self, x)` |
| `forward` | method | `apex26.py:190` | `def forward(self, x)` |
| `forward` | method | `apex26.py:245` | `def forward(self, x)` |
| `freeze` | method | `apex26.py:177` | `def freeze(self)` |
| `get_sparsity` | method | `apex26.py:239` | `def get_sparsity(self)` |
| `load_data` | method | `apex26.py:572` | `def load_data(self, cycle, batch_size)` |
| `main` | method | `apex26.py:1181` | `def main()` |
| `parse_args` | method | `apex26.py:1170` | `def parse_args()` |
| `perturb_mixer_targeted` | method | `apex26.py:377` | `def perturb_mixer_targeted(self, extractor)` |
| `run_evolution` | method | `apex26.py:855` | `def run_evolution(self, num_iterations, num_seeds, early_stop_patience)` |
| `run_hierarchy_benchmark` | method | `apex26.py:490` | `def run_hierarchy_benchmark(model_apex, model_blind, feature_extractor_apex, feature_extractor_blind, device)` |
| `set_seed` | function | `apex26.py:43` | `def set_seed(seed)` |
| `train_model` | method | `apex26.py:614` | `def train_model(self, model, cycle, chain_type, feature_extractor)` |
| `unfreeze_mixer_only` | method | `apex26.py:183` | `def unfreeze_mixer_only(self)` |
| `DynamicThresholdController` | class | `apex27.py:233` | `class DynamicThresholdController` |
| `GatedTokenMixer` | class | `apex27.py:83` | `class GatedTokenMixer(Module)` |
| `IterativeRefinementEngine` | class | `apex27.py:369` | `class IterativeRefinementEngine` |
| `IterativeTrainer` | class | `apex27.py:416` | `class IterativeTrainer` |
| `PatchFeatureExtractor` | class | `apex27.py:124` | `class PatchFeatureExtractor(Module)` |
| `SpectralMonitor` | class | `apex27.py:255` | `class SpectralMonitor` |
| `TaxonomicMLP` | class | `apex27.py:170` | `class TaxonomicMLP(Module)` |
| `TopologyController` | class | `apex27.py:275` | `class TopologyController` |
| `__init__` | method | `apex27.py:85` | `def __init__(self, num_patches, embed_dim)` |
| `__init__` | method | `apex27.py:126` | `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)` |
| `__init__` | method | `apex27.py:172` | `def __init__(self, input_dim, hidden_dim, num_classes, num_superclasses)` |
| `__init__` | method | `apex27.py:238` | `def __init__(self, window_size, percentile_trigger)` |
| `__init__` | method | `apex27.py:257` | `def __init__(self, epsilon)` |
| `__init__` | method | `apex27.py:280` | `def __init__(self, dynamic_threshold, stagnation_limit, mixer_noise_scale, dominant_energy_threshold, enable_surgery)` |
| `__init__` | method | `apex27.py:371` | `def __init__(self, device)` |
| `__init__` | method | `apex27.py:418` | `def __init__(self, device, output_dir, enable_surgery, enable_taxonomy)` |
| `_init_weights` | method | `apex27.py:104` | `def _init_weights(self)` |
| `_plot_results_v18` | method | `apex27.py:815` | `def _plot_results_v18(self, all_results)` |
| `_save_results` | method | `apex27.py:809` | `def _save_results(self, all_results)` |
| `apply_masks` | method | `apex27.py:192` | `def apply_masks(self)` |
| `check_intervention` | method | `apex27.py:291` | `def check_intervention(self, coarse_acc, extractor, current_topo_r, geo_window, alpha)` |
| `compute_metrics` | method | `apex27.py:260` | `def compute_metrics(self, weight)` |
| `compute_spectral_loss` | method | `apex27.py:217` | `def compute_spectral_loss(W)` |
| `compute_topology_ratio` | method | `apex27.py:683` | `def compute_topology_ratio(self, model, extractor, chain_type)` |
| `create_offspring` | method | `apex27.py:374` | `def create_offspring(self, parent_state, data_loader, feature_extractor, lambda_taxonomic, learning_rate)` |
| `forward` | method | `apex27.py:116` | `def forward(self, x)` |
| `forward` | method | `apex27.py:162` | `def forward(self, x)` |
| `forward` | method | `apex27.py:203` | `def forward(self, x)` |
| `freeze` | method | `apex27.py:151` | `def freeze(self)` |
| `get_sparsity` | method | `apex27.py:198` | `def get_sparsity(self)` |
| `is_stagnant` | method | `apex27.py:246` | `def is_stagnant(self, current_val)` |
| `load_data` | method | `apex27.py:441` | `def load_data(self, cycle, batch_size)` |
| `main` | method | `apex27.py:914` | `def main()` |
| `parse_args` | method | `apex27.py:900` | `def parse_args()` |
| `perturb_mixer_targeted` | method | `apex27.py:336` | `def perturb_mixer_targeted(self, extractor)` |
| `run_refinement` | method | `apex27.py:696` | `def run_refinement(self, num_iterations, num_seeds, early_stop_patience)` |
| `set_seed` | function | `apex27.py:38` | `def set_seed(seed)` |
| `train_model` | method | `apex27.py:479` | `def train_model(self, model, cycle, chain_type, feature_extractor)` |
| `unfreeze_mixer_only` | method | `apex27.py:156` | `def unfreeze_mixer_only(self)` |
| `update` | method | `apex27.py:243` | `def update(self, value)` |
| `AdaptiveTopologyController` | class | `apex28.py:316` | `class AdaptiveTopologyController` |
| `CoarseCIFAR100` | class | `apex28.py:484` | `class CoarseCIFAR100(CIFAR100)` |
| `GatedTokenMixer` | class | `apex28.py:93` | `class GatedTokenMixer(Module)` |
| `IterativeRefinementEngine` | class | `apex28.py:415` | `class IterativeRefinementEngine` |
| `IterativeRefinementTrainer` | class | `apex28.py:611` | `class IterativeRefinementTrainer` |
| `PatchFeatureExtractor` | class | `apex28.py:150` | `class PatchFeatureExtractor(Module)` |
| `SpectralMonitor` | class | `apex28.py:285` | `class SpectralMonitor` |
| `TaxonomicMLP` | class | `apex28.py:211` | `class TaxonomicMLP(Module)` |
| `__getitem__` | method | `apex28.py:489` | `def __getitem__(self, index)` |
| `__init__` | method | `apex28.py:95` | `def __init__(self, num_patches, embed_dim)` |
| `__init__` | method | `apex28.py:152` | `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)` |
| `__init__` | method | `apex28.py:213` | `def __init__(self, input_dim, hidden_dim, num_classes, num_superclasses)` |
| `__init__` | method | `apex28.py:287` | `def __init__(self, epsilon)` |
| `__init__` | method | `apex28.py:318` | `def __init__(self, semantic_plasticity_threshold, stagnation_limit, mixer_noise_scale, dominant_energy_threshold...` |
| `__init__` | method | `apex28.py:417` | `def __init__(self, device)` |
| `__init__` | method | `apex28.py:613` | `def __init__(self, device, output_dir)` |
| `_init_weights` | method | `apex28.py:118` | `def _init_weights(self)` |
| `_plot_results` | method | `apex28.py:1181` | `def _plot_results(self, all_results)` |
| `_save_results` | method | `apex28.py:1126` | `def _save_results(self, all_results, best_overall, hierarchy_delta)` |
| `apply_masks` | method | `apex28.py:236` | `def apply_masks(self)` |
| `apply_rank_capping` | method | `apex28.py:421` | `def apply_rank_capping(self, model, layer_name, keep_ratio)` |
| `compute_metrics` | method | `apex28.py:290` | `def compute_metrics(self, weight)` |
| `compute_semantic_plasticity_ratio` | method | `apex28.py:330` | `def compute_semantic_plasticity_ratio(self)` |
| `compute_spectral_loss` | method | `apex28.py:269` | `def compute_spectral_loss(W)` |
| `compute_topology_ratio` | method | `apex28.py:926` | `def compute_topology_ratio(self, model, extractor, chain_type)` |
| `create_refined_model` | method | `apex28.py:438` | `def create_refined_model(self, parent_state, data_loader, feature_extractor, lambda_taxonomic, learning_rate)` |
| `detect_intervention_need` | method | `apex28.py:343` | `def detect_intervention_need(self, phase_state, extractor)` |
| `detect_phase_state` | method | `apex28.py:908` | `def detect_phase_state(self, ratio_history, phase_window, phase_std_dev_limit)` |
| `evaluate` | method | `apex28.py:507` | `def evaluate(model, extractor, name)` |
| `forward` | method | `apex28.py:132` | `def forward(self, x)` |
| `forward` | method | `apex28.py:194` | `def forward(self, x)` |
| `forward` | method | `apex28.py:249` | `def forward(self, x)` |
| `freeze` | method | `apex28.py:181` | `def freeze(self)` |
| `get_singular_values` | method | `apex28.py:306` | `def get_singular_values(self, weight)` |
| `get_singular_values` | method | `apex28.py:556` | `def get_singular_values(model, extractor, name)` |
| `get_sparsity` | method | `apex28.py:243` | `def get_sparsity(self)` |
| `load_data` | method | `apex28.py:639` | `def load_data(self, cycle, batch_size)` |
| `main` | method | `apex28.py:1282` | `def main()` |
| `parse_args` | method | `apex28.py:1270` | `def parse_args()` |
| `perturb_mixer_targeted` | method | `apex28.py:382` | `def perturb_mixer_targeted(self, extractor)` |
| `run_hierarchy_benchmark` | method | `apex28.py:494` | `def run_hierarchy_benchmark(model_apex, model_blind, feature_extractor_apex, feature_extractor_blind, device)` |
| `run_refinement` | method | `apex28.py:944` | `def run_refinement(self, num_iterations, num_seeds, early_stop_patience)` |
| `set_seed` | function | `apex28.py:47` | `def set_seed(seed)` |
| `train_model` | method | `apex28.py:681` | `def train_model(self, model, cycle, chain_type, feature_extractor)` |
| `unfreeze_mixer_only` | method | `apex28.py:187` | `def unfreeze_mixer_only(self)` |
| `update_history` | method | `apex28.py:365` | `def update_history(self, topo_ratio, coarse_acc)` |
| `visualize_singular_values` | method | `apex28.py:548` | `def visualize_singular_values(model_apex, model_blind, feature_extractor_apex, feature_extractor_blind, device...` |
| `AdaptiveTopologyController` | class | `apex29.py:309` | `class AdaptiveTopologyController` |
| `CoarseCIFAR100` | class | `apex29.py:476` | `class CoarseCIFAR100(CIFAR100)` |
| `GatedTokenMixer` | class | `apex29.py:86` | `class GatedTokenMixer(Module)` |
| `IterativeRefinementEngine` | class | `apex29.py:408` | `class IterativeRefinementEngine` |
| `IterativeRefinementTrainer` | class | `apex29.py:606` | `class IterativeRefinementTrainer` |
| `PatchFeatureExtractor` | class | `apex29.py:143` | `class PatchFeatureExtractor(Module)` |
| `SpectralMonitor` | class | `apex29.py:278` | `class SpectralMonitor` |
| `TaxonomicMLP` | class | `apex29.py:204` | `class TaxonomicMLP(Module)` |
| `__getitem__` | method | `apex29.py:481` | `def __getitem__(self, index)` |
| `__init__` | method | `apex29.py:88` | `def __init__(self, num_patches, embed_dim)` |
| `__init__` | method | `apex29.py:145` | `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)` |
| `__init__` | method | `apex29.py:206` | `def __init__(self, input_dim, hidden_dim, num_classes, num_superclasses)` |
| `__init__` | method | `apex29.py:280` | `def __init__(self, epsilon)` |
| `__init__` | method | `apex29.py:311` | `def __init__(self, semantic_plasticity_threshold, stagnation_limit, mixer_noise_scale, dominant_energy_threshold...` |
| `__init__` | method | `apex29.py:410` | `def __init__(self, device)` |
| `__init__` | method | `apex29.py:608` | `def __init__(self, device, output_dir)` |
| `_init_weights` | method | `apex29.py:111` | `def _init_weights(self)` |
| `_plot_results` | method | `apex29.py:1164` | `def _plot_results(self, all_results)` |
| `_save_results` | method | `apex29.py:1109` | `def _save_results(self, all_results, best_overall, hierarchy_delta)` |
| `apply_masks` | method | `apex29.py:229` | `def apply_masks(self)` |
| `apply_rank_capping` | method | `apex29.py:414` | `def apply_rank_capping(self, model, layer_name, keep_ratio)` |
| `compute_metrics` | method | `apex29.py:283` | `def compute_metrics(self, weight)` |
| `compute_semantic_plasticity_ratio` | method | `apex29.py:323` | `def compute_semantic_plasticity_ratio(self)` |
| `compute_spectral_loss` | method | `apex29.py:262` | `def compute_spectral_loss(W)` |
| `compute_topology_ratio` | method | `apex29.py:912` | `def compute_topology_ratio(self, model, extractor, chain_type)` |
| `create_refined_model` | method | `apex29.py:430` | `def create_refined_model(self, parent_state, data_loader, feature_extractor, lambda_taxonomic, learning_rate)` |
| `detect_intervention_need` | method | `apex29.py:336` | `def detect_intervention_need(self, phase_state, extractor)` |
| `detect_phase_state` | method | `apex29.py:894` | `def detect_phase_state(self, ratio_history, phase_window, phase_std_dev_limit)` |

Next: [SYMBOLS_p2.md](SYMBOLS_p2.md)
