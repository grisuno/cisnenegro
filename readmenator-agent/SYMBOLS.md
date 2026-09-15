# Symbols

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
| `__init__` | method | `apex28.py:318` | `def __init__(self, semantic_plasticity_threshold, stagnation_limit, mixer_noise_scale, dominant_energy_threshold, geo_wi` |
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
| `visualize_singular_values` | method | `apex28.py:548` | `def visualize_singular_values(model_apex, model_blind, feature_extractor_apex, feature_extractor_blind, device, output_d` |
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
| `__init__` | method | `apex29.py:311` | `def __init__(self, semantic_plasticity_threshold, stagnation_limit, mixer_noise_scale, dominant_energy_threshold, geo_wi` |
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
| `evaluate` | method | `apex29.py:499` | `def evaluate(model, extractor, name)` |
| `forward` | method | `apex29.py:125` | `def forward(self, x)` |
| `forward` | method | `apex29.py:187` | `def forward(self, x)` |
| `forward` | method | `apex29.py:242` | `def forward(self, x)` |
| `freeze` | method | `apex29.py:174` | `def freeze(self)` |
| `get_singular_values` | method | `apex29.py:299` | `def get_singular_values(self, weight)` |
| `get_singular_values` | method | `apex29.py:551` | `def get_singular_values(model, extractor, name)` |
| `get_sparsity` | method | `apex29.py:236` | `def get_sparsity(self)` |
| `load_data` | method | `apex29.py:634` | `def load_data(self, cycle, batch_size)` |
| `main` | method | `apex29.py:1255` | `def main()` |
| `parse_args` | method | `apex29.py:1244` | `def parse_args()` |
| `perturb_mixer_targeted` | method | `apex29.py:375` | `def perturb_mixer_targeted(self, extractor)` |
| `run_hierarchy_benchmark` | method | `apex29.py:486` | `def run_hierarchy_benchmark(model_apex, model_blind, feature_extractor_apex, feature_extractor_blind, device)` |
| `run_refinement` | method | `apex29.py:931` | `def run_refinement(self, num_iterations, num_seeds, early_stop_patience)` |
| `set_seed` | function | `apex29.py:43` | `def set_seed(seed)` |
| `train_model` | method | `apex29.py:674` | `def train_model(self, model, cycle, chain_type, feature_extractor)` |
| `unfreeze_mixer_only` | method | `apex29.py:180` | `def unfreeze_mixer_only(self)` |
| `update_history` | method | `apex29.py:358` | `def update_history(self, topo_ratio, coarse_acc)` |
| `visualize_singular_values` | method | `apex29.py:543` | `def visualize_singular_values(model_apex, model_blind, feature_extractor_apex, feature_extractor_blind, device, output_d` |
| `AdaptiveTopologyController` | class | `apex30.py:277` | `class AdaptiveTopologyController` |
| `CoarseCIFAR100` | class | `apex30.py:459` | `class CoarseCIFAR100(CIFAR100)` |
| `GatedTokenMixer` | class | `apex30.py:87` | `class GatedTokenMixer(Module)` |
| `IterativeRefinementEngine` | class | `apex30.py:397` | `class IterativeRefinementEngine` |
| `IterativeRefinementTrainer` | class | `apex30.py:570` | `class IterativeRefinementTrainer` |
| `PatchFeatureExtractor` | class | `apex30.py:142` | `class PatchFeatureExtractor(Module)` |
| `SpectralMonitor` | class | `apex30.py:248` | `class SpectralMonitor` |
| `TaxonomicMLP` | class | `apex30.py:185` | `class TaxonomicMLP(Module)` |
| `__getitem__` | method | `apex30.py:460` | `def __getitem__(self, index)` |
| `__init__` | method | `apex30.py:89` | `def __init__(self, num_patches, embed_dim)` |
| `__init__` | method | `apex30.py:143` | `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)` |
| `__init__` | method | `apex30.py:187` | `def __init__(self, input_dim, hidden_dim, num_classes, num_superclasses)` |
| `__init__` | method | `apex30.py:250` | `def __init__(self, epsilon)` |
| `__init__` | method | `apex30.py:279` | `def __init__(self, semantic_plasticity_threshold, stagnation_limit, mixer_noise_scale, dominant_energy_threshold, geo_wi` |
| `__init__` | method | `apex30.py:399` | `def __init__(self, device)` |
| `__init__` | method | `apex30.py:571` | `def __init__(self, device, output_dir)` |
| `_init_weights` | method | `apex30.py:111` | `def _init_weights(self)` |
| `_plot_results` | method | `apex30.py:983` | `def _plot_results(self, all_results, best_overall)` |
| `_save_results` | method | `apex30.py:956` | `def _save_results(self, all_results, best_overall, hierarchy_delta)` |
| `apply_masks` | method | `apex30.py:207` | `def apply_masks(self)` |
| `apply_rank_capping` | method | `apex30.py:403` | `def apply_rank_capping(self, model, layer_name, keep_ratio)` |
| `compute_metrics` | method | `apex30.py:253` | `def compute_metrics(self, weight)` |
| `compute_semantic_plasticity_ratio` | method | `apex30.py:295` | `def compute_semantic_plasticity_ratio(self)` |
| `compute_spectral_loss` | method | `apex30.py:232` | `def compute_spectral_loss(W)` |
| `compute_topology_ratio` | method | `apex30.py:813` | `def compute_topology_ratio(self, model, extractor, chain_type)` |
| `create_refined_model` | method | `apex30.py:419` | `def create_refined_model(self, parent_state, data_loader, feature_extractor, lambda_taxonomic, learning_rate)` |
| `detect_intervention_need` | method | `apex30.py:308` | `def detect_intervention_need(self, phase_state, extractor)` |
| `detect_phase_state` | method | `apex30.py:797` | `def detect_phase_state(self, ratio_history, phase_window, phase_std_dev_limit)` |
| `evaluate` | method | `apex30.py:471` | `def evaluate(model, extractor)` |
| `forward` | method | `apex30.py:134` | `def forward(self, x)` |
| `forward` | method | `apex30.py:175` | `def forward(self, x)` |
| `forward` | method | `apex30.py:218` | `def forward(self, x)` |
| `freeze` | method | `apex30.py:164` | `def freeze(self)` |
| `get_singular_values` | method | `apex30.py:268` | `def get_singular_values(self, weight)` |
| `get_singular_values` | method | `apex30.py:521` | `def get_singular_values(model, extractor)` |
| `get_sparsity` | method | `apex30.py:213` | `def get_sparsity(self)` |
| `load_data` | method | `apex30.py:593` | `def load_data(self, cycle, batch_size)` |
| `main` | method | `apex30.py:1071` | `def main()` |
| `parse_args` | method | `apex30.py:1062` | `def parse_args()` |
| `perturb_mixer_targeted` | method | `apex30.py:363` | `def perturb_mixer_targeted(self, extractor)` |
| `run_hierarchy_benchmark` | method | `apex30.py:464` | `def run_hierarchy_benchmark(model_apex, model_blind, feature_extractor_apex, feature_extractor_blind, device)` |
| `run_refinement` | method | `apex30.py:828` | `def run_refinement(self, num_iterations, num_seeds, early_stop_patience)` |
| `set_seed` | function | `apex30.py:43` | `def set_seed(seed)` |
| `train_model` | method | `apex30.py:617` | `def train_model(self, model, cycle, chain_type, feature_extractor)` |
| `unfreeze_mixer_only` | method | `apex30.py:169` | `def unfreeze_mixer_only(self)` |
| `update_history` | method | `apex30.py:348` | `def update_history(self, topo_ratio, coarse_acc)` |
| `visualize_singular_values` | method | `apex30.py:510` | `def visualize_singular_values(model_apex, model_blind, feature_extractor_apex, feature_extractor_blind, device, output_d` |
| `AdaptiveTopologyController` | class | `apex31.py:279` | `class AdaptiveTopologyController` |
| `CoarseCIFAR100` | class | `apex31.py:470` | `class CoarseCIFAR100(CIFAR100)` |
| `GatedTokenMixer` | class | `apex31.py:89` | `class GatedTokenMixer(Module)` |
| `IterativeRefinementEngine` | class | `apex31.py:404` | `class IterativeRefinementEngine` |
| `IterativeRefinementTrainer` | class | `apex31.py:581` | `class IterativeRefinementTrainer` |
| `PatchFeatureExtractor` | class | `apex31.py:144` | `class PatchFeatureExtractor(Module)` |
| `SpectralMonitor` | class | `apex31.py:250` | `class SpectralMonitor` |
| `TaxonomicMLP` | class | `apex31.py:187` | `class TaxonomicMLP(Module)` |
| `__getitem__` | method | `apex31.py:471` | `def __getitem__(self, index)` |
| `__init__` | method | `apex31.py:91` | `def __init__(self, num_patches, embed_dim)` |
| `__init__` | method | `apex31.py:145` | `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)` |
| `__init__` | method | `apex31.py:189` | `def __init__(self, input_dim, hidden_dim, num_classes, num_superclasses)` |
| `__init__` | method | `apex31.py:252` | `def __init__(self, epsilon)` |
| `__init__` | method | `apex31.py:281` | `def __init__(self, semantic_plasticity_threshold, stagnation_limit, mixer_noise_scale, dominant_energy_threshold, geo_wi` |
| `__init__` | method | `apex31.py:406` | `def __init__(self, device)` |
| `__init__` | method | `apex31.py:582` | `def __init__(self, device, output_dir)` |
| `_init_weights` | method | `apex31.py:113` | `def _init_weights(self)` |
| `_plot_results` | method | `apex31.py:994` | `def _plot_results(self, all_results, best_overall)` |
| `_save_results` | method | `apex31.py:967` | `def _save_results(self, all_results, best_overall, hierarchy_delta)` |
| `apply_masks` | method | `apex31.py:209` | `def apply_masks(self)` |
| `apply_rank_capping` | method | `apex31.py:410` | `def apply_rank_capping(self, model, layer_name, keep_ratio)` |
| `compute_metrics` | method | `apex31.py:255` | `def compute_metrics(self, weight)` |
| `compute_semantic_plasticity_ratio` | method | `apex31.py:297` | `def compute_semantic_plasticity_ratio(self)` |
| `compute_spectral_loss` | method | `apex31.py:234` | `def compute_spectral_loss(W)` |
| `compute_topology_ratio` | method | `apex31.py:824` | `def compute_topology_ratio(self, model, extractor, chain_type)` |
| `create_refined_model` | method | `apex31.py:430` | `def create_refined_model(self, parent_state, data_loader, feature_extractor, lambda_taxonomic, learning_rate)` |
| `detect_intervention_need` | method | `apex31.py:310` | `def detect_intervention_need(self, phase_state, extractor)` |
| `detect_phase_state` | method | `apex31.py:808` | `def detect_phase_state(self, ratio_history, phase_window, phase_std_dev_limit)` |
| `evaluate` | method | `apex31.py:482` | `def evaluate(model, extractor)` |
| `forward` | method | `apex31.py:136` | `def forward(self, x)` |
| `forward` | method | `apex31.py:177` | `def forward(self, x)` |
| `forward` | method | `apex31.py:220` | `def forward(self, x)` |
| `freeze` | method | `apex31.py:166` | `def freeze(self)` |
| `get_singular_values` | method | `apex31.py:270` | `def get_singular_values(self, weight)` |
| `get_singular_values` | method | `apex31.py:532` | `def get_singular_values(model, extractor)` |
| `get_sparsity` | method | `apex31.py:215` | `def get_sparsity(self)` |
| `load_data` | method | `apex31.py:604` | `def load_data(self, cycle, batch_size)` |
| `main` | method | `apex31.py:1082` | `def main()` |
| `parse_args` | method | `apex31.py:1073` | `def parse_args()` |
| `perturb_mixer_targeted` | method | `apex31.py:369` | `def perturb_mixer_targeted(self, extractor)` |
| `run_hierarchy_benchmark` | method | `apex31.py:475` | `def run_hierarchy_benchmark(model_apex, model_blind, feature_extractor_apex, feature_extractor_blind, device)` |
| `run_refinement` | method | `apex31.py:839` | `def run_refinement(self, num_iterations, num_seeds, early_stop_patience)` |
| `set_seed` | function | `apex31.py:45` | `def set_seed(seed)` |
| `train_model` | method | `apex31.py:628` | `def train_model(self, model, cycle, chain_type, feature_extractor)` |
| `unfreeze_mixer_only` | method | `apex31.py:171` | `def unfreeze_mixer_only(self)` |
| `update_history` | method | `apex31.py:354` | `def update_history(self, topo_ratio, coarse_acc)` |
| `visualize_singular_values` | method | `apex31.py:521` | `def visualize_singular_values(model_apex, model_blind, feature_extractor_apex, feature_extractor_blind, device, output_d` |
| `CoarseCIFAR100` | class | `apex32.py:404` | `class CoarseCIFAR100(CIFAR100)` |
| `GatedTokenMixer` | class | `apex32.py:92` | `class GatedTokenMixer(Module)` |
| `PatchFeatureExtractor` | class | `apex32.py:109` | `class PatchFeatureExtractor(Module)` |
| `SpectralMonitor` | class | `apex32.py:187` | `class SpectralMonitor` |
| `TaxonomicMLP` | class | `apex32.py:148` | `class TaxonomicMLP(Module)` |
| `TaxonomicTrainer` | class | `apex32.py:226` | `class TaxonomicTrainer` |
| `__getitem__` | method | `apex32.py:405` | `def __getitem__(self, index)` |
| `__init__` | method | `apex32.py:93` | `def __init__(self, num_patches, embed_dim)` |
| `__init__` | method | `apex32.py:110` | `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)` |
| `__init__` | method | `apex32.py:149` | `def __init__(self, input_dim, hidden_dim, num_classes)` |
| `__init__` | method | `apex32.py:227` | `def __init__(self, device, extractor_apex, extractor_blind)` |
| `_preprocess_batch` | method | `apex32.py:234` | `def _preprocess_batch(self, x, extractor)` |
| `apply_masks` | method | `apex32.py:164` | `def apply_masks(self)` |
| `compute_metrics` | method | `apex32.py:188` | `def compute_metrics(self, weight)` |
| `compute_spectral_loss` | function | `apex32.py:76` | `def compute_spectral_loss(W)` |
| `compute_topology_ratio` | method | `apex32.py:211` | `def compute_topology_ratio(self, model, extractor, chain_type)` |
| `detect_phase_state` | method | `apex32.py:200` | `def detect_phase_state(self, ratio_history)` |
| `evaluate` | method | `apex32.py:422` | `def evaluate(model, extractor, name)` |
| `forward` | method | `apex32.py:102` | `def forward(self, x)` |
| `forward` | method | `apex32.py:136` | `def forward(self, x)` |
| `forward` | method | `apex32.py:176` | `def forward(self, x)` |
| `freeze` | method | `apex32.py:126` | `def freeze(self)` |
| `get_curriculum_dataset` | method | `apex32.py:238` | `def get_curriculum_dataset(self, cycle)` |
| `get_sparsity` | method | `apex32.py:171` | `def get_sparsity(self)` |
| `main` | method | `apex32.py:455` | `def main()` |
| `run_hierarchy_benchmark` | method | `apex32.py:409` | `def run_hierarchy_benchmark(model_apex, model_blind, device)` |
| `train_single_chain` | method | `apex32.py:244` | `def train_single_chain(self, model, cycle, chain_type)` |
| `unfreeze_mixer_only` | method | `apex32.py:130` | `def unfreeze_mixer_only(self)` |
| `AdaptiveTopologyController` | class | `apex33.py:227` | `class AdaptiveTopologyController` |
| `CoarseCIFAR100` | class | `apex33.py:802` | `class CoarseCIFAR100(CIFAR100)` |
| `GatedTokenMixer` | class | `apex33.py:72` | `class GatedTokenMixer(Module)` |
| `IterativeRefinementEngine` | class | `apex33.py:310` | `class IterativeRefinementEngine` |
| `IterativeRefinementTrainer` | class | `apex33.py:343` | `class IterativeRefinementTrainer` |
| `PatchFeatureExtractor` | class | `apex33.py:118` | `class PatchFeatureExtractor(Module)` |
| `SpectralMonitor` | class | `apex33.py:201` | `class SpectralMonitor` |
| `TaxonomicMLP` | class | `apex33.py:150` | `class TaxonomicMLP(Module)` |
| `__getitem__` | method | `apex33.py:803` | `def __getitem__(self, index)` |
| `__init__` | method | `apex33.py:74` | `def __init__(self, num_patches, embed_dim)` |
| `__init__` | method | `apex33.py:119` | `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)` |
| `__init__` | method | `apex33.py:152` | `def __init__(self, input_dim, hidden_dim, num_classes, num_superclasses)` |
| `__init__` | method | `apex33.py:202` | `def __init__(self, epsilon)` |
| `__init__` | method | `apex33.py:229` | `def __init__(self, semantic_plasticity_threshold, stagnation_limit, mixer_noise_scale, dominant_energy_threshold, geo_wi` |
| `__init__` | method | `apex33.py:311` | `def __init__(self, device)` |
| `__init__` | method | `apex33.py:344` | `def __init__(self, device, output_dir)` |
| `_init_weights` | method | `apex33.py:92` | `def _init_weights(self)` |
| `_plot_results` | method | `apex33.py:738` | `def _plot_results(self, all_results, best_overall)` |
| `_save_results` | method | `apex33.py:712` | `def _save_results(self, all_results, best_overall, hierarchy_delta)` |
| `apply_masks` | method | `apex33.py:167` | `def apply_masks(self)` |
| `compute_metrics` | method | `apex33.py:205` | `def compute_metrics(self, weight)` |
| `compute_semantic_plasticity_ratio` | method | `apex33.py:242` | `def compute_semantic_plasticity_ratio(self)` |
| `compute_spectral_loss` | method | `apex33.py:190` | `def compute_spectral_loss(W)` |
| `compute_topology_ratio` | method | `apex33.py:580` | `def compute_topology_ratio(self, model, extractor, chain_type)` |
| `create_refined_model` | method | `apex33.py:315` | `def create_refined_model(self, parent_state, data_loader, feature_extractor, lambda_taxonomic, learning_rate)` |
| `detect_intervention_need` | method | `apex33.py:250` | `def detect_intervention_need(self, phase_state, extractor)` |
| `detect_phase_state` | method | `apex33.py:570` | `def detect_phase_state(self, ratio_history, phase_window, phase_std_dev_limit)` |
| `evaluate` | method | `apex33.py:814` | `def evaluate(model, extractor)` |
| `forward` | method | `apex33.py:110` | `def forward(self, x)` |
| `forward` | method | `apex33.py:143` | `def forward(self, x)` |
| `forward` | method | `apex33.py:178` | `def forward(self, x)` |
| `freeze` | method | `apex33.py:135` | `def freeze(self)` |
| `get_singular_values` | method | `apex33.py:219` | `def get_singular_values(self, weight)` |
| `get_singular_values` | method | `apex33.py:851` | `def get_singular_values(model, extractor)` |
| `get_sparsity` | method | `apex33.py:173` | `def get_sparsity(self)` |
| `load_data` | method | `apex33.py:367` | `def load_data(self, cycle, batch_size)` |
| `main` | method | `apex33.py:903` | `def main()` |
| `parse_args` | method | `apex33.py:894` | `def parse_args()` |
| `perturb_mixer_targeted` | method | `apex33.py:284` | `def perturb_mixer_targeted(self, extractor)` |
| `run_hierarchy_benchmark` | method | `apex33.py:807` | `def run_hierarchy_benchmark(model_apex, model_blind, feature_extractor_apex, feature_extractor_blind, device)` |
| `run_refinement` | method | `apex33.py:592` | `def run_refinement(self, num_iterations, num_seeds, early_stop_patience)` |
| `set_seed` | function | `apex33.py:41` | `def set_seed(seed)` |
| `train_model` | method | `apex33.py:385` | `def train_model(self, model, cycle, chain_type, feature_extractor)` |
| `unfreeze_mixer_only` | method | `apex33.py:138` | `def unfreeze_mixer_only(self)` |
| `update_history` | method | `apex33.py:273` | `def update_history(self, topo_ratio, coarse_acc)` |
| `visualize_singular_values` | method | `apex33.py:845` | `def visualize_singular_values(model_apex, model_blind, feature_extractor_apex, feature_extractor_blind, device, output_d` |
| `AdaptiveTopologyController` | class | `apex34.py:227` | `class AdaptiveTopologyController` |
| `CoarseCIFAR100` | class | `apex34.py:802` | `class CoarseCIFAR100(CIFAR100)` |
| `GatedTokenMixer` | class | `apex34.py:72` | `class GatedTokenMixer(Module)` |
| `IterativeRefinementEngine` | class | `apex34.py:310` | `class IterativeRefinementEngine` |
| `IterativeRefinementTrainer` | class | `apex34.py:343` | `class IterativeRefinementTrainer` |
| `PatchFeatureExtractor` | class | `apex34.py:118` | `class PatchFeatureExtractor(Module)` |
| `SpectralMonitor` | class | `apex34.py:201` | `class SpectralMonitor` |
| `TaxonomicMLP` | class | `apex34.py:150` | `class TaxonomicMLP(Module)` |
| `__getitem__` | method | `apex34.py:803` | `def __getitem__(self, index)` |
| `__init__` | method | `apex34.py:74` | `def __init__(self, num_patches, embed_dim)` |
| `__init__` | method | `apex34.py:119` | `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)` |
| `__init__` | method | `apex34.py:152` | `def __init__(self, input_dim, hidden_dim, num_classes, num_superclasses)` |
| `__init__` | method | `apex34.py:202` | `def __init__(self, epsilon)` |
| `__init__` | method | `apex34.py:229` | `def __init__(self, semantic_plasticity_threshold, stagnation_limit, mixer_noise_scale, dominant_energy_threshold, geo_wi` |
| `__init__` | method | `apex34.py:311` | `def __init__(self, device)` |
| `__init__` | method | `apex34.py:344` | `def __init__(self, device, output_dir)` |
| `_init_weights` | method | `apex34.py:92` | `def _init_weights(self)` |
| `_plot_results` | method | `apex34.py:738` | `def _plot_results(self, all_results, best_overall)` |
| `_save_results` | method | `apex34.py:712` | `def _save_results(self, all_results, best_overall, hierarchy_delta)` |
| `apply_masks` | method | `apex34.py:167` | `def apply_masks(self)` |
| `compute_metrics` | method | `apex34.py:205` | `def compute_metrics(self, weight)` |
| `compute_semantic_plasticity_ratio` | method | `apex34.py:242` | `def compute_semantic_plasticity_ratio(self)` |
| `compute_spectral_loss` | method | `apex34.py:190` | `def compute_spectral_loss(W)` |
| `compute_topology_ratio` | method | `apex34.py:580` | `def compute_topology_ratio(self, model, extractor, chain_type)` |
| `create_refined_model` | method | `apex34.py:315` | `def create_refined_model(self, parent_state, data_loader, feature_extractor, lambda_taxonomic, learning_rate)` |
| `detect_intervention_need` | method | `apex34.py:250` | `def detect_intervention_need(self, phase_state, extractor)` |
| `detect_phase_state` | method | `apex34.py:570` | `def detect_phase_state(self, ratio_history, phase_window, phase_std_dev_limit)` |
| `evaluate` | method | `apex34.py:814` | `def evaluate(model, extractor)` |
| `forward` | method | `apex34.py:110` | `def forward(self, x)` |
| `forward` | method | `apex34.py:143` | `def forward(self, x)` |
| `forward` | method | `apex34.py:178` | `def forward(self, x)` |
| `freeze` | method | `apex34.py:135` | `def freeze(self)` |
| `get_singular_values` | method | `apex34.py:219` | `def get_singular_values(self, weight)` |
| `get_singular_values` | method | `apex34.py:851` | `def get_singular_values(model, extractor)` |
| `get_sparsity` | method | `apex34.py:173` | `def get_sparsity(self)` |
| `load_data` | method | `apex34.py:367` | `def load_data(self, cycle, batch_size)` |
| `main` | method | `apex34.py:894` | `def main()` |
| `perturb_mixer_targeted` | method | `apex34.py:284` | `def perturb_mixer_targeted(self, extractor)` |
| `run_hierarchy_benchmark` | method | `apex34.py:807` | `def run_hierarchy_benchmark(model_apex, model_blind, feature_extractor_apex, feature_extractor_blind, device)` |
| `run_refinement` | method | `apex34.py:592` | `def run_refinement(self, num_iterations, num_seeds, early_stop_patience)` |
| `set_seed` | function | `apex34.py:41` | `def set_seed(seed)` |
| `train_model` | method | `apex34.py:385` | `def train_model(self, model, cycle, chain_type, feature_extractor)` |
| `unfreeze_mixer_only` | method | `apex34.py:138` | `def unfreeze_mixer_only(self)` |
| `update_history` | method | `apex34.py:273` | `def update_history(self, topo_ratio, coarse_acc)` |
| `visualize_singular_values` | method | `apex34.py:845` | `def visualize_singular_values(model_apex, model_blind, feature_extractor_apex, feature_extractor_blind, device, output_d` |
| `BlackMirrorMonitor` | class | `apex35.py:227` | `class BlackMirrorMonitor` |
| `CoarseCIFAR100` | class | `apex35.py:439` | `class CoarseCIFAR100(CIFAR100)` |
| `E8FusionLayer` | class | `apex35.py:112` | `class E8FusionLayer(Module)` |
| `GatedTokenMixer` | class | `apex35.py:66` | `class GatedTokenMixer(Module)` |
| `IterativeRefinementTrainer` | class | `apex35.py:253` | `class IterativeRefinementTrainer` |
| `PatchFeatureExtractor` | class | `apex35.py:156` | `class PatchFeatureExtractor(Module)` |
| `TaxonomicMLP` | class | `apex35.py:188` | `class TaxonomicMLP(Module)` |
| `__getitem__` | method | `apex35.py:440` | `def __getitem__(self, index)` |
| `__init__` | method | `apex35.py:68` | `def __init__(self, num_patches, embed_dim)` |
| `__init__` | method | `apex35.py:118` | `def __init__(self, embed_dim, num_heads)` |
| `__init__` | method | `apex35.py:157` | `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)` |
| `__init__` | method | `apex35.py:189` | `def __init__(self, input_dim, hidden_dim, num_classes, num_superclasses)` |
| `__init__` | method | `apex35.py:229` | `def __init__(self, epsilon)` |
| `__init__` | method | `apex35.py:254` | `def __init__(self, device, output_dir)` |
| `_init_weights` | method | `apex35.py:86` | `def _init_weights(self)` |
| `apply_masks` | method | `apex35.py:204` | `def apply_masks(self)` |
| `evaluate_safe` | method | `apex35.py:483` | `def evaluate_safe(model, extractor)` |
| `forward` | method | `apex35.py:104` | `def forward(self, x)` |
| `forward` | method | `apex35.py:136` | `def forward(self, x)` |
| `forward` | method | `apex35.py:181` | `def forward(self, x)` |
| `forward` | method | `apex35.py:215` | `def forward(self, x)` |
| `freeze` | method | `apex35.py:173` | `def freeze(self)` |
| `get_sparsity` | method | `apex35.py:210` | `def get_sparsity(self)` |
| `inspect` | method | `apex35.py:232` | `def inspect(self, weight)` |
| `load_data` | method | `apex35.py:274` | `def load_data(self, cycle, batch_size)` |
| `main` | method | `apex35.py:444` | `def main()` |
| `set_seed` | function | `apex35.py:36` | `def set_seed(seed)` |
| `train_model` | method | `apex35.py:292` | `def train_model(self, model, cycle, chain_type, feature_extractor)` |
| `unfreeze_mixer_only` | method | `apex35.py:176` | `def unfreeze_mixer_only(self)` |
| `EvolutionaryResonanceEngine` | class | `app.py:121` | `class EvolutionaryResonanceEngine` |
| `PersistentPruner` | class | `app.py:76` | `class PersistentPruner` |
| `SpectralMLP` | class | `app.py:98` | `class SpectralMLP(Module)` |
| `SpectralMonitor` | class | `app.py:51` | `class SpectralMonitor` |
| `__init__` | method | `app.py:52` | `def __init__(self, epsilon_c)` |
| `__init__` | method | `app.py:77` | `def __init__(self, sparsity_target)` |
| `__init__` | method | `app.py:99` | `def __init__(self, input_dim, hidden_dim, num_classes)` |
| `__init__` | method | `app.py:122` | `def __init__(self, device, base_target_acc)` |
| `apply_to_model` | method | `app.py:81` | `def apply_to_model(self, model)` |
| `compute_L` | method | `app.py:55` | `def compute_L(self, weight)` |
| `enforce_during_training` | method | `app.py:91` | `def enforce_during_training(self, model)` |
| `execute_resonance_cycle` | method | `app.py:351` | `def execute_resonance_cycle(self, cycle, test_loader, expansion_factor)` |
| `extract_seed_from_checkpoint` | method | `app.py:228` | `def extract_seed_from_checkpoint(self, checkpoint, expected_hidden_dim)` |
| `extract_seed_weights` | method | `app.py:254` | `def extract_seed_weights(self, model)` |
| `forward` | method | `app.py:113` | `def forward(self, x)` |
| `inoculate_seed_adaptive` | method | `app.py:260` | `def inoculate_seed_adaptive(self, large_model, seed_weights)` |
| `load_best_legacy_model` | method | `app.py:137` | `def load_best_legacy_model(self, cycle)` |
| `main` | method | `app.py:561` | `def main()` |
| `measure_functional_alignment` | method | `app.py:288` | `def measure_functional_alignment(self, model1, model2, test_loader)` |
| `progressive_pruning_with_target` | method | `app.py:309` | `def progressive_pruning_with_target(self, model, target_acc, test_loader, max_density)` |
| `reduce_input` | method | `app.py:107` | `def reduce_input(self, x)` |
| `run_evolutionary_experiment` | method | `app.py:484` | `def run_evolutionary_experiment(self, num_cycles)` |
| `train_base_model_to_target` | method | `app.py:175` | `def train_base_model_to_target(self, hidden_dim, target_acc, test_loader)` |
| `BlackMirrorMonitor` | class | `plank.py:28` | `class BlackMirrorMonitor` |
| `NeuroSovereign` | class | `plank.py:108` | `class NeuroSovereign(Module)` |
| `SovereignNeuron` | class | `plank.py:62` | `class SovereignNeuron(Module)` |
| `SovereignTrainer` | class | `plank.py:132` | `class SovereignTrainer` |
| `__init__` | method | `plank.py:33` | `def __init__(self, epsilon_c)` |
| `__init__` | method | `plank.py:63` | `def __init__(self, in_features, out_features, sparsity_target)` |
| `__init__` | method | `plank.py:109` | `def __init__(self, sparsity_target)` |
| `__init__` | method | `plank.py:133` | `def __init__(self, model, device)` |
| `apply_black_swan_refraction` | method | `plank.py:87` | `def apply_black_swan_refraction(self)` |
| `forward` | method | `plank.py:70` | `def forward(self, x, inject_lies)` |
| `forward` | method | `plank.py:117` | `def forward(self, x, inject_lies)` |
| `inspect` | method | `plank.py:36` | `def inspect(self, weights)` |
| `main` | method | `plank.py:178` | `def main()` |
| `train_epoch` | method | `plank.py:139` | `def train_epoch(self, dataloader, epoch)` |
| `ApexEvolutionEngine` | class | `plank10.py:148` | `class ApexEvolutionEngine` |
| `ApexTrainer` | class | `plank10.py:247` | `class ApexTrainer` |
| `LotteryMLP` | class | `plank10.py:91` | `class LotteryMLP(Module)` |
| `PatchFeatureExtractor` | class | `plank10.py:34` | `class PatchFeatureExtractor(Module)` |
| `SpectralMonitor` | class | `plank10.py:133` | `class SpectralMonitor` |
| `StandardBaseline` | class | `plank10.py:121` | `class StandardBaseline(Module)` |
| `__init__` | method | `plank10.py:39` | `def __init__(self, img_size, patch_size, in_chans, embed_dim)` |
| `__init__` | method | `plank10.py:92` | `def __init__(self, input_dim, hidden_dim, num_classes)` |
| `__init__` | method | `plank10.py:123` | `def __init__(self, input_dim, hidden_dim)` |
| `__init__` | method | `plank10.py:149` | `def __init__(self, device)` |
| `__init__` | method | `plank10.py:248` | `def __init__(self, device, feature_extractor)` |
| `_apply_dynamic_spectral_shock` | method | `plank10.py:188` | `def _apply_dynamic_spectral_shock(self, model, layer_name)` |
| `_gradient_nudge_inheritance` | method | `plank10.py:154` | `def _gradient_nudge_inheritance(self, child_model, elk_state, data_loader, feature_extractor, nudge_lr)` |
| `_preprocess_batch` | method | `plank10.py:255` | `def _preprocess_batch(self, x)` |
| `apply_masks` | method | `plank10.py:104` | `def apply_masks(self)` |
| `compute_L` | method | `plank10.py:134` | `def compute_L(self, weight)` |
| `create_apex_offspring` | method | `plank10.py:211` | `def create_apex_offspring(self, elk_state, new_hidden_dim, cycle, data_loader, feature_extractor, parent_gap)` |
| `forward` | method | `plank10.py:75` | `def forward(self, x)` |
| `forward` | method | `plank10.py:114` | `def forward(self, x)` |
| `forward` | method | `plank10.py:127` | `def forward(self, x)` |
| `freeze` | method | `plank10.py:67` | `def freeze(self)` |
| `get_curriculum_dataset` | method | `plank10.py:259` | `def get_curriculum_dataset(self, cycle)` |
| `get_sparsity` | method | `plank10.py:109` | `def get_sparsity(self)` |
| `main` | method | `plank10.py:354` | `def main()` |
| `train_model` | method | `plank10.py:265` | `def train_model(self, model, cycle, is_baseline)` |
| `unfreeze` | method | `plank10.py:71` | `def unfreeze(self)` |
| `LotteryMLP` | class | `plank11.py:97` | `class LotteryMLP(Module)` |
| `OrthogonalEvolutionEngine` | class | `plank11.py:152` | `class OrthogonalEvolutionEngine` |
| `OrthogonalTrainer` | class | `plank11.py:260` | `class OrthogonalTrainer` |
| `PatchFeatureExtractor` | class | `plank11.py:57` | `class PatchFeatureExtractor(Module)` |
| `SpectralMonitor` | class | `plank11.py:137` | `class SpectralMonitor` |
| `StandardBaseline` | class | `plank11.py:126` | `class StandardBaseline(Module)` |
| `TokenMixer` | class | `plank11.py:36` | `class TokenMixer(Module)` |
| `__init__` | method | `plank11.py:41` | `def __init__(self, num_tokens)` |
| `__init__` | method | `plank11.py:61` | `def __init__(self, img_size, patch_size, in_chans, embed_dim)` |
| `__init__` | method | `plank11.py:98` | `def __init__(self, input_dim, hidden_dim, num_classes)` |
| `__init__` | method | `plank11.py:127` | `def __init__(self, input_dim, hidden_dim)` |
| `__init__` | method | `plank11.py:153` | `def __init__(self, device)` |
| `__init__` | method | `plank11.py:261` | `def __init__(self, device, feature_extractor)` |
| `_apply_minimalistic_shock` | method | `plank11.py:190` | `def _apply_minimalistic_shock(self, model, layer_name, target_rank_ratio)` |
| `_gradient_nudge_inheritance` | method | `plank11.py:158` | `def _gradient_nudge_inheritance(self, child_model, elk_state, data_loader, feature_extractor, nudge_lr)` |
| `_preprocess_batch` | method | `plank11.py:268` | `def _preprocess_batch(self, x)` |
| `apply_masks` | method | `plank11.py:109` | `def apply_masks(self)` |
| `compute_L` | method | `plank11.py:138` | `def compute_L(self, weight)` |
| `create_orthogonal_offspring` | method | `plank11.py:223` | `def create_orthogonal_offspring(self, elk_state, new_hidden_dim, cycle, data_loader, feature_extractor, parent_gap)` |
| `forward` | method | `plank11.py:50` | `def forward(self, x)` |
| `forward` | method | `plank11.py:83` | `def forward(self, x)` |
| `forward` | method | `plank11.py:119` | `def forward(self, x)` |
| `forward` | method | `plank11.py:131` | `def forward(self, x)` |
| `freeze` | method | `plank11.py:75` | `def freeze(self)` |
| `get_curriculum_dataset` | method | `plank11.py:272` | `def get_curriculum_dataset(self, cycle)` |
| `get_sparsity` | method | `plank11.py:114` | `def get_sparsity(self)` |
| `main` | method | `plank11.py:379` | `def main()` |
| `train_model` | method | `plank11.py:278` | `def train_model(self, model, cycle, is_baseline)` |
| `unfreeze` | method | `plank11.py:79` | `def unfreeze(self)` |
| `LotteryMLP` | class | `plank12.py:87` | `class LotteryMLP(Module)` |
| `OrthogonalEvolutionEngine` | class | `plank12.py:154` | `class OrthogonalEvolutionEngine` |
| `OrthogonalTrainer` | class | `plank12.py:248` | `class OrthogonalTrainer` |
| `PatchFeatureExtractor` | class | `plank12.py:51` | `class PatchFeatureExtractor(Module)` |
| `SpectralMonitor` | class | `plank12.py:127` | `class SpectralMonitor` |
| `StandardBaseline` | class | `plank12.py:116` | `class StandardBaseline(Module)` |
| `TokenMixer` | class | `plank12.py:35` | `class TokenMixer(Module)` |
| `__init__` | method | `plank12.py:37` | `def __init__(self, num_tokens)` |
| `__init__` | method | `plank12.py:52` | `def __init__(self, img_size, patch_size, in_chans, embed_dim)` |
| `__init__` | method | `plank12.py:88` | `def __init__(self, input_dim, hidden_dim, num_classes)` |
| `__init__` | method | `plank12.py:117` | `def __init__(self, input_dim, hidden_dim)` |
| `__init__` | method | `plank12.py:155` | `def __init__(self, device)` |
| `__init__` | method | `plank12.py:249` | `def __init__(self, device, feature_extractor)` |
| `_apply_rank_capping_shock` | method | `plank12.py:192` | `def _apply_rank_capping_shock(self, model, layer_name, keep_ratio)` |
| `_gradient_nudge_inheritance` | method | `plank12.py:160` | `def _gradient_nudge_inheritance(self, child_model, elk_state, data_loader, feature_extractor, nudge_lr)` |
| `_preprocess_batch` | method | `plank12.py:256` | `def _preprocess_batch(self, x)` |
| `apply_masks` | method | `plank12.py:99` | `def apply_masks(self)` |
| `compute_metrics` | method | `plank12.py:128` | `def compute_metrics(self, weight)` |
| `create_refined_offspring` | method | `plank12.py:224` | `def create_refined_offspring(self, elk_state, data_loader, feature_extractor)` |
| `forward` | method | `plank12.py:44` | `def forward(self, x)` |
| `forward` | method | `plank12.py:74` | `def forward(self, x)` |
| `forward` | method | `plank12.py:109` | `def forward(self, x)` |
| `forward` | method | `plank12.py:121` | `def forward(self, x)` |
| `freeze` | method | `plank12.py:66` | `def freeze(self)` |
| `get_curriculum_dataset` | method | `plank12.py:260` | `def get_curriculum_dataset(self, cycle)` |
| `get_sparsity` | method | `plank12.py:104` | `def get_sparsity(self)` |
| `main` | method | `plank12.py:363` | `def main()` |
| `train_model` | method | `plank12.py:266` | `def train_model(self, model, cycle, is_baseline)` |
| `unfreeze` | method | `plank12.py:70` | `def unfreeze(self)` |
| `DualTrainer` | class | `plank13.py:224` | `class DualTrainer` |
| `LotteryMLP` | class | `plank13.py:92` | `class LotteryMLP(Module)` |
| `OrthogonalEvolutionEngine` | class | `plank13.py:149` | `class OrthogonalEvolutionEngine` |
| `PatchFeatureExtractor` | class | `plank13.py:47` | `class PatchFeatureExtractor(Module)` |
| `SpectralMonitor` | class | `plank13.py:123` | `class SpectralMonitor` |
| `TokenMixer` | class | `plank13.py:32` | `class TokenMixer(Module)` |
| `__init__` | method | `plank13.py:34` | `def __init__(self, num_tokens)` |
| `__init__` | method | `plank13.py:53` | `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)` |
| `__init__` | method | `plank13.py:93` | `def __init__(self, input_dim, hidden_dim, num_classes)` |
| `__init__` | method | `plank13.py:150` | `def __init__(self, device)` |
| `__init__` | method | `plank13.py:225` | `def __init__(self, device, extractor_apex, extractor_blind)` |
| `_apply_rank_capping_shock` | method | `plank13.py:187` | `def _apply_rank_capping_shock(self, model, layer_name, keep_ratio)` |
| `_gradient_nudge_inheritance` | method | `plank13.py:155` | `def _gradient_nudge_inheritance(self, child_model, elk_state, data_loader, feature_extractor, nudge_lr)` |
| `_preprocess_batch` | method | `plank13.py:233` | `def _preprocess_batch(self, x, extractor)` |
| `apply_masks` | method | `plank13.py:104` | `def apply_masks(self)` |
| `compute_metrics` | method | `plank13.py:124` | `def compute_metrics(self, weight)` |
| `create_refined_offspring` | method | `plank13.py:207` | `def create_refined_offspring(self, elk_state, data_loader, feature_extractor)` |
| `forward` | method | `plank13.py:41` | `def forward(self, x)` |
| `forward` | method | `plank13.py:78` | `def forward(self, x)` |
| `forward` | method | `plank13.py:114` | `def forward(self, x)` |
| `freeze` | method | `plank13.py:70` | `def freeze(self)` |
| `get_curriculum_dataset` | method | `plank13.py:237` | `def get_curriculum_dataset(self, cycle)` |
| `get_sparsity` | method | `plank13.py:109` | `def get_sparsity(self)` |
| `main` | method | `plank13.py:329` | `def main()` |
| `train_single_chain` | method | `plank13.py:243` | `def train_single_chain(self, model, cycle, chain_type)` |
| `unfreeze` | method | `plank13.py:74` | `def unfreeze(self)` |
| `PersistentPruner` | class | `plank2.py:82` | `class PersistentPruner` |
| `SpectralMLP` | class | `plank2.py:117` | `class SpectralMLP(Module)` |
| `SpectralMonitor` | class | `plank2.py:41` | `class SpectralMonitor` |
| `__init__` | method | `plank2.py:46` | `def __init__(self, epsilon_c)` |
| `__init__` | method | `plank2.py:87` | `def __init__(self, sparsity_target)` |
| `__init__` | method | `plank2.py:119` | `def __init__(self, input_dim, hidden_dim, num_classes)` |
| `apply_to_model` | method | `plank2.py:91` | `def apply_to_model(self, model)` |
| `compute_L` | method | `plank2.py:49` | `def compute_L(self, weight)` |
| `enforce_during_training` | method | `plank2.py:105` | `def enforce_during_training(self, model)` |
| `forward` | method | `plank2.py:135` | `def forward(self, x)` |
| `main` | method | `plank2.py:281` | `def main()` |
| `reduce_input` | method | `plank2.py:128` | `def reduce_input(self, x)` |
| `train_condition` | method | `plank2.py:173` | `def train_condition(condition_name, config, device, seed)` |
| `PersistentPruner` | class | `plank3.py:62` | `class PersistentPruner` |
| `SpectralMLP` | class | `plank3.py:87` | `class SpectralMLP(Module)` |
| `SpectralMonitor` | class | `plank3.py:34` | `class SpectralMonitor` |
| `__init__` | method | `plank3.py:35` | `def __init__(self, epsilon_c)` |
| `__init__` | method | `plank3.py:63` | `def __init__(self, sparsity_target)` |
| `__init__` | method | `plank3.py:88` | `def __init__(self, input_dim, hidden_dim, num_classes)` |
| `apply_to_model` | method | `plank3.py:67` | `def apply_to_model(self, model)` |
| `compute_L` | method | `plank3.py:38` | `def compute_L(self, weight)` |
| `enforce_during_training` | method | `plank3.py:77` | `def enforce_during_training(self, model)` |
| `find_critical_threshold` | method | `plank3.py:264` | `def find_critical_threshold(pruning_df, target_acc)` |
| `forward` | method | `plank3.py:102` | `def forward(self, x)` |
| `main` | method | `plank3.py:294` | `def main()` |
| `progressive_pruning_search` | method | `plank3.py:188` | `def progressive_pruning_search(model, device, target_acc)` |
| `reduce_input` | method | `plank3.py:96` | `def reduce_input(self, x)` |
| `train_dense_to_target` | method | `plank3.py:110` | `def train_dense_to_target(device, target_acc)` |
| `FractalSovereigntyEngine` | class | `plank4.py:111` | `class FractalSovereigntyEngine` |
| `PersistentPruner` | class | `plank4.py:66` | `class PersistentPruner` |
| `SpectralMLP` | class | `plank4.py:88` | `class SpectralMLP(Module)` |
| `SpectralMonitor` | class | `plank4.py:41` | `class SpectralMonitor` |
| `__init__` | method | `plank4.py:42` | `def __init__(self, epsilon_c)` |
| `__init__` | method | `plank4.py:67` | `def __init__(self, sparsity_target)` |
| `__init__` | method | `plank4.py:89` | `def __init__(self, input_dim, hidden_dim, num_classes)` |
| `__init__` | method | `plank4.py:112` | `def __init__(self, device, base_target_acc)` |
| `apply_to_model` | method | `plank4.py:71` | `def apply_to_model(self, model)` |
| `compute_L` | method | `plank4.py:45` | `def compute_L(self, weight)` |
| `enforce_during_training` | method | `plank4.py:81` | `def enforce_during_training(self, model)` |
| `execute_cycle` | method | `plank4.py:236` | `def execute_cycle(self, cycle, base_hidden_dim, expansion_factor)` |
| `extract_seed_weights` | method | `plank4.py:168` | `def extract_seed_weights(self, model)` |
| `forward` | method | `plank4.py:103` | `def forward(self, x)` |
| `inoculate_seed` | method | `plank4.py:174` | `def inoculate_seed(self, large_model, seed_weights)` |
| `main` | method | `plank4.py:329` | `def main()` |
| `progressive_pruning` | method | `plank4.py:192` | `def progressive_pruning(self, model, target_acc)` |
| `reduce_input` | method | `plank4.py:97` | `def reduce_input(self, x)` |
| `run_experiment` | method | `plank4.py:281` | `def run_experiment(self, num_cycles)` |
| `train_dense_model` | method | `plank4.py:123` | `def train_dense_model(self, hidden_dim, target_acc)` |
| `EvolutionCycle` | class | `plank5.py:211` | `class EvolutionCycle` |
| `EvolutionaryBlackSwanChain` | class | `plank5.py:392` | `class EvolutionaryBlackSwanChain` |
| `GrokkingDetector` | class | `plank5.py:88` | `class GrokkingDetector` |
| `PersistentPruner` | class | `plank5.py:50` | `class PersistentPruner` |
| `SpectralMLP` | class | `plank5.py:65` | `class SpectralMLP(Module)` |
| `SpectralMonitor` | class | `plank5.py:30` | `class SpectralMonitor` |
| `SyntheticBlackSwanGenerator` | class | `plank5.py:121` | `class SyntheticBlackSwanGenerator` |
| `__init__` | method | `plank5.py:31` | `def __init__(self, epsilon_c)` |
| `__init__` | method | `plank5.py:51` | `def __init__(self, sparsity_target)` |
| `__init__` | method | `plank5.py:66` | `def __init__(self, input_dim, hidden_dim, num_classes)` |
| `__init__` | method | `plank5.py:89` | `def __init__(self, patience, gap_threshold)` |
| `__init__` | method | `plank5.py:122` | `def __init__(self, device, target_acc, min_L)` |
| `__init__` | method | `plank5.py:212` | `def __init__(self, device, base_acc)` |
| `__init__` | method | `plank5.py:393` | `def __init__(self, device, num_cycles, base_acc)` |
| `apply_to_model` | method | `plank5.py:55` | `def apply_to_model(self, model)` |
| `compute_L` | method | `plank5.py:34` | `def compute_L(self, weight)` |
| `detect_grokking` | method | `plank5.py:103` | `def detect_grokking(self)` |
| `distill_sparse_model` | method | `plank5.py:337` | `def distill_sparse_model(self, model, target_acc)` |
| `forward` | method | `plank5.py:80` | `def forward(self, x)` |
| `generate` | method | `plank5.py:128` | `def generate(self, hidden_dim)` |
| `inoculate_dna` | method | `plank5.py:218` | `def inoculate_dna(self, large_model, seed_weights, noise_scale)` |
| `load_legacy_or_generate_seed` | method | `plank5.py:402` | `def load_legacy_or_generate_seed(self)` |
| `main` | method | `plank5.py:556` | `def main()` |
| `print_evolution_summary` | method | `plank5.py:519` | `def print_evolution_summary(self)` |
| `reduce_input` | method | `plank5.py:74` | `def reduce_input(self, x)` |
| `run_evolutionary_chain` | method | `plank5.py:419` | `def run_evolutionary_chain(self)` |
| `save_chain_results` | method | `plank5.py:501` | `def save_chain_results(self)` |
| `train_with_grokking` | method | `plank5.py:243` | `def train_with_grokking(self, model, seed_model, target_acc)` |
| `update` | method | `plank5.py:94` | `def update(self, train_acc, test_acc, epoch)` |
| `GrokkingDetector` | class | `plank6.py:78` | `class GrokkingDetector` |
| `GuidedElkHuntingEngine` | class | `plank6.py:101` | `class GuidedElkHuntingEngine` |
| `SpectralMLP` | class | `plank6.py:51` | `class SpectralMLP(Module)` |
| `SpectralMonitor` | class | `plank6.py:29` | `class SpectralMonitor` |
| `TrainingCycle` | class | `plank6.py:199` | `class TrainingCycle` |
| `__init__` | method | `plank6.py:31` | `def __init__(self, epsilon_c)` |
| `__init__` | method | `plank6.py:53` | `def __init__(self, input_dim, hidden_dim, num_classes)` |
| `__init__` | method | `plank6.py:79` | `def __init__(self, patience, gap_threshold)` |
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
