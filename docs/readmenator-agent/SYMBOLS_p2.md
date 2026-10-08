# Symbols (page 2 of 3)
Previous: [SYMBOLS.md](SYMBOLS.md)

| Symbol | Kind | File:Line | Signature |
|--------|------|-----------|-----------|
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
| `visualize_singular_values` | method | `apex29.py:543` | `def visualize_singular_values(model_apex, model_blind, feature_extractor_apex, feature_extractor_blind, device...` |
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
| `__init__` | method | `apex30.py:279` | `def __init__(self, semantic_plasticity_threshold, stagnation_limit, mixer_noise_scale, dominant_energy_threshold...` |
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
| `visualize_singular_values` | method | `apex30.py:510` | `def visualize_singular_values(model_apex, model_blind, feature_extractor_apex, feature_extractor_blind, device...` |
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
| `__init__` | method | `apex31.py:281` | `def __init__(self, semantic_plasticity_threshold, stagnation_limit, mixer_noise_scale, dominant_energy_threshold...` |
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
| `visualize_singular_values` | method | `apex31.py:521` | `def visualize_singular_values(model_apex, model_blind, feature_extractor_apex, feature_extractor_blind, device...` |
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
| `__init__` | method | `apex33.py:229` | `def __init__(self, semantic_plasticity_threshold, stagnation_limit, mixer_noise_scale, dominant_energy_threshold...` |
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
| `visualize_singular_values` | method | `apex33.py:845` | `def visualize_singular_values(model_apex, model_blind, feature_extractor_apex, feature_extractor_blind, device...` |
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
| `__init__` | method | `apex34.py:229` | `def __init__(self, semantic_plasticity_threshold, stagnation_limit, mixer_noise_scale, dominant_energy_threshold...` |
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
| `visualize_singular_values` | method | `apex34.py:845` | `def visualize_singular_values(model_apex, model_blind, feature_extractor_apex, feature_extractor_blind, device...` |
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

Next: [SYMBOLS_p3.md](SYMBOLS_p3.md)
