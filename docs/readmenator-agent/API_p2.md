# API (page 2 of 2)
Previous: [API.md](API.md)

## apex33.py
- `set_seed` (function) `apex33.py:41` `def set_seed(seed)` -- Ensure full reproducibility across runs
- `GatedTokenMixer.__init__` (method) `apex33.py:74` `def __init__(self, num_patches, embed_dim)`
- `GatedTokenMixer.forward` (method) `apex33.py:110` `def forward(self, x)`
- `PatchFeatureExtractor.__init__` (method) `apex33.py:119` `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)`
- `PatchFeatureExtractor.freeze` (method) `apex33.py:135` `def freeze(self)`
- `PatchFeatureExtractor.unfreeze_mixer_only` (method) `apex33.py:138` `def unfreeze_mixer_only(self)`
- `PatchFeatureExtractor.forward` (method) `apex33.py:143` `def forward(self, x)`
- `TaxonomicMLP.__init__` (method) `apex33.py:152` `def __init__(self, input_dim, hidden_dim, num_classes, num_superclasses)`
- `TaxonomicMLP.apply_masks` (method) `apex33.py:167` `def apply_masks(self)`
- `TaxonomicMLP.get_sparsity` (method) `apex33.py:173` `def get_sparsity(self)`
- `TaxonomicMLP.forward` (method) `apex33.py:178` `def forward(self, x)`
- `TaxonomicMLP.compute_spectral_loss` (method) `apex33.py:190` `def compute_spectral_loss(W)`
- `SpectralMonitor.__init__` (method) `apex33.py:202` `def __init__(self, epsilon)`
- `SpectralMonitor.compute_metrics` (method) `apex33.py:205` `def compute_metrics(self, weight)`
- `SpectralMonitor.get_singular_values` (method) `apex33.py:219` `def get_singular_values(self, weight)`
- `AdaptiveTopologyController.__init__` (method) `apex33.py:229` `def __init__(self, semantic_plasticity_threshold, stagnation_limit, mixer_noise_scale, dominant_energy_threshold...`
- `AdaptiveTopologyController.compute_semantic_plasticity_ratio` (method) `apex33.py:242` `def compute_semantic_plasticity_ratio(self)`
- `AdaptiveTopologyController.detect_intervention_need` (method) `apex33.py:250` `def detect_intervention_need(self, phase_state, extractor)`
- `AdaptiveTopologyController.update_history` (method) `apex33.py:273` `def update_history(self, topo_ratio, coarse_acc)`
- `AdaptiveTopologyController.perturb_mixer_targeted` (method) `apex33.py:284` `def perturb_mixer_targeted(self, extractor)`
- `IterativeRefinementEngine.__init__` (method) `apex33.py:311` `def __init__(self, device)`
- `IterativeRefinementEngine.create_refined_model` (method) `apex33.py:315` `def create_refined_model(self, parent_state, data_loader, feature_extractor, lambda_taxonomic, learning_rate)`
- `IterativeRefinementTrainer.__init__` (method) `apex33.py:344` `def __init__(self, device, output_dir)`
- `IterativeRefinementTrainer.load_data` (method) `apex33.py:367` `def load_data(self, cycle, batch_size)`
- `IterativeRefinementTrainer.train_model` (method) `apex33.py:385` `def train_model(self, model, cycle, chain_type, feature_extractor)`
- `IterativeRefinementTrainer.detect_phase_state` (method) `apex33.py:570` `def detect_phase_state(self, ratio_history, phase_window, phase_std_dev_limit)`
- `IterativeRefinementTrainer.compute_topology_ratio` (method) `apex33.py:580` `def compute_topology_ratio(self, model, extractor, chain_type)`
- `IterativeRefinementTrainer.run_refinement` (method) `apex33.py:592` `def run_refinement(self, num_iterations, num_seeds, early_stop_patience)`
- `CoarseCIFAR100.run_hierarchy_benchmark` (method) `apex33.py:807` `def run_hierarchy_benchmark(model_apex, model_blind, feature_extractor_apex, feature_extractor_blind, device)`
- `CoarseCIFAR100.evaluate` (method) `apex33.py:814` `def evaluate(model, extractor)`
- `CoarseCIFAR100.visualize_singular_values` (method) `apex33.py:845` `def visualize_singular_values(model_apex, model_blind, feature_extractor_apex, feature_extractor_blind, device...`
- `CoarseCIFAR100.get_singular_values` (method) `apex33.py:851` `def get_singular_values(model, extractor)`
- `CoarseCIFAR100.parse_args` (method) `apex33.py:894` `def parse_args()`
- `CoarseCIFAR100.main` (method) `apex33.py:903` `def main()`

## apex34.py
- `set_seed` (function) `apex34.py:41` `def set_seed(seed)` -- Ensure full reproducibility across runs
- `GatedTokenMixer.__init__` (method) `apex34.py:74` `def __init__(self, num_patches, embed_dim)`
- `GatedTokenMixer.forward` (method) `apex34.py:110` `def forward(self, x)`
- `PatchFeatureExtractor.__init__` (method) `apex34.py:119` `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)`
- `PatchFeatureExtractor.freeze` (method) `apex34.py:135` `def freeze(self)`
- `PatchFeatureExtractor.unfreeze_mixer_only` (method) `apex34.py:138` `def unfreeze_mixer_only(self)`
- `PatchFeatureExtractor.forward` (method) `apex34.py:143` `def forward(self, x)`
- `TaxonomicMLP.__init__` (method) `apex34.py:152` `def __init__(self, input_dim, hidden_dim, num_classes, num_superclasses)`
- `TaxonomicMLP.apply_masks` (method) `apex34.py:167` `def apply_masks(self)`
- `TaxonomicMLP.get_sparsity` (method) `apex34.py:173` `def get_sparsity(self)`
- `TaxonomicMLP.forward` (method) `apex34.py:178` `def forward(self, x)`
- `TaxonomicMLP.compute_spectral_loss` (method) `apex34.py:190` `def compute_spectral_loss(W)`
- `SpectralMonitor.__init__` (method) `apex34.py:202` `def __init__(self, epsilon)`
- `SpectralMonitor.compute_metrics` (method) `apex34.py:205` `def compute_metrics(self, weight)`
- `SpectralMonitor.get_singular_values` (method) `apex34.py:219` `def get_singular_values(self, weight)`
- `AdaptiveTopologyController.__init__` (method) `apex34.py:229` `def __init__(self, semantic_plasticity_threshold, stagnation_limit, mixer_noise_scale, dominant_energy_threshold...`
- `AdaptiveTopologyController.compute_semantic_plasticity_ratio` (method) `apex34.py:242` `def compute_semantic_plasticity_ratio(self)`
- `AdaptiveTopologyController.detect_intervention_need` (method) `apex34.py:250` `def detect_intervention_need(self, phase_state, extractor)`
- `AdaptiveTopologyController.update_history` (method) `apex34.py:273` `def update_history(self, topo_ratio, coarse_acc)`
- `AdaptiveTopologyController.perturb_mixer_targeted` (method) `apex34.py:284` `def perturb_mixer_targeted(self, extractor)`
- `IterativeRefinementEngine.__init__` (method) `apex34.py:311` `def __init__(self, device)`
- `IterativeRefinementEngine.create_refined_model` (method) `apex34.py:315` `def create_refined_model(self, parent_state, data_loader, feature_extractor, lambda_taxonomic, learning_rate)`
- `IterativeRefinementTrainer.__init__` (method) `apex34.py:344` `def __init__(self, device, output_dir)`
- `IterativeRefinementTrainer.load_data` (method) `apex34.py:367` `def load_data(self, cycle, batch_size)`
- `IterativeRefinementTrainer.train_model` (method) `apex34.py:385` `def train_model(self, model, cycle, chain_type, feature_extractor)`
- `IterativeRefinementTrainer.detect_phase_state` (method) `apex34.py:570` `def detect_phase_state(self, ratio_history, phase_window, phase_std_dev_limit)`
- `IterativeRefinementTrainer.compute_topology_ratio` (method) `apex34.py:580` `def compute_topology_ratio(self, model, extractor, chain_type)`
- `IterativeRefinementTrainer.run_refinement` (method) `apex34.py:592` `def run_refinement(self, num_iterations, num_seeds, early_stop_patience)`
- `CoarseCIFAR100.run_hierarchy_benchmark` (method) `apex34.py:807` `def run_hierarchy_benchmark(model_apex, model_blind, feature_extractor_apex, feature_extractor_blind, device)`
- `CoarseCIFAR100.evaluate` (method) `apex34.py:814` `def evaluate(model, extractor)`
- `CoarseCIFAR100.visualize_singular_values` (method) `apex34.py:845` `def visualize_singular_values(model_apex, model_blind, feature_extractor_apex, feature_extractor_blind, device...`
- `CoarseCIFAR100.get_singular_values` (method) `apex34.py:851` `def get_singular_values(model, extractor)`
- `CoarseCIFAR100.main` (method) `apex34.py:894` `def main()`

## apex35.py
- `set_seed` (function) `apex35.py:36` `def set_seed(seed)`
- `GatedTokenMixer.__init__` (method) `apex35.py:68` `def __init__(self, num_patches, embed_dim)`
- `GatedTokenMixer.forward` (method) `apex35.py:104` `def forward(self, x)`
- `E8FusionLayer.__init__` (method) `apex35.py:118` `def __init__(self, embed_dim, num_heads)`
- `E8FusionLayer.forward` (method) `apex35.py:136` `def forward(self, x)`
- `PatchFeatureExtractor.__init__` (method) `apex35.py:157` `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)`
- `PatchFeatureExtractor.freeze` (method) `apex35.py:173` `def freeze(self)`
- `PatchFeatureExtractor.unfreeze_mixer_only` (method) `apex35.py:176` `def unfreeze_mixer_only(self)`
- `PatchFeatureExtractor.forward` (method) `apex35.py:181` `def forward(self, x)`
- `TaxonomicMLP.__init__` (method) `apex35.py:189` `def __init__(self, input_dim, hidden_dim, num_classes, num_superclasses)`
- `TaxonomicMLP.apply_masks` (method) `apex35.py:204` `def apply_masks(self)`
- `TaxonomicMLP.get_sparsity` (method) `apex35.py:210` `def get_sparsity(self)`
- `TaxonomicMLP.forward` (method) `apex35.py:215` `def forward(self, x)`
- `BlackMirrorMonitor.__init__` (method) `apex35.py:229` `def __init__(self, epsilon)`
- `BlackMirrorMonitor.inspect` (method) `apex35.py:232` `def inspect(self, weight)`
- `IterativeRefinementTrainer.__init__` (method) `apex35.py:254` `def __init__(self, device, output_dir)`
- `IterativeRefinementTrainer.load_data` (method) `apex35.py:274` `def load_data(self, cycle, batch_size)`
- `IterativeRefinementTrainer.train_model` (method) `apex35.py:292` `def train_model(self, model, cycle, chain_type, feature_extractor)`
- `CoarseCIFAR100.main` (method) `apex35.py:444` `def main()`
- `CoarseCIFAR100.evaluate_safe` (method) `apex35.py:483` `def evaluate_safe(model, extractor)`

## app.py
- `SpectralMonitor.__init__` (method) `app.py:52` `def __init__(self, epsilon_c)`
- `SpectralMonitor.compute_L` (method) `app.py:55` `def compute_L(self, weight)`
- `PersistentPruner.__init__` (method) `app.py:77` `def __init__(self, sparsity_target)`
- `PersistentPruner.apply_to_model` (method) `app.py:81` `def apply_to_model(self, model)`
- `PersistentPruner.enforce_during_training` (method) `app.py:91` `def enforce_during_training(self, model)`
- `SpectralMLP.__init__` (method) `app.py:99` `def __init__(self, input_dim, hidden_dim, num_classes)`
- `SpectralMLP.reduce_input` (method) `app.py:107` `def reduce_input(self, x)`
- `SpectralMLP.forward` (method) `app.py:113` `def forward(self, x)`
- `EvolutionaryResonanceEngine.__init__` (method) `app.py:122` `def __init__(self, device, base_target_acc)`
- `EvolutionaryResonanceEngine.load_best_legacy_model` (method) `app.py:137` `def load_best_legacy_model(self, cycle)` -- Load the best model from previous cycle, with fallback to initial seed
- `EvolutionaryResonanceEngine.train_base_model_to_target` (method) `app.py:175` `def train_base_model_to_target(self, hidden_dim, target_acc, test_loader)` -- Train a base model to target accuracy
- `EvolutionaryResonanceEngine.extract_seed_from_checkpoint` (method) `app.py:228` `def extract_seed_from_checkpoint(self, checkpoint, expected_hidden_dim)` -- Extract seed weights from checkpoint, handling different formats
- `EvolutionaryResonanceEngine.extract_seed_weights` (method) `app.py:254` `def extract_seed_weights(self, model)`
- `EvolutionaryResonanceEngine.inoculate_seed_adaptive` (method) `app.py:260` `def inoculate_seed_adaptive(self, large_model, seed_weights)` -- Adaptive inoculation that handles dimension mismatches
- `EvolutionaryResonanceEngine.measure_functional_alignment` (method) `app.py:288` `def measure_functional_alignment(self, model1, model2, test_loader)` -- Measure functional alignment via logit cosine similarity
- `EvolutionaryResonanceEngine.progressive_pruning_with_target` (method) `app.py:309` `def progressive_pruning_with_target(self, model, target_acc, test_loader, max_density)` -- Prune while maintaining target accuracy, with density constraint
- `EvolutionaryResonanceEngine.execute_resonance_cycle` (method) `app.py:351` `def execute_resonance_cycle(self, cycle, test_loader, expansion_factor)`
- `EvolutionaryResonanceEngine.run_evolutionary_experiment` (method) `app.py:484` `def run_evolutionary_experiment(self, num_cycles)`
- `EvolutionaryResonanceEngine.main` (method) `app.py:561` `def main()`

## plank.py
- `BlackMirrorMonitor.__init__` (method) `plank.py:33` `def __init__(self, epsilon_c)`
- `BlackMirrorMonitor.inspect` (method) `plank.py:36` `def inspect(self, weights)`
- `SovereignNeuron.__init__` (method) `plank.py:63` `def __init__(self, in_features, out_features, sparsity_target)`
- `SovereignNeuron.forward` (method) `plank.py:70` `def forward(self, x, inject_lies)`
- `SovereignNeuron.apply_black_swan_refraction` (method) `plank.py:87` `def apply_black_swan_refraction(self)` -- Purificación extrema: sparsity 0.0004%
- `NeuroSovereign.__init__` (method) `plank.py:109` `def __init__(self, sparsity_target)`
- `NeuroSovereign.forward` (method) `plank.py:117` `def forward(self, x, inject_lies)`
- `SovereignTrainer.__init__` (method) `plank.py:133` `def __init__(self, model, device)`
- `SovereignTrainer.train_epoch` (method) `plank.py:139` `def train_epoch(self, dataloader, epoch)`
- `SovereignTrainer.main` (method) `plank.py:178` `def main()`

## plank10.py
- `PatchFeatureExtractor.__init__` (method) `plank10.py:39` `def __init__(self, img_size, patch_size, in_chans, embed_dim)`
- `PatchFeatureExtractor.freeze` (method) `plank10.py:67` `def freeze(self)`
- `PatchFeatureExtractor.unfreeze` (method) `plank10.py:71` `def unfreeze(self)`
- `PatchFeatureExtractor.forward` (method) `plank10.py:75` `def forward(self, x)`
- `LotteryMLP.__init__` (method) `plank10.py:92` `def __init__(self, input_dim, hidden_dim, num_classes)`
- `LotteryMLP.apply_masks` (method) `plank10.py:104` `def apply_masks(self)`
- `LotteryMLP.get_sparsity` (method) `plank10.py:109` `def get_sparsity(self)`
- `LotteryMLP.forward` (method) `plank10.py:114` `def forward(self, x)`
- `StandardBaseline.__init__` (method) `plank10.py:123` `def __init__(self, input_dim, hidden_dim)`
- `StandardBaseline.forward` (method) `plank10.py:127` `def forward(self, x)`
- `SpectralMonitor.compute_L` (method) `plank10.py:134` `def compute_L(self, weight)`
- `ApexEvolutionEngine.__init__` (method) `plank10.py:149` `def __init__(self, device)`
- `ApexEvolutionEngine.create_apex_offspring` (method) `plank10.py:211` `def create_apex_offspring(self, elk_state, new_hidden_dim, cycle, data_loader, feature_extractor, parent_gap)`
- `ApexTrainer.__init__` (method) `plank10.py:248` `def __init__(self, device, feature_extractor)`
- `ApexTrainer.get_curriculum_dataset` (method) `plank10.py:259` `def get_curriculum_dataset(self, cycle)`
- `ApexTrainer.train_model` (method) `plank10.py:265` `def train_model(self, model, cycle, is_baseline)`
- `ApexTrainer.main` (method) `plank10.py:354` `def main()`

## plank11.py
- `TokenMixer.__init__` (method) `plank11.py:41` `def __init__(self, num_tokens)`
- `TokenMixer.forward` (method) `plank11.py:50` `def forward(self, x)`
- `PatchFeatureExtractor.__init__` (method) `plank11.py:61` `def __init__(self, img_size, patch_size, in_chans, embed_dim)`
- `PatchFeatureExtractor.freeze` (method) `plank11.py:75` `def freeze(self)`
- `PatchFeatureExtractor.unfreeze` (method) `plank11.py:79` `def unfreeze(self)`
- `PatchFeatureExtractor.forward` (method) `plank11.py:83` `def forward(self, x)`
- `LotteryMLP.__init__` (method) `plank11.py:98` `def __init__(self, input_dim, hidden_dim, num_classes)`
- `LotteryMLP.apply_masks` (method) `plank11.py:109` `def apply_masks(self)`
- `LotteryMLP.get_sparsity` (method) `plank11.py:114` `def get_sparsity(self)`
- `LotteryMLP.forward` (method) `plank11.py:119` `def forward(self, x)`
- `StandardBaseline.__init__` (method) `plank11.py:127` `def __init__(self, input_dim, hidden_dim)`
- `StandardBaseline.forward` (method) `plank11.py:131` `def forward(self, x)`
- `SpectralMonitor.compute_L` (method) `plank11.py:138` `def compute_L(self, weight)`
- `OrthogonalEvolutionEngine.__init__` (method) `plank11.py:153` `def __init__(self, device)`
- `OrthogonalEvolutionEngine.create_orthogonal_offspring` (method) `plank11.py:223` `def create_orthogonal_offspring(self, elk_state, new_hidden_dim, cycle, data_loader, feature_extractor, parent_gap)`
- `OrthogonalTrainer.__init__` (method) `plank11.py:261` `def __init__(self, device, feature_extractor)`
- `OrthogonalTrainer.get_curriculum_dataset` (method) `plank11.py:272` `def get_curriculum_dataset(self, cycle)`
- `OrthogonalTrainer.train_model` (method) `plank11.py:278` `def train_model(self, model, cycle, is_baseline)`
- `OrthogonalTrainer.main` (method) `plank11.py:379` `def main()`

## plank12.py
- `TokenMixer.__init__` (method) `plank12.py:37` `def __init__(self, num_tokens)`
- `TokenMixer.forward` (method) `plank12.py:44` `def forward(self, x)`
- `PatchFeatureExtractor.__init__` (method) `plank12.py:52` `def __init__(self, img_size, patch_size, in_chans, embed_dim)`
- `PatchFeatureExtractor.freeze` (method) `plank12.py:66` `def freeze(self)`
- `PatchFeatureExtractor.unfreeze` (method) `plank12.py:70` `def unfreeze(self)`
- `PatchFeatureExtractor.forward` (method) `plank12.py:74` `def forward(self, x)`
- `LotteryMLP.__init__` (method) `plank12.py:88` `def __init__(self, input_dim, hidden_dim, num_classes)`
- `LotteryMLP.apply_masks` (method) `plank12.py:99` `def apply_masks(self)`
- `LotteryMLP.get_sparsity` (method) `plank12.py:104` `def get_sparsity(self)`
- `LotteryMLP.forward` (method) `plank12.py:109` `def forward(self, x)`
- `StandardBaseline.__init__` (method) `plank12.py:117` `def __init__(self, input_dim, hidden_dim)`
- `StandardBaseline.forward` (method) `plank12.py:121` `def forward(self, x)`
- `SpectralMonitor.compute_metrics` (method) `plank12.py:128` `def compute_metrics(self, weight)` -- Returns: (L, Rank_Efficient, S_vN) Used for logging and decision making (NOT for backprop).
- `OrthogonalEvolutionEngine.__init__` (method) `plank12.py:155` `def __init__(self, device)`
- `OrthogonalEvolutionEngine.create_refined_offspring` (method) `plank12.py:224` `def create_refined_offspring(self, elk_state, data_loader, feature_extractor)` -- Crea un hijo de las MISMAS dimensiones (Fixed Width).
- `OrthogonalTrainer.__init__` (method) `plank12.py:249` `def __init__(self, device, feature_extractor)`
- `OrthogonalTrainer.get_curriculum_dataset` (method) `plank12.py:260` `def get_curriculum_dataset(self, cycle)`
- `OrthogonalTrainer.train_model` (method) `plank12.py:266` `def train_model(self, model, cycle, is_baseline)`
- `OrthogonalTrainer.main` (method) `plank12.py:363` `def main()`

## plank13.py
- `TokenMixer.__init__` (method) `plank13.py:34` `def __init__(self, num_tokens)`
- `TokenMixer.forward` (method) `plank13.py:41` `def forward(self, x)`
- `PatchFeatureExtractor.__init__` (method) `plank13.py:53` `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)`
- `PatchFeatureExtractor.freeze` (method) `plank13.py:70` `def freeze(self)`
- `PatchFeatureExtractor.unfreeze` (method) `plank13.py:74` `def unfreeze(self)`
- `PatchFeatureExtractor.forward` (method) `plank13.py:78` `def forward(self, x)`
- `LotteryMLP.__init__` (method) `plank13.py:93` `def __init__(self, input_dim, hidden_dim, num_classes)`
- `LotteryMLP.apply_masks` (method) `plank13.py:104` `def apply_masks(self)`
- `LotteryMLP.get_sparsity` (method) `plank13.py:109` `def get_sparsity(self)`
- `LotteryMLP.forward` (method) `plank13.py:114` `def forward(self, x)`
- `SpectralMonitor.compute_metrics` (method) `plank13.py:124` `def compute_metrics(self, weight)` -- Returns: (L, Rank_Efficient, S_vN)
- `OrthogonalEvolutionEngine.__init__` (method) `plank13.py:150` `def __init__(self, device)`
- `OrthogonalEvolutionEngine.create_refined_offspring` (method) `plank13.py:207` `def create_refined_offspring(self, elk_state, data_loader, feature_extractor)`
- `DualTrainer.__init__` (method) `plank13.py:225` `def __init__(self, device, extractor_apex, extractor_blind)`
- `DualTrainer.get_curriculum_dataset` (method) `plank13.py:237` `def get_curriculum_dataset(self, cycle)`
- `DualTrainer.train_single_chain` (method) `plank13.py:243` `def train_single_chain(self, model, cycle, chain_type)` -- Entrena una cadena específica (Apex o Blind).
- `DualTrainer.main` (method) `plank13.py:329` `def main()`

## plank2.py
- `SpectralMonitor.__init__` (method) `plank2.py:46` `def __init__(self, epsilon_c)`
- `SpectralMonitor.compute_L` (method) `plank2.py:49` `def compute_L(self, weight)` -- Returns: (L, S_vN, rank_eff, regime)
- `PersistentPruner.__init__` (method) `plank2.py:87` `def __init__(self, sparsity_target)`
- `PersistentPruner.apply_to_model` (method) `plank2.py:91` `def apply_to_model(self, model)` -- Apply pruning mask and register backward hook to zero gradients.
- `PersistentPruner.enforce_during_training` (method) `plank2.py:105` `def enforce_during_training(self, model)` -- Call this after every optimizer.step()
- `SpectralMLP.__init__` (method) `plank2.py:119` `def __init__(self, input_dim, hidden_dim, num_classes)`
- `SpectralMLP.reduce_input` (method) `plank2.py:128` `def reduce_input(self, x)` -- Reduce CIFAR-10 (32x32x3) to 32D for focus
- `SpectralMLP.forward` (method) `plank2.py:135` `def forward(self, x)`
- `SpectralMLP.train_condition` (method) `plank2.py:173` `def train_condition(condition_name, config, device, seed)` -- Train one condition and return full log as DataFrame.
- `SpectralMLP.main` (method) `plank2.py:281` `def main()`

## plank3.py
- `SpectralMonitor.__init__` (method) `plank3.py:35` `def __init__(self, epsilon_c)`
- `SpectralMonitor.compute_L` (method) `plank3.py:38` `def compute_L(self, weight)`
- `PersistentPruner.__init__` (method) `plank3.py:63` `def __init__(self, sparsity_target)`
- `PersistentPruner.apply_to_model` (method) `plank3.py:67` `def apply_to_model(self, model)`
- `PersistentPruner.enforce_during_training` (method) `plank3.py:77` `def enforce_during_training(self, model)`
- `SpectralMLP.__init__` (method) `plank3.py:88` `def __init__(self, input_dim, hidden_dim, num_classes)`
- `SpectralMLP.reduce_input` (method) `plank3.py:96` `def reduce_input(self, x)`
- `SpectralMLP.forward` (method) `plank3.py:102` `def forward(self, x)`
- `SpectralMLP.train_dense_to_target` (method) `plank3.py:110` `def train_dense_to_target(device, target_acc)` -- Train dense model until it reaches target accuracy.
- `SpectralMLP.progressive_pruning_search` (method) `plank3.py:188` `def progressive_pruning_search(model, device, target_acc)` -- Progressively prune model and find critical density threshold.
- `SpectralMLP.find_critical_threshold` (method) `plank3.py:264` `def find_critical_threshold(pruning_df, target_acc)` -- Find the minimum density where accuracy >= target_acc.
- `SpectralMLP.main` (method) `plank3.py:294` `def main()`

## plank4.py
- `SpectralMonitor.__init__` (method) `plank4.py:42` `def __init__(self, epsilon_c)`
- `SpectralMonitor.compute_L` (method) `plank4.py:45` `def compute_L(self, weight)`
- `PersistentPruner.__init__` (method) `plank4.py:67` `def __init__(self, sparsity_target)`
- `PersistentPruner.apply_to_model` (method) `plank4.py:71` `def apply_to_model(self, model)`
- `PersistentPruner.enforce_during_training` (method) `plank4.py:81` `def enforce_during_training(self, model)`
- `SpectralMLP.__init__` (method) `plank4.py:89` `def __init__(self, input_dim, hidden_dim, num_classes)`
- `SpectralMLP.reduce_input` (method) `plank4.py:97` `def reduce_input(self, x)`
- `SpectralMLP.forward` (method) `plank4.py:103` `def forward(self, x)`
- `FractalSovereigntyEngine.__init__` (method) `plank4.py:112` `def __init__(self, device, base_target_acc)`
- `FractalSovereigntyEngine.train_dense_model` (method) `plank4.py:123` `def train_dense_model(self, hidden_dim, target_acc)`
- `FractalSovereigntyEngine.extract_seed_weights` (method) `plank4.py:168` `def extract_seed_weights(self, model)`
- `FractalSovereigntyEngine.inoculate_seed` (method) `plank4.py:174` `def inoculate_seed(self, large_model, seed_weights)` -- Embed seed into larger architecture
- `FractalSovereigntyEngine.progressive_pruning` (method) `plank4.py:192` `def progressive_pruning(self, model, target_acc)`
- `FractalSovereigntyEngine.execute_cycle` (method) `plank4.py:236` `def execute_cycle(self, cycle, base_hidden_dim, expansion_factor)`
- `FractalSovereigntyEngine.run_experiment` (method) `plank4.py:281` `def run_experiment(self, num_cycles)`
- `FractalSovereigntyEngine.main` (method) `plank4.py:329` `def main()`

## plank5.py
- `SpectralMonitor.__init__` (method) `plank5.py:31` `def __init__(self, epsilon_c)`
- `SpectralMonitor.compute_L` (method) `plank5.py:34` `def compute_L(self, weight)`
- `PersistentPruner.__init__` (method) `plank5.py:51` `def __init__(self, sparsity_target)`
- `PersistentPruner.apply_to_model` (method) `plank5.py:55` `def apply_to_model(self, model)`
- `SpectralMLP.__init__` (method) `plank5.py:66` `def __init__(self, input_dim, hidden_dim, num_classes)`
- `SpectralMLP.reduce_input` (method) `plank5.py:74` `def reduce_input(self, x)`
- `SpectralMLP.forward` (method) `plank5.py:80` `def forward(self, x)`
- `GrokkingDetector.__init__` (method) `plank5.py:89` `def __init__(self, patience, gap_threshold)`
- `GrokkingDetector.update` (method) `plank5.py:94` `def update(self, train_acc, test_acc, epoch)`
- `GrokkingDetector.detect_grokking` (method) `plank5.py:103` `def detect_grokking(self)`
- `SyntheticBlackSwanGenerator.__init__` (method) `plank5.py:122` `def __init__(self, device, target_acc, min_L)`
- `SyntheticBlackSwanGenerator.generate` (method) `plank5.py:128` `def generate(self, hidden_dim)` -- Genera cisne negro sintético si no existe legacy
- `EvolutionCycle.__init__` (method) `plank5.py:212` `def __init__(self, device, base_acc)`
- `EvolutionCycle.inoculate_dna` (method) `plank5.py:218` `def inoculate_dna(self, large_model, seed_weights, noise_scale)` -- Inocula ADN del cisne anterior con mutación controlada
- `EvolutionCycle.train_with_grokking` (method) `plank5.py:243` `def train_with_grokking(self, model, seed_model, target_acc)` -- Entrena modelo induciendo grokking y monitoreando transición de fase
- `EvolutionCycle.distill_sparse_model` (method) `plank5.py:337` `def distill_sparse_model(self, model, target_acc)` -- Pruning progresivo para extraer nuevo cisne negro
- `EvolutionaryBlackSwanChain.__init__` (method) `plank5.py:393` `def __init__(self, device, num_cycles, base_acc)`
- `EvolutionaryBlackSwanChain.load_legacy_or_generate_seed` (method) `plank5.py:402` `def load_legacy_or_generate_seed(self)` -- Carga legacy seed o genera uno sintético
- `EvolutionaryBlackSwanChain.run_evolutionary_chain` (method) `plank5.py:419` `def run_evolutionary_chain(self)` -- Ejecuta la cadena evolutiva completa
- `EvolutionaryBlackSwanChain.save_chain_results` (method) `plank5.py:501` `def save_chain_results(self)` -- Guarda resultados completos de la cadena evolutiva
- `EvolutionaryBlackSwanChain.print_evolution_summary` (method) `plank5.py:519` `def print_evolution_summary(self)` -- Imprime resumen ejecutivo de la cadena evolutiva
- `EvolutionaryBlackSwanChain.main` (method) `plank5.py:556` `def main()`

## plank6.py
- `SpectralMonitor.__init__` (method) `plank6.py:31` `def __init__(self, epsilon_c)`
- `SpectralMonitor.compute_L` (method) `plank6.py:34` `def compute_L(self, weight)`
- `SpectralMLP.__init__` (method) `plank6.py:53` `def __init__(self, input_dim, hidden_dim, num_classes)`
- `SpectralMLP.reduce_input` (method) `plank6.py:63` `def reduce_input(self, x)`
- `SpectralMLP.forward` (method) `plank6.py:70` `def forward(self, x)`
- `GrokkingDetector.__init__` (method) `plank6.py:79` `def __init__(self, patience, gap_threshold)`
- `GrokkingDetector.update` (method) `plank6.py:84` `def update(self, train_acc, test_acc, epoch)`
- `GrokkingDetector.detect_grokking` (method) `plank6.py:87` `def detect_grokking(self)`
- `GuidedElkHuntingEngine.__init__` (method) `plank6.py:106` `def __init__(self, device)`
- `GuidedElkHuntingEngine.create_offspring_from_elk` (method) `plank6.py:158` `def create_offspring_from_elk(self, elk_state, new_hidden_dim, generation)` -- Crea un nuevo modelo (Cisne Negro) basado en el Elk (mejor modelo previo).
- `TrainingCycle.__init__` (method) `plank6.py:200` `def __init__(self, device)`
- `TrainingCycle.train_phase` (method) `plank6.py:204` `def train_phase(self, model, cycle_id)`
- `TrainingCycle.main` (method) `plank6.py:293` `def main()`

## plank7.py
- `SpectralMonitor.__init__` (method) `plank7.py:30` `def __init__(self, epsilon_c)`
- `SpectralMonitor.compute_L` (method) `plank7.py:33` `def compute_L(self, weight)`
- `SpectralMLP.__init__` (method) `plank7.py:46` `def __init__(self, input_dim, hidden_dim, num_classes)`
- `SpectralMLP.reduce_input` (method) `plank7.py:54` `def reduce_input(self, x)`
- `SpectralMLP.forward` (method) `plank7.py:60` `def forward(self, x)`
- `AdvancedEvolutionEngine.__init__` (method) `plank7.py:72` `def __init__(self, device)`
- `AdvancedEvolutionEngine.create_advanced_offspring` (method) `plank7.py:124` `def create_advanced_offspring(self, elk_state, new_hidden_dim, cycle, data_loader)` -- Crea un hijo combinando: 1.
- `CurriculumTrainingCycle.__init__` (method) `plank7.py:178` `def __init__(self, device)`
- `CurriculumTrainingCycle.get_curriculum_dataset` (method) `plank7.py:186` `def get_curriculum_dataset(self, cycle)` -- Estrategia de Curriculum: Ciclos 1-3: Subset pequeño (Foco en estructura).
- `CurriculumTrainingCycle.train_phase` (method) `plank7.py:204` `def train_phase(self, model, cycle)`
- `CurriculumTrainingCycle.main` (method) `plank7.py:283` `def main()`

## plank8.py
- `PatchFeatureExtractor.__init__` (method) `plank8.py:39` `def __init__(self, img_size, patch_size, in_chans, embed_dim)`
- `PatchFeatureExtractor.freeze` (method) `plank8.py:57` `def freeze(self)`
- `PatchFeatureExtractor.unfreeze` (method) `plank8.py:61` `def unfreeze(self)`
- `PatchFeatureExtractor.forward` (method) `plank8.py:65` `def forward(self, x)`
- `LotteryMLP.__init__` (method) `plank8.py:79` `def __init__(self, input_dim, hidden_dim, num_classes)`
- `LotteryMLP.apply_masks` (method) `plank8.py:92` `def apply_masks(self)`
- `LotteryMLP.get_sparsity` (method) `plank8.py:97` `def get_sparsity(self)`
- `LotteryMLP.forward` (method) `plank8.py:102` `def forward(self, x)`
- `StandardBaseline.__init__` (method) `plank8.py:111` `def __init__(self, input_dim, hidden_dim)`
- `StandardBaseline.forward` (method) `plank8.py:115` `def forward(self, x)`
- `SpectralMonitor.compute_L` (method) `plank8.py:122` `def compute_L(self, weight)`
- `ApexEvolutionEngine.__init__` (method) `plank8.py:137` `def __init__(self, device)`
- `ApexEvolutionEngine.create_apex_offspring` (method) `plank8.py:199` `def create_apex_offspring(self, elk_state, new_hidden_dim, cycle, data_loader, feature_extractor, parent_gap)`
- `ApexTrainer.__init__` (method) `plank8.py:236` `def __init__(self, device, feature_extractor)`
- `ApexTrainer.get_curriculum_dataset` (method) `plank8.py:247` `def get_curriculum_dataset(self, cycle)`
- `ApexTrainer.train_model` (method) `plank8.py:253` `def train_model(self, model, cycle, is_baseline)`
- `ApexTrainer.main` (method) `plank8.py:342` `def main()`

## resmav2_1.py
- `OptimizedE8Layer.__init__` (method) `resmav2_1.py:25` `def __init__(self, in_features, out_features, edge_index, num_nodes)`
- `OptimizedE8Layer.forward` (method) `resmav2_1.py:40` `def forward(self, x)`
- `RESMAv2Fast.__init__` (method) `resmav2_1.py:50` `def __init__(self, input_dim, hidden_dim, edge_index, num_nodes, dropout)`
- `RESMAv2Fast.forward` (method) `resmav2_1.py:73` `def forward(self, x, edge_index)`
- `RESMAv2Standard.__init__` (method) `resmav2_1.py:88` `def __init__(self, input_dim, hidden_dim, edge_index, num_nodes, dropout)`
- `RESMAv2Standard.forward` (method) `resmav2_1.py:113` `def forward(self, x, edge_index)`
- `RESMAv2Deep.__init__` (method) `resmav2_1.py:136` `def __init__(self, input_dim, hidden_dim, edge_index, num_nodes, dropout)`
- `RESMAv2Deep.forward` (method) `resmav2_1.py:168` `def forward(self, x, edge_index)`
- `GAT_Baseline.__init__` (method) `resmav2_1.py:184` `def __init__(self, input_dim, hidden_dim, dropout)`
- `GAT_Baseline.forward` (method) `resmav2_1.py:194` `def forward(self, x, edge_index)`
- `GAT_Baseline.load_elliptic_data` (method) `resmav2_1.py:207` `def load_elliptic_data()`
- `GAT_Baseline.train_and_evaluate` (method) `resmav2_1.py:264` `def train_and_evaluate(model, X, y, edge_index, train_idx, val_idx, epochs, lr, name, fold)`
- `GAT_Baseline.cross_validate_model` (method) `resmav2_1.py:321` `def cross_validate_model(model_class, X, y, edge_index, num_nodes, n_splits, seed, name)`

