# API

## apex14.py

### main (method) `def main()`
- Defined: `apex14.py:336`

### __init__ (method) `def __init__(self, num_tokens, embed_dim)`
- Defined: `apex14.py:37`

### forward (method) `def forward(self, x)`
- Defined: `apex14.py:54`

### __init__ (method) `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)`
- Defined: `apex14.py:72`

### freeze (method) `def freeze(self)`
- Defined: `apex14.py:89`

### unfreeze (method) `def unfreeze(self)`
- Defined: `apex14.py:93`

### forward (method) `def forward(self, x)`
- Defined: `apex14.py:97`

### __init__ (method) `def __init__(self, input_dim, hidden_dim, num_classes)`
- Defined: `apex14.py:112`

### apply_masks (method) `def apply_masks(self)`
- Defined: `apex14.py:123`

### get_sparsity (method) `def get_sparsity(self)`
- Defined: `apex14.py:128`

### forward (method) `def forward(self, x)`
- Defined: `apex14.py:133`

### compute_metrics (method) `def compute_metrics(self, weight)`
- Defined: `apex14.py:143`

### __init__ (method) `def __init__(self, device)`
- Defined: `apex14.py:159`

### _gradient_nudge_inheritance (method) `def _gradient_nudge_inheritance(self, child_model, elk_state, data_loader, feature_extractor, nudge_lr)`
- Defined: `apex14.py:164`

### _apply_rank_capping_shock (method) `def _apply_rank_capping_shock(self, model, layer_name, keep_ratio)`
- Defined: `apex14.py:198`

### create_refined_offspring (method) `def create_refined_offspring(self, elk_state, data_loader, feature_extractor)`
- Defined: `apex14.py:217`

### __init__ (method) `def __init__(self, device, extractor_apex, extractor_blind)`
- Defined: `apex14.py:232`

### _preprocess_batch (method) `def _preprocess_batch(self, x, extractor)`
- Defined: `apex14.py:240`

### get_curriculum_dataset (method) `def get_curriculum_dataset(self, cycle)`
- Defined: `apex14.py:244`

### train_single_chain (method) `def train_single_chain(self, model, cycle, chain_type)`
- Defined: `apex14.py:250`

## apex15.py

### compute_spectral_loss (function) `def compute_spectral_loss(W, target_rank_factor)`
- Defined: `apex15.py:63`
- Doc: Penaliza la desalineación entre Entropía Espectral y Rango Efectivo.

### main (method) `def main()`
- Defined: `apex15.py:367`

### __init__ (method) `def __init__(self, num_patches, embed_dim)`
- Defined: `apex15.py:89`

### forward (method) `def forward(self, x)`
- Defined: `apex15.py:98`

### __init__ (method) `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)`
- Defined: `apex15.py:106`

### freeze (method) `def freeze(self)`
- Defined: `apex15.py:121`

### unfreeze_mixer_only (method) `def unfreeze_mixer_only(self)`
- Defined: `apex15.py:125`

### forward (method) `def forward(self, x)`
- Defined: `apex15.py:131`

### __init__ (method) `def __init__(self, input_dim, hidden_dim, num_classes)`
- Defined: `apex15.py:144`

### apply_masks (method) `def apply_masks(self)`
- Defined: `apex15.py:158`

### get_sparsity (method) `def get_sparsity(self)`
- Defined: `apex15.py:164`

### forward (method) `def forward(self, x)`
- Defined: `apex15.py:169`

### compute_metrics (method) `def compute_metrics(self, weight)`
- Defined: `apex15.py:180`

### __init__ (method) `def __init__(self, device, extractor_apex, extractor_blind)`
- Defined: `apex15.py:196`

### _preprocess_batch (method) `def _preprocess_batch(self, x, extractor)`
- Defined: `apex15.py:203`

### get_curriculum_dataset (method) `def get_curriculum_dataset(self, cycle)`
- Defined: `apex15.py:207`

### train_single_chain (method) `def train_single_chain(self, model, cycle, chain_type)`
- Defined: `apex15.py:213`

## apex16.py

### compute_spectral_loss (function) `def compute_spectral_loss(W, target_rank_factor)`
- Defined: `apex16.py:63`
- Doc: Penaliza la desalineación entre Entropía Espectral y Rango Efectivo.

### main (method) `def main()`
- Defined: `apex16.py:377`

### __init__ (method) `def __init__(self, num_patches, embed_dim)`
- Defined: `apex16.py:83`

### forward (method) `def forward(self, x)`
- Defined: `apex16.py:92`

### __init__ (method) `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)`
- Defined: `apex16.py:100`

### freeze (method) `def freeze(self)`
- Defined: `apex16.py:115`

### unfreeze_mixer_only (method) `def unfreeze_mixer_only(self)`
- Defined: `apex16.py:119`

### forward (method) `def forward(self, x)`
- Defined: `apex16.py:125`

### __init__ (method) `def __init__(self, input_dim, hidden_dim, num_classes)`
- Defined: `apex16.py:138`

### apply_masks (method) `def apply_masks(self)`
- Defined: `apex16.py:152`

### get_sparsity (method) `def get_sparsity(self)`
- Defined: `apex16.py:158`

### forward (method) `def forward(self, x)`
- Defined: `apex16.py:163`

### compute_metrics (method) `def compute_metrics(self, weight)`
- Defined: `apex16.py:174`

### __init__ (method) `def __init__(self, device, extractor_apex, extractor_blind)`
- Defined: `apex16.py:190`

### _preprocess_batch (method) `def _preprocess_batch(self, x, extractor)`
- Defined: `apex16.py:197`

### get_curriculum_dataset (method) `def get_curriculum_dataset(self, cycle)`
- Defined: `apex16.py:201`

### train_single_chain (method) `def train_single_chain(self, model, cycle, chain_type)`
- Defined: `apex16.py:207`

## apex17.py

### compute_spectral_loss (function) `def compute_spectral_loss(W)`
- Defined: `apex17.py:65`
- Doc: v15.0: Optimization Objective for Spectral Control.

### run_hierarchy_benchmark (method) `def run_hierarchy_benchmark(model_apex, model_blind, device)`
- Defined: `apex17.py:378`

### main (method) `def main()`
- Defined: `apex17.py:428`

### __init__ (method) `def __init__(self, num_patches, embed_dim)`
- Defined: `apex17.py:82`

### forward (method) `def forward(self, x)`
- Defined: `apex17.py:91`

### __init__ (method) `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)`
- Defined: `apex17.py:99`

### freeze (method) `def freeze(self)`
- Defined: `apex17.py:114`

### unfreeze_mixer_only (method) `def unfreeze_mixer_only(self)`
- Defined: `apex17.py:118`

### forward (method) `def forward(self, x)`
- Defined: `apex17.py:124`

### __init__ (method) `def __init__(self, input_dim, hidden_dim, num_classes)`
- Defined: `apex17.py:137`

### apply_masks (method) `def apply_masks(self)`
- Defined: `apex17.py:151`

### get_sparsity (method) `def get_sparsity(self)`
- Defined: `apex17.py:157`

### forward (method) `def forward(self, x)`
- Defined: `apex17.py:162`

### compute_metrics (method) `def compute_metrics(self, weight)`
- Defined: `apex17.py:173`
- Doc: L_mon: Used for plotting and historical reporting, not optimization.

### __init__ (method) `def __init__(self, device, extractor_apex, extractor_blind)`
- Defined: `apex17.py:191`

### _preprocess_batch (method) `def _preprocess_batch(self, x, extractor)`
- Defined: `apex17.py:198`

### get_curriculum_dataset (method) `def get_curriculum_dataset(self, cycle)`
- Defined: `apex17.py:202`

### train_single_chain (method) `def train_single_chain(self, model, cycle, chain_type)`
- Defined: `apex17.py:208`

### __getitem__ (method) `def __getitem__(self, index)`
- Defined: `apex17.py:373`

### evaluate (method) `def evaluate(model, extractor, name)`
- Defined: `apex17.py:391`

## apex18.py

### compute_spectral_loss (function) `def compute_spectral_loss(W)`
- Defined: `apex18.py:65`
- Doc: v15.1: Optimization Objective for Spectral Control (Applied to both APEX and BLIND).

### run_hierarchy_benchmark (method) `def run_hierarchy_benchmark(model_apex, model_blind, device)`
- Defined: `apex18.py:374`

### main (method) `def main()`
- Defined: `apex18.py:420`

### __init__ (method) `def __init__(self, num_patches, embed_dim)`
- Defined: `apex18.py:82`

### forward (method) `def forward(self, x)`
- Defined: `apex18.py:91`

### __init__ (method) `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)`
- Defined: `apex18.py:99`

### freeze (method) `def freeze(self)`
- Defined: `apex18.py:114`

### unfreeze_mixer_only (method) `def unfreeze_mixer_only(self)`
- Defined: `apex18.py:118`

### forward (method) `def forward(self, x)`
- Defined: `apex18.py:124`

### __init__ (method) `def __init__(self, input_dim, hidden_dim, num_classes)`
- Defined: `apex18.py:137`

### apply_masks (method) `def apply_masks(self)`
- Defined: `apex18.py:151`

### get_sparsity (method) `def get_sparsity(self)`
- Defined: `apex18.py:157`

### forward (method) `def forward(self, x)`
- Defined: `apex18.py:162`

### compute_metrics (method) `def compute_metrics(self, weight)`
- Defined: `apex18.py:173`

### __init__ (method) `def __init__(self, device, extractor_apex, extractor_blind)`
- Defined: `apex18.py:189`

### _preprocess_batch (method) `def _preprocess_batch(self, x, extractor)`
- Defined: `apex18.py:196`

### get_curriculum_dataset (method) `def get_curriculum_dataset(self, cycle)`
- Defined: `apex18.py:200`

### train_single_chain (method) `def train_single_chain(self, model, cycle, chain_type)`
- Defined: `apex18.py:206`

### __getitem__ (method) `def __getitem__(self, index)`
- Defined: `apex18.py:370`

### evaluate (method) `def evaluate(model, extractor, name)`
- Defined: `apex18.py:386`

## apex19.py

### compute_spectral_loss (function) `def compute_spectral_loss(W)`
- Defined: `apex19.py:68`
- Doc: L_opt: Optimization Objective for Structural Control.

### run_hierarchy_benchmark (method) `def run_hierarchy_benchmark(model_apex, model_blind, device)`
- Defined: `apex19.py:399`

### main (method) `def main()`
- Defined: `apex19.py:445`

### __init__ (method) `def __init__(self, num_patches, embed_dim)`
- Defined: `apex19.py:85`

### forward (method) `def forward(self, x)`
- Defined: `apex19.py:94`

### __init__ (method) `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)`
- Defined: `apex19.py:102`

### freeze (method) `def freeze(self)`
- Defined: `apex19.py:117`

### unfreeze_mixer_only (method) `def unfreeze_mixer_only(self)`
- Defined: `apex19.py:121`

### forward (method) `def forward(self, x)`
- Defined: `apex19.py:127`

### __init__ (method) `def __init__(self, input_dim, hidden_dim, num_classes)`
- Defined: `apex19.py:140`

### apply_masks (method) `def apply_masks(self)`
- Defined: `apex19.py:154`

### get_sparsity (method) `def get_sparsity(self)`
- Defined: `apex19.py:160`

### forward (method) `def forward(self, x)`
- Defined: `apex19.py:165`

### compute_metrics (method) `def compute_metrics(self, weight)`
- Defined: `apex19.py:176`

### compute_topology_ratio (method) `def compute_topology_ratio(self, model, extractor, chain_type)`
- Defined: `apex19.py:188`
- Doc: v15.2: Calcula el ratio R = L_opt / L_mon.

### __init__ (method) `def __init__(self, device, extractor_apex, extractor_blind)`
- Defined: `apex19.py:212`

### _preprocess_batch (method) `def _preprocess_batch(self, x, extractor)`
- Defined: `apex19.py:219`

### get_curriculum_dataset (method) `def get_curriculum_dataset(self, cycle)`
- Defined: `apex19.py:223`

### train_single_chain (method) `def train_single_chain(self, model, cycle, chain_type)`
- Defined: `apex19.py:229`

### __getitem__ (method) `def __getitem__(self, index)`
- Defined: `apex19.py:395`

### evaluate (method) `def evaluate(model, extractor, name)`
- Defined: `apex19.py:411`

## apex20.py

### compute_spectral_loss (function) `def compute_spectral_loss(W)`
- Defined: `apex20.py:69`
- Doc: L_opt: Optimization Objective for Structural Control.

### run_hierarchy_benchmark (method) `def run_hierarchy_benchmark(model_apex, model_blind, device)`
- Defined: `apex20.py:431`

### main (method) `def main()`
- Defined: `apex20.py:490`

### __init__ (method) `def __init__(self, num_patches, embed_dim)`
- Defined: `apex20.py:86`

### forward (method) `def forward(self, x)`
- Defined: `apex20.py:95`

### __init__ (method) `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)`
- Defined: `apex20.py:103`

### freeze (method) `def freeze(self)`
- Defined: `apex20.py:118`

### unfreeze_mixer_only (method) `def unfreeze_mixer_only(self)`
- Defined: `apex20.py:122`

### forward (method) `def forward(self, x)`
- Defined: `apex20.py:128`

### __init__ (method) `def __init__(self, input_dim, hidden_dim, num_classes)`
- Defined: `apex20.py:141`

### apply_masks (method) `def apply_masks(self)`
- Defined: `apex20.py:155`

### get_sparsity (method) `def get_sparsity(self)`
- Defined: `apex20.py:161`

### forward (method) `def forward(self, x)`
- Defined: `apex20.py:166`

### compute_metrics (method) `def compute_metrics(self, weight)`
- Defined: `apex20.py:177`

### detect_phase_state (method) `def detect_phase_state(self, ratio_history)`
- Defined: `apex20.py:190`
- Doc: v15.3: Detects phase state based on relative deviation, not absolute value.

### compute_topology_ratio (method) `def compute_topology_ratio(self, model, extractor, chain_type)`
- Defined: `apex20.py:216`
- Doc: v15.3: Returns L_opt components and Total Ratio.

### __init__ (method) `def __init__(self, device, extractor_apex, extractor_blind)`
- Defined: `apex20.py:243`

### _preprocess_batch (method) `def _preprocess_batch(self, x, extractor)`
- Defined: `apex20.py:250`

### get_curriculum_dataset (method) `def get_curriculum_dataset(self, cycle)`
- Defined: `apex20.py:254`

### train_single_chain (method) `def train_single_chain(self, model, cycle, chain_type)`
- Defined: `apex20.py:260`

### __getitem__ (method) `def __getitem__(self, index)`
- Defined: `apex20.py:427`

### evaluate (method) `def evaluate(model, extractor, name)`
- Defined: `apex20.py:443`

## apex21.py

### compute_spectral_loss (function) `def compute_spectral_loss(W)`
- Defined: `apex21.py:73`

### run_hierarchy_benchmark (method) `def run_hierarchy_benchmark(model_apex, model_blind, device)`
- Defined: `apex21.py:464`

### main (method) `def main()`
- Defined: `apex21.py:513`

### __init__ (method) `def __init__(self, num_patches, embed_dim)`
- Defined: `apex21.py:89`

### forward (method) `def forward(self, x)`
- Defined: `apex21.py:98`

### __init__ (method) `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)`
- Defined: `apex21.py:106`

### freeze (method) `def freeze(self)`
- Defined: `apex21.py:121`

### unfreeze_mixer_only (method) `def unfreeze_mixer_only(self)`
- Defined: `apex21.py:125`

### forward (method) `def forward(self, x)`
- Defined: `apex21.py:131`

### __init__ (method) `def __init__(self, input_dim, hidden_dim, num_classes)`
- Defined: `apex21.py:144`

### apply_masks (method) `def apply_masks(self)`
- Defined: `apex21.py:158`

### get_sparsity (method) `def get_sparsity(self)`
- Defined: `apex21.py:164`

### forward (method) `def forward(self, x)`
- Defined: `apex21.py:169`

### compute_metrics (method) `def compute_metrics(self, weight)`
- Defined: `apex21.py:180`

### detect_phase_state (method) `def detect_phase_state(self, ratio_history)`
- Defined: `apex21.py:192`

### compute_topology_ratio (method) `def compute_topology_ratio(self, model, extractor, chain_type)`
- Defined: `apex21.py:204`
- Doc: v15.3: Returns L_opt components and Total Ratio.

### __init__ (method) `def __init__(self)`
- Defined: `apex21.py:230`

### check_intervention (method) `def check_intervention(self, phase_state, coarse_acc, extractor)`
- Defined: `apex21.py:233`
- Doc: Decides whether to intervene.

### perturb_mixer (method) `def perturb_mixer(self, extractor)`
- Defined: `apex21.py:255`
- Doc: Causal Intervention: Inject topological noise to force phase shift.

### __init__ (method) `def __init__(self, device, extractor_apex, extractor_blind)`
- Defined: `apex21.py:271`

### _preprocess_batch (method) `def _preprocess_batch(self, x, extractor)`
- Defined: `apex21.py:279`

### get_curriculum_dataset (method) `def get_curriculum_dataset(self, cycle)`
- Defined: `apex21.py:283`

### train_single_chain (method) `def train_single_chain(self, model, cycle, chain_type)`
- Defined: `apex21.py:289`

### __getitem__ (method) `def __getitem__(self, index)`
- Defined: `apex21.py:460`

### evaluate (method) `def evaluate(model, extractor, name)`
- Defined: `apex21.py:476`

## apex22.py

### compute_spectral_loss (function) `def compute_spectral_loss(W)`
- Defined: `apex22.py:73`

### run_hierarchy_benchmark (method) `def run_hierarchy_benchmark(model_apex, model_blind, device)`
- Defined: `apex22.py:464`

### main (method) `def main()`
- Defined: `apex22.py:513`

### __init__ (method) `def __init__(self, num_patches, embed_dim)`
- Defined: `apex22.py:89`

### forward (method) `def forward(self, x)`
- Defined: `apex22.py:98`

### __init__ (method) `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)`
- Defined: `apex22.py:106`

### freeze (method) `def freeze(self)`
- Defined: `apex22.py:121`

### unfreeze_mixer_only (method) `def unfreeze_mixer_only(self)`
- Defined: `apex22.py:125`

### forward (method) `def forward(self, x)`
- Defined: `apex22.py:131`

### __init__ (method) `def __init__(self, input_dim, hidden_dim, num_classes)`
- Defined: `apex22.py:144`

### apply_masks (method) `def apply_masks(self)`
- Defined: `apex22.py:158`

### get_sparsity (method) `def get_sparsity(self)`
- Defined: `apex22.py:164`

### forward (method) `def forward(self, x)`
- Defined: `apex22.py:169`

### compute_metrics (method) `def compute_metrics(self, weight)`
- Defined: `apex22.py:180`

### detect_phase_state (method) `def detect_phase_state(self, ratio_history)`
- Defined: `apex22.py:192`

### __init__ (method) `def __init__(self)`
- Defined: `apex22.py:206`

### check_intervention (method) `def check_intervention(self, phase_state, coarse_acc, extractor)`
- Defined: `apex22.py:209`

### perturb_mixer_targeted (method) `def perturb_mixer_targeted(self, extractor)`
- Defined: `apex22.py:226`
- Doc: v15.5: Targeted Phase Surgery.

### __init__ (method) `def __init__(self, device, extractor_apex, extractor_blind)`
- Defined: `apex22.py:274`

### _preprocess_batch (method) `def _preprocess_batch(self, x, extractor)`
- Defined: `apex22.py:282`

### get_curriculum_dataset (method) `def get_curriculum_dataset(self, cycle)`
- Defined: `apex22.py:286`

### train_single_chain (method) `def train_single_chain(self, model, cycle, chain_type)`
- Defined: `apex22.py:292`

### __getitem__ (method) `def __getitem__(self, index)`
- Defined: `apex22.py:460`

### evaluate (method) `def evaluate(model, extractor, name)`
- Defined: `apex22.py:476`

## apex23.py

### compute_spectral_loss (function) `def compute_spectral_loss(W)`
- Defined: `apex23.py:74`
- Doc: L_opt: Computes the discrepancy between spectral entropy and effective rank.

### run_hierarchy_benchmark (method) `def run_hierarchy_benchmark(model_apex, model_blind, device)`
- Defined: `apex23.py:498`

### main (method) `def main()`
- Defined: `apex23.py:544`

### __init__ (method) `def __init__(self, num_patches, embed_dim)`
- Defined: `apex23.py:92`

### forward (method) `def forward(self, x)`
- Defined: `apex23.py:101`

### __init__ (method) `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)`
- Defined: `apex23.py:109`

### freeze (method) `def freeze(self)`
- Defined: `apex23.py:124`

### unfreeze_mixer_only (method) `def unfreeze_mixer_only(self)`
- Defined: `apex23.py:128`

### forward (method) `def forward(self, x)`
- Defined: `apex23.py:134`

### __init__ (method) `def __init__(self, input_dim, hidden_dim, num_classes)`
- Defined: `apex23.py:147`

### apply_masks (method) `def apply_masks(self)`
- Defined: `apex23.py:161`

### get_sparsity (method) `def get_sparsity(self)`
- Defined: `apex23.py:167`

### forward (method) `def forward(self, x)`
- Defined: `apex23.py:172`

### compute_metrics (method) `def compute_metrics(self, weight)`
- Defined: `apex23.py:183`
- Doc: L_mon: Legacy reporting metric.

### detect_phase_state (method) `def detect_phase_state(self, ratio_history)`
- Defined: `apex23.py:196`

### compute_topology_ratio (method) `def compute_topology_ratio(self, model, extractor, chain_type)`
- Defined: `apex23.py:208`
- Doc: Calculates Topo_R = L_opt / L_mon.

### __init__ (method) `def __init__(self)`
- Defined: `apex23.py:229`

### check_intervention (method) `def check_intervention(self, phase_state, coarse_acc, extractor)`
- Defined: `apex23.py:232`
- Doc: Decides if intervention is needed based on Phase and Performance.

### perturb_mixer_targeted (method) `def perturb_mixer_targeted(self, extractor)`
- Defined: `apex23.py:250`
- Doc: v15.5: Targeted Spectral Surgery.

### __init__ (method) `def __init__(self, device, extractor_apex, extractor_blind)`
- Defined: `apex23.py:307`

### _preprocess_batch (method) `def _preprocess_batch(self, x, extractor)`
- Defined: `apex23.py:315`

### get_curriculum_dataset (method) `def get_curriculum_dataset(self, cycle)`
- Defined: `apex23.py:319`

### train_single_chain (method) `def train_single_chain(self, model, cycle, chain_type)`
- Defined: `apex23.py:325`

### __getitem__ (method) `def __getitem__(self, index)`
- Defined: `apex23.py:494`

### evaluate (method) `def evaluate(model, extractor, name)`
- Defined: `apex23.py:510`

## apex24.py

### compute_spectral_loss (function) `def compute_spectral_loss(W)`
- Defined: `apex24.py:75`
- Doc: L_opt: Computes the discrepancy between spectral entropy and effective rank.

### run_hierarchy_benchmark (method) `def run_hierarchy_benchmark(model_apex, model_blind, device)`
- Defined: `apex24.py:532`

### main (method) `def main()`
- Defined: `apex24.py:578`

### __init__ (method) `def __init__(self, num_patches, embed_dim)`
- Defined: `apex24.py:93`

### forward (method) `def forward(self, x)`
- Defined: `apex24.py:102`

### __init__ (method) `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)`
- Defined: `apex24.py:110`

### freeze (method) `def freeze(self)`
- Defined: `apex24.py:125`

### unfreeze_mixer_only (method) `def unfreeze_mixer_only(self)`
- Defined: `apex24.py:129`

### forward (method) `def forward(self, x)`
- Defined: `apex24.py:135`

### __init__ (method) `def __init__(self, input_dim, hidden_dim, num_classes)`
- Defined: `apex24.py:148`

### apply_masks (method) `def apply_masks(self)`
- Defined: `apex24.py:162`

### get_sparsity (method) `def get_sparsity(self)`
- Defined: `apex24.py:168`

### forward (method) `def forward(self, x)`
- Defined: `apex24.py:173`

### compute_metrics (method) `def compute_metrics(self, weight)`
- Defined: `apex24.py:184`
- Doc: L_mon: Legacy reporting metric.

### detect_phase_state (method) `def detect_phase_state(self, ratio_history)`
- Defined: `apex24.py:197`

### compute_topology_ratio (method) `def compute_topology_ratio(self, model, extractor, chain_type)`
- Defined: `apex24.py:209`
- Doc: Calculates Topo_R = L_opt / L_mon.

### __init__ (method) `def __init__(self)`
- Defined: `apex24.py:230`

### check_intervention (method) `def check_intervention(self, phase_state, coarse_acc, extractor, current_topo_r)`
- Defined: `apex24.py:235`
- Doc: v15.5 Final: Geometric Mismatch Detection.

### perturb_mixer_targeted (method) `def perturb_mixer_targeted(self, extractor)`
- Defined: `apex24.py:284`
- Doc: v15.5: Targeted Spectral Surgery.

### __init__ (method) `def __init__(self, device, extractor_apex, extractor_blind)`
- Defined: `apex24.py:340`

### _preprocess_batch (method) `def _preprocess_batch(self, x, extractor)`
- Defined: `apex24.py:348`

### get_curriculum_dataset (method) `def get_curriculum_dataset(self, cycle)`
- Defined: `apex24.py:352`

### train_single_chain (method) `def train_single_chain(self, model, cycle, chain_type)`
- Defined: `apex24.py:358`

### __getitem__ (method) `def __getitem__(self, index)`
- Defined: `apex24.py:528`

### evaluate (method) `def evaluate(model, extractor, name)`
- Defined: `apex24.py:544`

## apex25.py

### compute_spectral_loss (function) `def compute_spectral_loss(W)`
- Defined: `apex25.py:75`
- Doc: L_opt: Computes the discrepancy between spectral entropy and effective rank.

### run_hierarchy_benchmark (method) `def run_hierarchy_benchmark(model_apex, model_blind, device)`
- Defined: `apex25.py:509`

### main (method) `def main()`
- Defined: `apex25.py:555`

### __init__ (method) `def __init__(self, num_patches, embed_dim)`
- Defined: `apex25.py:90`

### forward (method) `def forward(self, x)`
- Defined: `apex25.py:99`

### __init__ (method) `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)`
- Defined: `apex25.py:107`

### freeze (method) `def freeze(self)`
- Defined: `apex25.py:122`

### unfreeze_mixer_only (method) `def unfreeze_mixer_only(self)`
- Defined: `apex25.py:126`

### forward (method) `def forward(self, x)`
- Defined: `apex25.py:132`

### __init__ (method) `def __init__(self, input_dim, hidden_dim, num_classes)`
- Defined: `apex25.py:145`

### apply_masks (method) `def apply_masks(self)`
- Defined: `apex25.py:159`

### get_sparsity (method) `def get_sparsity(self)`
- Defined: `apex25.py:165`

### forward (method) `def forward(self, x)`
- Defined: `apex25.py:170`

### compute_metrics (method) `def compute_metrics(self, weight)`
- Defined: `apex25.py:181`
- Doc: L_mon: Legacy reporting metric.

### detect_phase_state (method) `def detect_phase_state(self, ratio_history)`
- Defined: `apex25.py:194`

### compute_topology_ratio (method) `def compute_topology_ratio(self, model, extractor, chain_type)`
- Defined: `apex25.py:206`
- Doc: Calculates Topo_R = L_opt / L_mon.

### __init__ (method) `def __init__(self)`
- Defined: `apex25.py:225`

### check_intervention (method) `def check_intervention(self, phase_state, coarse_acc, extractor, current_topo_r)`
- Defined: `apex25.py:230`
- Doc: v15.5 Final: Geometric Mismatch Detection.

### perturb_mixer_targeted (method) `def perturb_mixer_targeted(self, extractor)`
- Defined: `apex25.py:277`
- Doc: v15.5: Targeted Spectral Surgery.

### __init__ (method) `def __init__(self, device, extractor_apex, extractor_blind)`
- Defined: `apex25.py:317`

### _preprocess_batch (method) `def _preprocess_batch(self, x, extractor)`
- Defined: `apex25.py:325`

### get_curriculum_dataset (method) `def get_curriculum_dataset(self, cycle)`
- Defined: `apex25.py:329`

### train_single_chain (method) `def train_single_chain(self, model, cycle, chain_type)`
- Defined: `apex25.py:335`

### __getitem__ (method) `def __getitem__(self, index)`
- Defined: `apex25.py:505`

### evaluate (method) `def evaluate(model, extractor, name)`
- Defined: `apex25.py:521`

## apex26.py

### set_seed (function) `def set_seed(seed)`
- Defined: `apex26.py:43`
- Doc: Ensure full reproducibility across runs

### compute_spectral_loss (method) `def compute_spectral_loss(W)`
- Defined: `apex26.py:265`
- Doc: Optimization Objective for Spectral Control (L_opt)

### run_hierarchy_benchmark (method) `def run_hierarchy_benchmark(model_apex, model_blind, feature_extractor_apex, feature_extractor_blind, device)`
- Defined: `apex26.py:490`
- Doc: Run hierarchy stress test to validate inductive bias transfer

### parse_args (method) `def parse_args()`
- Defined: `apex26.py:1170`

### main (method) `def main()`
- Defined: `apex26.py:1181`
- Doc: Main execution function

### __init__ (method) `def __init__(self, num_patches, embed_dim)`
- Defined: `apex26.py:91`

### _init_weights (method) `def _init_weights(self)`
- Defined: `apex26.py:114`
- Doc: Initialize weights for stable training

### forward (method) `def forward(self, x)`
- Defined: `apex26.py:128`
- Doc: Input:  [B, num_patches, embed_dim]

### __init__ (method) `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)`
- Defined: `apex26.py:148`

### freeze (method) `def freeze(self)`
- Defined: `apex26.py:177`
- Doc: Freeze all parameters for transfer learning

### unfreeze_mixer_only (method) `def unfreeze_mixer_only(self)`
- Defined: `apex26.py:183`
- Doc: Unfreeze only the mixer parameters for fine-tuning

### forward (method) `def forward(self, x)`
- Defined: `apex26.py:190`
- Doc: Input:  [B, C, H, W]

### __init__ (method) `def __init__(self, input_dim, hidden_dim, num_classes, num_superclasses)`
- Defined: `apex26.py:209`

### apply_masks (method) `def apply_masks(self)`
- Defined: `apex26.py:232`
- Doc: Apply sparsity masks to weights

### get_sparsity (method) `def get_sparsity(self)`
- Defined: `apex26.py:239`
- Doc: Calculate overall sparsity percentage

### forward (method) `def forward(self, x)`
- Defined: `apex26.py:245`
- Doc: Input:  [B, input_dim]

### __init__ (method) `def __init__(self, epsilon)`
- Defined: `apex26.py:283`

### compute_metrics (method) `def compute_metrics(self, weight)`
- Defined: `apex26.py:286`
- Doc: Compute spectral coherence metrics

### __init__ (method) `def __init__(self, target_coarse_v, stagnation_limit, mixer_noise_scale, dominant_energy_threshold)`
- Defined: `apex26.py:304`

### detect_phase_state (method) `def detect_phase_state(self, ratio_history, phase_window, phase_std_dev_limit)`
- Defined: `apex26.py:314`
- Doc: Detect phase state based on topology ratio history

### check_intervention (method) `def check_intervention(self, phase_state, coarse_acc, extractor, current_topo_r, geo_window)`
- Defined: `apex26.py:332`
- Doc: Check if intervention is needed based on geometric mismatch detection

### perturb_mixer_targeted (method) `def perturb_mixer_targeted(self, extractor)`
- Defined: `apex26.py:377`
- Doc: Targeted Spectral Surgery: Inject noise in the nullspace of dominant subspace

### __init__ (method) `def __init__(self, device, target_L)`
- Defined: `apex26.py:412`

### apply_rank_capping (method) `def apply_rank_capping(self, model, layer_name, keep_ratio)`
- Defined: `apex26.py:417`
- Doc: Apply rank capping shock to prevent over-specialization

### create_offspring (method) `def create_offspring(self, parent_state, data_loader, feature_extractor, lambda_taxonomic, learning_rate)`
- Defined: `apex26.py:434`
- Doc: Create refined offspring through gradient-based inheritance

### __getitem__ (method) `def __getitem__(self, index)`
- Defined: `apex26.py:485`

### evaluate (method) `def evaluate(model, extractor, name)`
- Defined: `apex26.py:503`

### __init__ (method) `def __init__(self, device, output_dir)`
- Defined: `apex26.py:546`

### load_data (method) `def load_data(self, cycle, batch_size)`
- Defined: `apex26.py:572`
- Doc: Load curriculum dataset based on evolutionary cycle

### train_model (method) `def train_model(self, model, cycle, chain_type, feature_extractor)`
- Defined: `apex26.py:614`
- Doc: Train model with evolutionary pressure and hierarchical learning

### compute_topology_ratio (method) `def compute_topology_ratio(self, model, extractor, chain_type)`
- Defined: `apex26.py:837`
- Doc: Calculates Topo_R = L_opt / L_mon.

### run_evolution (method) `def run_evolution(self, num_iterations, num_seeds, early_stop_patience)`
- Defined: `apex26.py:855`
- Doc: Run full evolutionary experiment with statistical validation

### _save_results (method) `def _save_results(self, all_results, best_overall, hierarchy_delta)`
- Defined: `apex26.py:1027`
- Doc: Save results to files

### _plot_results (method) `def _plot_results(self, all_results)`
- Defined: `apex26.py:1081`
- Doc: Create publication-quality plots

## apex27.py

### set_seed (function) `def set_seed(seed)`
- Defined: `apex27.py:38`
- Doc: Ensure full reproducibility across runs

### compute_spectral_loss (method) `def compute_spectral_loss(W)`
- Defined: `apex27.py:217`
- Doc: Optimization Objective for Spectral Control (L_opt)

### parse_args (method) `def parse_args()`
- Defined: `apex27.py:900`

### main (method) `def main()`
- Defined: `apex27.py:914`

### __init__ (method) `def __init__(self, num_patches, embed_dim)`
- Defined: `apex27.py:85`

### _init_weights (method) `def _init_weights(self)`
- Defined: `apex27.py:104`

### forward (method) `def forward(self, x)`
- Defined: `apex27.py:116`

### __init__ (method) `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)`
- Defined: `apex27.py:126`

### freeze (method) `def freeze(self)`
- Defined: `apex27.py:151`

### unfreeze_mixer_only (method) `def unfreeze_mixer_only(self)`
- Defined: `apex27.py:156`

### forward (method) `def forward(self, x)`
- Defined: `apex27.py:162`

### __init__ (method) `def __init__(self, input_dim, hidden_dim, num_classes, num_superclasses)`
- Defined: `apex27.py:172`

### apply_masks (method) `def apply_masks(self)`
- Defined: `apex27.py:192`

### get_sparsity (method) `def get_sparsity(self)`
- Defined: `apex27.py:198`

### forward (method) `def forward(self, x)`
- Defined: `apex27.py:203`

### __init__ (method) `def __init__(self, window_size, percentile_trigger)`
- Defined: `apex27.py:238`

### update (method) `def update(self, value)`
- Defined: `apex27.py:243`

### is_stagnant (method) `def is_stagnant(self, current_val)`
- Defined: `apex27.py:246`

### __init__ (method) `def __init__(self, epsilon)`
- Defined: `apex27.py:257`

### compute_metrics (method) `def compute_metrics(self, weight)`
- Defined: `apex27.py:260`

### __init__ (method) `def __init__(self, dynamic_threshold, stagnation_limit, mixer_noise_scale, dominant_energy_threshold, enable_surgery)`
- Defined: `apex27.py:280`

### check_intervention (method) `def check_intervention(self, coarse_acc, extractor, current_topo_r, geo_window, alpha)`
- Defined: `apex27.py:291`

### perturb_mixer_targeted (method) `def perturb_mixer_targeted(self, extractor)`
- Defined: `apex27.py:336`
- Doc: Targeted Spectral Surgery: Inject noise in the nullspace of dominant subspace

### __init__ (method) `def __init__(self, device)`
- Defined: `apex27.py:371`

### create_offspring (method) `def create_offspring(self, parent_state, data_loader, feature_extractor, lambda_taxonomic, learning_rate)`
- Defined: `apex27.py:374`
- Doc: Create refined offspring through gradient-based inheritance

### __init__ (method) `def __init__(self, device, output_dir, enable_surgery, enable_taxonomy)`
- Defined: `apex27.py:418`

### load_data (method) `def load_data(self, cycle, batch_size)`
- Defined: `apex27.py:441`

### train_model (method) `def train_model(self, model, cycle, chain_type, feature_extractor)`
- Defined: `apex27.py:479`

### compute_topology_ratio (method) `def compute_topology_ratio(self, model, extractor, chain_type)`
- Defined: `apex27.py:683`

### run_refinement (method) `def run_refinement(self, num_iterations, num_seeds, early_stop_patience)`
- Defined: `apex27.py:696`

### _save_results (method) `def _save_results(self, all_results)`
- Defined: `apex27.py:809`

### _plot_results_v18 (method) `def _plot_results_v18(self, all_results)`
- Defined: `apex27.py:815`

## apex28.py

### set_seed (function) `def set_seed(seed)`
- Defined: `apex28.py:47`
- Doc: Ensure full reproducibility across runs

### compute_spectral_loss (method) `def compute_spectral_loss(W)`
- Defined: `apex28.py:269`
- Doc: Optimization Objective for Spectral Control (L_opt)

### run_hierarchy_benchmark (method) `def run_hierarchy_benchmark(model_apex, model_blind, feature_extractor_apex, feature_extractor_blind, device)`
- Defined: `apex28.py:494`
- Doc: Run hierarchy stress test to validate inductive bias transfer

### visualize_singular_values (method) `def visualize_singular_values(model_apex, model_blind, feature_extractor_apex, feature_extractor_blind, device, output_dir)`
- Defined: `apex28.py:548`
- Doc: Generate publication-quality singular value visualizations

### parse_args (method) `def parse_args()`
- Defined: `apex28.py:1270`

### main (method) `def main()`
- Defined: `apex28.py:1282`
- Doc: Main execution function

### __init__ (method) `def __init__(self, num_patches, embed_dim)`
- Defined: `apex28.py:95`

### _init_weights (method) `def _init_weights(self)`
- Defined: `apex28.py:118`
- Doc: Initialize weights for stable training

### forward (method) `def forward(self, x)`
- Defined: `apex28.py:132`
- Doc: Input:  [B, num_patches, embed_dim]

### __init__ (method) `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)`
- Defined: `apex28.py:152`

### freeze (method) `def freeze(self)`
- Defined: `apex28.py:181`
- Doc: Freeze all parameters for transfer learning

### unfreeze_mixer_only (method) `def unfreeze_mixer_only(self)`
- Defined: `apex28.py:187`
- Doc: Unfreeze only the mixer parameters for fine-tuning

### forward (method) `def forward(self, x)`
- Defined: `apex28.py:194`
- Doc: Input:  [B, C, H, W]

### __init__ (method) `def __init__(self, input_dim, hidden_dim, num_classes, num_superclasses)`
- Defined: `apex28.py:213`

### apply_masks (method) `def apply_masks(self)`
- Defined: `apex28.py:236`
- Doc: Apply sparsity masks to weights

### get_sparsity (method) `def get_sparsity(self)`
- Defined: `apex28.py:243`
- Doc: Calculate overall sparsity percentage

### forward (method) `def forward(self, x)`
- Defined: `apex28.py:249`
- Doc: Input:  [B, input_dim]

### __init__ (method) `def __init__(self, epsilon)`
- Defined: `apex28.py:287`

### compute_metrics (method) `def compute_metrics(self, weight)`
- Defined: `apex28.py:290`
- Doc: Compute spectral coherence metrics

### get_singular_values (method) `def get_singular_values(self, weight)`
- Defined: `apex28.py:306`
- Doc: Get singular values for visualization

### __init__ (method) `def __init__(self, semantic_plasticity_threshold, stagnation_limit, mixer_noise_scale, dominant_energy_threshold, geo_window)`
- Defined: `apex28.py:318`

### compute_semantic_plasticity_ratio (method) `def compute_semantic_plasticity_ratio(self)`
- Defined: `apex28.py:330`
- Doc: Compute ratio of semantic gain to structural change

### detect_intervention_need (method) `def detect_intervention_need(self, phase_state, extractor)`
- Defined: `apex28.py:343`
- Doc: Determine if intervention is needed using adaptive criteria

### update_history (method) `def update_history(self, topo_ratio, coarse_acc)`
- Defined: `apex28.py:365`
- Doc: Update history for adaptive control

### perturb_mixer_targeted (method) `def perturb_mixer_targeted(self, extractor)`
- Defined: `apex28.py:382`
- Doc: Targeted Spectral Surgery: Inject noise in the nullspace of dominant subspace

### __init__ (method) `def __init__(self, device)`
- Defined: `apex28.py:417`

### apply_rank_capping (method) `def apply_rank_capping(self, model, layer_name, keep_ratio)`
- Defined: `apex28.py:421`
- Doc: Apply rank capping shock to prevent over-specialization

### create_refined_model (method) `def create_refined_model(self, parent_state, data_loader, feature_extractor, lambda_taxonomic, learning_rate)`
- Defined: `apex28.py:438`
- Doc: Create refined model through gradient-based inheritance

### __getitem__ (method) `def __getitem__(self, index)`
- Defined: `apex28.py:489`

### evaluate (method) `def evaluate(model, extractor, name)`
- Defined: `apex28.py:507`

### get_singular_values (method) `def get_singular_values(model, extractor, name)`
- Defined: `apex28.py:556`
- Doc: Get singular values from model weights

### __init__ (method) `def __init__(self, device, output_dir)`
- Defined: `apex28.py:613`

### load_data (method) `def load_data(self, cycle, batch_size)`
- Defined: `apex28.py:639`
- Doc: Load curriculum dataset based on refinement cycle

### train_model (method) `def train_model(self, model, cycle, chain_type, feature_extractor)`
- Defined: `apex28.py:681`
- Doc: Train model with iterative refinement and hierarchical learning

### detect_phase_state (method) `def detect_phase_state(self, ratio_history, phase_window, phase_std_dev_limit)`
- Defined: `apex28.py:908`
- Doc: Detect phase state based on topology ratio history

### compute_topology_ratio (method) `def compute_topology_ratio(self, model, extractor, chain_type)`
- Defined: `apex28.py:926`
- Doc: Calculates Topo_R = L_opt / L_mon.

### run_refinement (method) `def run_refinement(self, num_iterations, num_seeds, early_stop_patience)`
- Defined: `apex28.py:944`
- Doc: Run full iterative refinement experiment with statistical validation

### _save_results (method) `def _save_results(self, all_results, best_overall, hierarchy_delta)`
- Defined: `apex28.py:1126`
- Doc: Save results to files

### _plot_results (method) `def _plot_results(self, all_results)`
- Defined: `apex28.py:1181`
- Doc: Create publication-quality plots

## apex29.py

### set_seed (function) `def set_seed(seed)`
- Defined: `apex29.py:43`
- Doc: Ensure full reproducibility across runs

### compute_spectral_loss (method) `def compute_spectral_loss(W)`
- Defined: `apex29.py:262`
- Doc: Optimization Objective for Spectral Control (L_opt)

### run_hierarchy_benchmark (method) `def run_hierarchy_benchmark(model_apex, model_blind, feature_extractor_apex, feature_extractor_blind, device)`
- Defined: `apex29.py:486`
- Doc: Run hierarchy stress test to validate inductive bias transfer

### visualize_singular_values (method) `def visualize_singular_values(model_apex, model_blind, feature_extractor_apex, feature_extractor_blind, device, output_dir)`
- Defined: `apex29.py:543`
- Doc: Generate publication-quality singular value visualizations

### parse_args (method) `def parse_args()`
- Defined: `apex29.py:1244`

### main (method) `def main()`
- Defined: `apex29.py:1255`
- Doc: Main execution function

### __init__ (method) `def __init__(self, num_patches, embed_dim)`
- Defined: `apex29.py:88`

### _init_weights (method) `def _init_weights(self)`
- Defined: `apex29.py:111`
- Doc: Initialize weights for stable training

### forward (method) `def forward(self, x)`
- Defined: `apex29.py:125`
- Doc: Input:  [B, num_patches, embed_dim]

### __init__ (method) `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)`
- Defined: `apex29.py:145`

### freeze (method) `def freeze(self)`
- Defined: `apex29.py:174`
- Doc: Freeze all parameters for transfer learning

### unfreeze_mixer_only (method) `def unfreeze_mixer_only(self)`
- Defined: `apex29.py:180`
- Doc: Unfreeze only the mixer parameters for fine-tuning

### forward (method) `def forward(self, x)`
- Defined: `apex29.py:187`
- Doc: Input:  [B, C, H, W]

### __init__ (method) `def __init__(self, input_dim, hidden_dim, num_classes, num_superclasses)`
- Defined: `apex29.py:206`

### apply_masks (method) `def apply_masks(self)`
- Defined: `apex29.py:229`
- Doc: Apply sparsity masks to weights

### get_sparsity (method) `def get_sparsity(self)`
- Defined: `apex29.py:236`
- Doc: Calculate overall sparsity percentage

### forward (method) `def forward(self, x)`
- Defined: `apex29.py:242`
- Doc: Input:  [B, input_dim]

### __init__ (method) `def __init__(self, epsilon)`
- Defined: `apex29.py:280`

### compute_metrics (method) `def compute_metrics(self, weight)`
- Defined: `apex29.py:283`
- Doc: Compute spectral coherence metrics

### get_singular_values (method) `def get_singular_values(self, weight)`
- Defined: `apex29.py:299`
- Doc: Get singular values for visualization

### __init__ (method) `def __init__(self, semantic_plasticity_threshold, stagnation_limit, mixer_noise_scale, dominant_energy_threshold, geo_window)`
- Defined: `apex29.py:311`

### compute_semantic_plasticity_ratio (method) `def compute_semantic_plasticity_ratio(self)`
- Defined: `apex29.py:323`
- Doc: Compute ratio of semantic gain to structural change

### detect_intervention_need (method) `def detect_intervention_need(self, phase_state, extractor)`
- Defined: `apex29.py:336`
- Doc: Determine if intervention is needed using adaptive criteria

### update_history (method) `def update_history(self, topo_ratio, coarse_acc)`
- Defined: `apex29.py:358`
- Doc: Update history for adaptive control

### perturb_mixer_targeted (method) `def perturb_mixer_targeted(self, extractor)`
- Defined: `apex29.py:375`
- Doc: Targeted Spectral Surgery: Inject noise in the nullspace of dominant subspace

### __init__ (method) `def __init__(self, device)`
- Defined: `apex29.py:410`

### apply_rank_capping (method) `def apply_rank_capping(self, model, layer_name, keep_ratio)`
- Defined: `apex29.py:414`
- Doc: Apply rank capping shock to prevent over-specialization

### create_refined_model (method) `def create_refined_model(self, parent_state, data_loader, feature_extractor, lambda_taxonomic, learning_rate)`
- Defined: `apex29.py:430`
- Doc: Create refined model through gradient-based inheritance

### __getitem__ (method) `def __getitem__(self, index)`
- Defined: `apex29.py:481`

### evaluate (method) `def evaluate(model, extractor, name)`
- Defined: `apex29.py:499`

### get_singular_values (method) `def get_singular_values(model, extractor, name)`
- Defined: `apex29.py:551`
- Doc: Get singular values from model weights

### __init__ (method) `def __init__(self, device, output_dir)`
- Defined: `apex29.py:608`

### load_data (method) `def load_data(self, cycle, batch_size)`
- Defined: `apex29.py:634`
- Doc: Load curriculum dataset based on refinement cycle

### train_model (method) `def train_model(self, model, cycle, chain_type, feature_extractor)`
- Defined: `apex29.py:674`
- Doc: Train model with iterative refinement and hierarchical learning

### detect_phase_state (method) `def detect_phase_state(self, ratio_history, phase_window, phase_std_dev_limit)`
- Defined: `apex29.py:894`
- Doc: Detect phase state based on topology ratio history

### compute_topology_ratio (method) `def compute_topology_ratio(self, model, extractor, chain_type)`
- Defined: `apex29.py:912`
- Doc: Calculates Topo_R = L_opt / L_mon.

### run_refinement (method) `def run_refinement(self, num_iterations, num_seeds, early_stop_patience)`
- Defined: `apex29.py:931`
- Doc: Run full iterative refinement experiment with statistical validation

### _save_results (method) `def _save_results(self, all_results, best_overall, hierarchy_delta)`
- Defined: `apex29.py:1109`
- Doc: Save results to files

### _plot_results (method) `def _plot_results(self, all_results)`
- Defined: `apex29.py:1164`
- Doc: Create publication-quality plots

## apex30.py

### set_seed (function) `def set_seed(seed)`
- Defined: `apex30.py:43`
- Doc: Ensure full reproducibility across runs

### compute_spectral_loss (method) `def compute_spectral_loss(W)`
- Defined: `apex30.py:232`
- Doc: Optimization Objective for Spectral Control (L_opt)

### run_hierarchy_benchmark (method) `def run_hierarchy_benchmark(model_apex, model_blind, feature_extractor_apex, feature_extractor_blind, device)`
- Defined: `apex30.py:464`

### visualize_singular_values (method) `def visualize_singular_values(model_apex, model_blind, feature_extractor_apex, feature_extractor_blind, device, output_dir, monitor)`
- Defined: `apex30.py:510`
- Doc: v18.1 FIX: Now accepts 'monitor' explicitly.

### parse_args (method) `def parse_args()`
- Defined: `apex30.py:1062`

### main (method) `def main()`
- Defined: `apex30.py:1071`

### __init__ (method) `def __init__(self, num_patches, embed_dim)`
- Defined: `apex30.py:89`

### _init_weights (method) `def _init_weights(self)`
- Defined: `apex30.py:111`

### forward (method) `def forward(self, x)`
- Defined: `apex30.py:134`

### __init__ (method) `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)`
- Defined: `apex30.py:143`

### freeze (method) `def freeze(self)`
- Defined: `apex30.py:164`

### unfreeze_mixer_only (method) `def unfreeze_mixer_only(self)`
- Defined: `apex30.py:169`

### forward (method) `def forward(self, x)`
- Defined: `apex30.py:175`

### __init__ (method) `def __init__(self, input_dim, hidden_dim, num_classes, num_superclasses)`
- Defined: `apex30.py:187`

### apply_masks (method) `def apply_masks(self)`
- Defined: `apex30.py:207`

### get_sparsity (method) `def get_sparsity(self)`
- Defined: `apex30.py:213`

### forward (method) `def forward(self, x)`
- Defined: `apex30.py:218`

### __init__ (method) `def __init__(self, epsilon)`
- Defined: `apex30.py:250`

### compute_metrics (method) `def compute_metrics(self, weight)`
- Defined: `apex30.py:253`

### get_singular_values (method) `def get_singular_values(self, weight)`
- Defined: `apex30.py:268`

### __init__ (method) `def __init__(self, semantic_plasticity_threshold, stagnation_limit, mixer_noise_scale, dominant_energy_threshold, geo_window)`
- Defined: `apex30.py:279`

### compute_semantic_plasticity_ratio (method) `def compute_semantic_plasticity_ratio(self)`
- Defined: `apex30.py:295`
- Doc: Compute ratio of semantic gain to structural change

### detect_intervention_need (method) `def detect_intervention_need(self, phase_state, extractor)`
- Defined: `apex30.py:308`
- Doc: Determine if intervention is needed using adaptive criteria.

### update_history (method) `def update_history(self, topo_ratio, coarse_acc)`
- Defined: `apex30.py:348`
- Doc: Update history for adaptive control

### perturb_mixer_targeted (method) `def perturb_mixer_targeted(self, extractor)`
- Defined: `apex30.py:363`
- Doc: Targeted Spectral Surgery: Inject noise in the nullspace of dominant subspace

### __init__ (method) `def __init__(self, device)`
- Defined: `apex30.py:399`

### apply_rank_capping (method) `def apply_rank_capping(self, model, layer_name, keep_ratio)`
- Defined: `apex30.py:403`

### create_refined_model (method) `def create_refined_model(self, parent_state, data_loader, feature_extractor, lambda_taxonomic, learning_rate)`
- Defined: `apex30.py:419`

### __getitem__ (method) `def __getitem__(self, index)`
- Defined: `apex30.py:460`

### evaluate (method) `def evaluate(model, extractor)`
- Defined: `apex30.py:471`

### get_singular_values (method) `def get_singular_values(model, extractor)`
- Defined: `apex30.py:521`

### __init__ (method) `def __init__(self, device, output_dir)`
- Defined: `apex30.py:571`

### load_data (method) `def load_data(self, cycle, batch_size)`
- Defined: `apex30.py:593`

### train_model (method) `def train_model(self, model, cycle, chain_type, feature_extractor)`
- Defined: `apex30.py:617`

### detect_phase_state (method) `def detect_phase_state(self, ratio_history, phase_window, phase_std_dev_limit)`
- Defined: `apex30.py:797`

### compute_topology_ratio (method) `def compute_topology_ratio(self, model, extractor, chain_type)`
- Defined: `apex30.py:813`

### run_refinement (method) `def run_refinement(self, num_iterations, num_seeds, early_stop_patience)`
- Defined: `apex30.py:828`

### _save_results (method) `def _save_results(self, all_results, best_overall, hierarchy_delta)`
- Defined: `apex30.py:956`

### _plot_results (method) `def _plot_results(self, all_results, best_overall)`
- Defined: `apex30.py:983`
- Doc: v18.1 FIX: Explicitly accepts best_overall to fix scope bug.

## apex31.py

### set_seed (function) `def set_seed(seed)`
- Defined: `apex31.py:45`
- Doc: Ensure full reproducibility across runs

### compute_spectral_loss (method) `def compute_spectral_loss(W)`
- Defined: `apex31.py:234`
- Doc: Optimization Objective for Spectral Control (L_opt)

### run_hierarchy_benchmark (method) `def run_hierarchy_benchmark(model_apex, model_blind, feature_extractor_apex, feature_extractor_blind, device)`
- Defined: `apex31.py:475`

### visualize_singular_values (method) `def visualize_singular_values(model_apex, model_blind, feature_extractor_apex, feature_extractor_blind, device, output_dir, monitor)`
- Defined: `apex31.py:521`
- Doc: v18.1 FIX: Now accepts 'monitor' explicitly.

### parse_args (method) `def parse_args()`
- Defined: `apex31.py:1073`

### main (method) `def main()`
- Defined: `apex31.py:1082`

### __init__ (method) `def __init__(self, num_patches, embed_dim)`
- Defined: `apex31.py:91`

### _init_weights (method) `def _init_weights(self)`
- Defined: `apex31.py:113`

### forward (method) `def forward(self, x)`
- Defined: `apex31.py:136`

### __init__ (method) `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)`
- Defined: `apex31.py:145`

### freeze (method) `def freeze(self)`
- Defined: `apex31.py:166`

### unfreeze_mixer_only (method) `def unfreeze_mixer_only(self)`
- Defined: `apex31.py:171`

### forward (method) `def forward(self, x)`
- Defined: `apex31.py:177`

### __init__ (method) `def __init__(self, input_dim, hidden_dim, num_classes, num_superclasses)`
- Defined: `apex31.py:189`

### apply_masks (method) `def apply_masks(self)`
- Defined: `apex31.py:209`

### get_sparsity (method) `def get_sparsity(self)`
- Defined: `apex31.py:215`

### forward (method) `def forward(self, x)`
- Defined: `apex31.py:220`

### __init__ (method) `def __init__(self, epsilon)`
- Defined: `apex31.py:252`

### compute_metrics (method) `def compute_metrics(self, weight)`
- Defined: `apex31.py:255`

### get_singular_values (method) `def get_singular_values(self, weight)`
- Defined: `apex31.py:270`

### __init__ (method) `def __init__(self, semantic_plasticity_threshold, stagnation_limit, mixer_noise_scale, dominant_energy_threshold, geo_window)`
- Defined: `apex31.py:281`

### compute_semantic_plasticity_ratio (method) `def compute_semantic_plasticity_ratio(self)`
- Defined: `apex31.py:297`
- Doc: Compute ratio of semantic gain to structural change

### detect_intervention_need (method) `def detect_intervention_need(self, phase_state, extractor)`
- Defined: `apex31.py:310`
- Doc: Determine if intervention is needed using adaptive criteria.

### update_history (method) `def update_history(self, topo_ratio, coarse_acc)`
- Defined: `apex31.py:354`
- Doc: Update history for adaptive control

### perturb_mixer_targeted (method) `def perturb_mixer_targeted(self, extractor)`
- Defined: `apex31.py:369`
- Doc: Targeted Spectral Surgery: Inject noise in the nullspace of dominant subspace

### __init__ (method) `def __init__(self, device)`
- Defined: `apex31.py:406`

### apply_rank_capping (method) `def apply_rank_capping(self, model, layer_name, keep_ratio)`
- Defined: `apex31.py:410`

### create_refined_model (method) `def create_refined_model(self, parent_state, data_loader, feature_extractor, lambda_taxonomic, learning_rate)`
- Defined: `apex31.py:430`

### __getitem__ (method) `def __getitem__(self, index)`
- Defined: `apex31.py:471`

### evaluate (method) `def evaluate(model, extractor)`
- Defined: `apex31.py:482`

### get_singular_values (method) `def get_singular_values(model, extractor)`
- Defined: `apex31.py:532`

### __init__ (method) `def __init__(self, device, output_dir)`
- Defined: `apex31.py:582`

### load_data (method) `def load_data(self, cycle, batch_size)`
- Defined: `apex31.py:604`

### train_model (method) `def train_model(self, model, cycle, chain_type, feature_extractor)`
- Defined: `apex31.py:628`

### detect_phase_state (method) `def detect_phase_state(self, ratio_history, phase_window, phase_std_dev_limit)`
- Defined: `apex31.py:808`

### compute_topology_ratio (method) `def compute_topology_ratio(self, model, extractor, chain_type)`
- Defined: `apex31.py:824`

### run_refinement (method) `def run_refinement(self, num_iterations, num_seeds, early_stop_patience)`
- Defined: `apex31.py:839`

### _save_results (method) `def _save_results(self, all_results, best_overall, hierarchy_delta)`
- Defined: `apex31.py:967`

### _plot_results (method) `def _plot_results(self, all_results, best_overall)`
- Defined: `apex31.py:994`
- Doc: v18.1 FIX: Explicitly accepts best_overall to fix scope bug.

## apex32.py

### compute_spectral_loss (function) `def compute_spectral_loss(W)`
- Defined: `apex32.py:76`
- Doc: L_opt: Optimization Objective for Structural Control.

### run_hierarchy_benchmark (method) `def run_hierarchy_benchmark(model_apex, model_blind, device)`
- Defined: `apex32.py:409`

### main (method) `def main()`
- Defined: `apex32.py:455`

### __init__ (method) `def __init__(self, num_patches, embed_dim)`
- Defined: `apex32.py:93`

### forward (method) `def forward(self, x)`
- Defined: `apex32.py:102`

### __init__ (method) `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)`
- Defined: `apex32.py:110`

### freeze (method) `def freeze(self)`
- Defined: `apex32.py:126`

### unfreeze_mixer_only (method) `def unfreeze_mixer_only(self)`
- Defined: `apex32.py:130`

### forward (method) `def forward(self, x)`
- Defined: `apex32.py:136`

### __init__ (method) `def __init__(self, input_dim, hidden_dim, num_classes)`
- Defined: `apex32.py:149`

### apply_masks (method) `def apply_masks(self)`
- Defined: `apex32.py:164`
- Doc: Zero out weights based on masks. Runs on device (CUDA).

### get_sparsity (method) `def get_sparsity(self)`
- Defined: `apex32.py:171`

### forward (method) `def forward(self, x)`
- Defined: `apex32.py:176`

### compute_metrics (method) `def compute_metrics(self, weight)`
- Defined: `apex32.py:188`

### detect_phase_state (method) `def detect_phase_state(self, ratio_history)`
- Defined: `apex32.py:200`

### compute_topology_ratio (method) `def compute_topology_ratio(self, model, extractor, chain_type)`
- Defined: `apex32.py:211`

### __init__ (method) `def __init__(self, device, extractor_apex, extractor_blind)`
- Defined: `apex32.py:227`

### _preprocess_batch (method) `def _preprocess_batch(self, x, extractor)`
- Defined: `apex32.py:234`

### get_curriculum_dataset (method) `def get_curriculum_dataset(self, cycle)`
- Defined: `apex32.py:238`

### train_single_chain (method) `def train_single_chain(self, model, cycle, chain_type)`
- Defined: `apex32.py:244`

### __getitem__ (method) `def __getitem__(self, index)`
- Defined: `apex32.py:405`

### evaluate (method) `def evaluate(model, extractor, name)`
- Defined: `apex32.py:422`

## apex33.py

### set_seed (function) `def set_seed(seed)`
- Defined: `apex33.py:41`
- Doc: Ensure full reproducibility across runs

### compute_spectral_loss (method) `def compute_spectral_loss(W)`
- Defined: `apex33.py:190`

### run_hierarchy_benchmark (method) `def run_hierarchy_benchmark(model_apex, model_blind, feature_extractor_apex, feature_extractor_blind, device)`
- Defined: `apex33.py:807`

### visualize_singular_values (method) `def visualize_singular_values(model_apex, model_blind, feature_extractor_apex, feature_extractor_blind, device, output_dir, monitor)`
- Defined: `apex33.py:845`

### parse_args (method) `def parse_args()`
- Defined: `apex33.py:894`

### main (method) `def main()`
- Defined: `apex33.py:903`

### __init__ (method) `def __init__(self, num_patches, embed_dim)`
- Defined: `apex33.py:74`

### _init_weights (method) `def _init_weights(self)`
- Defined: `apex33.py:92`

### forward (method) `def forward(self, x)`
- Defined: `apex33.py:110`

### __init__ (method) `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)`
- Defined: `apex33.py:119`

### freeze (method) `def freeze(self)`
- Defined: `apex33.py:135`

### unfreeze_mixer_only (method) `def unfreeze_mixer_only(self)`
- Defined: `apex33.py:138`

### forward (method) `def forward(self, x)`
- Defined: `apex33.py:143`

### __init__ (method) `def __init__(self, input_dim, hidden_dim, num_classes, num_superclasses)`
- Defined: `apex33.py:152`

### apply_masks (method) `def apply_masks(self)`
- Defined: `apex33.py:167`

### get_sparsity (method) `def get_sparsity(self)`
- Defined: `apex33.py:173`

### forward (method) `def forward(self, x)`
- Defined: `apex33.py:178`

### __init__ (method) `def __init__(self, epsilon)`
- Defined: `apex33.py:202`

### compute_metrics (method) `def compute_metrics(self, weight)`
- Defined: `apex33.py:205`

### get_singular_values (method) `def get_singular_values(self, weight)`
- Defined: `apex33.py:219`

### __init__ (method) `def __init__(self, semantic_plasticity_threshold, stagnation_limit, mixer_noise_scale, dominant_energy_threshold, geo_window)`
- Defined: `apex33.py:229`

### compute_semantic_plasticity_ratio (method) `def compute_semantic_plasticity_ratio(self)`
- Defined: `apex33.py:242`

### detect_intervention_need (method) `def detect_intervention_need(self, phase_state, extractor)`
- Defined: `apex33.py:250`

### update_history (method) `def update_history(self, topo_ratio, coarse_acc)`
- Defined: `apex33.py:273`

### perturb_mixer_targeted (method) `def perturb_mixer_targeted(self, extractor)`
- Defined: `apex33.py:284`

### __init__ (method) `def __init__(self, device)`
- Defined: `apex33.py:311`

### create_refined_model (method) `def create_refined_model(self, parent_state, data_loader, feature_extractor, lambda_taxonomic, learning_rate)`
- Defined: `apex33.py:315`

### __init__ (method) `def __init__(self, device, output_dir)`
- Defined: `apex33.py:344`

### load_data (method) `def load_data(self, cycle, batch_size)`
- Defined: `apex33.py:367`

### train_model (method) `def train_model(self, model, cycle, chain_type, feature_extractor)`
- Defined: `apex33.py:385`

### detect_phase_state (method) `def detect_phase_state(self, ratio_history, phase_window, phase_std_dev_limit)`
- Defined: `apex33.py:570`

### compute_topology_ratio (method) `def compute_topology_ratio(self, model, extractor, chain_type)`
- Defined: `apex33.py:580`

### run_refinement (method) `def run_refinement(self, num_iterations, num_seeds, early_stop_patience)`
- Defined: `apex33.py:592`

### _save_results (method) `def _save_results(self, all_results, best_overall, hierarchy_delta)`
- Defined: `apex33.py:712`

### _plot_results (method) `def _plot_results(self, all_results, best_overall)`
- Defined: `apex33.py:738`

### __getitem__ (method) `def __getitem__(self, index)`
- Defined: `apex33.py:803`

### evaluate (method) `def evaluate(model, extractor)`
- Defined: `apex33.py:814`

### get_singular_values (method) `def get_singular_values(model, extractor)`
- Defined: `apex33.py:851`

## apex34.py

### set_seed (function) `def set_seed(seed)`
- Defined: `apex34.py:41`
- Doc: Ensure full reproducibility across runs

### compute_spectral_loss (method) `def compute_spectral_loss(W)`
- Defined: `apex34.py:190`

### run_hierarchy_benchmark (method) `def run_hierarchy_benchmark(model_apex, model_blind, feature_extractor_apex, feature_extractor_blind, device)`
- Defined: `apex34.py:807`

### visualize_singular_values (method) `def visualize_singular_values(model_apex, model_blind, feature_extractor_apex, feature_extractor_blind, device, output_dir, monitor)`
- Defined: `apex34.py:845`

### main (method) `def main()`
- Defined: `apex34.py:894`

### __init__ (method) `def __init__(self, num_patches, embed_dim)`
- Defined: `apex34.py:74`

### _init_weights (method) `def _init_weights(self)`
- Defined: `apex34.py:92`

### forward (method) `def forward(self, x)`
- Defined: `apex34.py:110`

### __init__ (method) `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)`
- Defined: `apex34.py:119`

### freeze (method) `def freeze(self)`
- Defined: `apex34.py:135`

### unfreeze_mixer_only (method) `def unfreeze_mixer_only(self)`
- Defined: `apex34.py:138`

### forward (method) `def forward(self, x)`
- Defined: `apex34.py:143`

### __init__ (method) `def __init__(self, input_dim, hidden_dim, num_classes, num_superclasses)`
- Defined: `apex34.py:152`

### apply_masks (method) `def apply_masks(self)`
- Defined: `apex34.py:167`

### get_sparsity (method) `def get_sparsity(self)`
- Defined: `apex34.py:173`

### forward (method) `def forward(self, x)`
- Defined: `apex34.py:178`

### __init__ (method) `def __init__(self, epsilon)`
- Defined: `apex34.py:202`

### compute_metrics (method) `def compute_metrics(self, weight)`
- Defined: `apex34.py:205`

### get_singular_values (method) `def get_singular_values(self, weight)`
- Defined: `apex34.py:219`

### __init__ (method) `def __init__(self, semantic_plasticity_threshold, stagnation_limit, mixer_noise_scale, dominant_energy_threshold, geo_window)`
- Defined: `apex34.py:229`

### compute_semantic_plasticity_ratio (method) `def compute_semantic_plasticity_ratio(self)`
- Defined: `apex34.py:242`

### detect_intervention_need (method) `def detect_intervention_need(self, phase_state, extractor)`
- Defined: `apex34.py:250`

### update_history (method) `def update_history(self, topo_ratio, coarse_acc)`
- Defined: `apex34.py:273`

### perturb_mixer_targeted (method) `def perturb_mixer_targeted(self, extractor)`
- Defined: `apex34.py:284`

### __init__ (method) `def __init__(self, device)`
- Defined: `apex34.py:311`

### create_refined_model (method) `def create_refined_model(self, parent_state, data_loader, feature_extractor, lambda_taxonomic, learning_rate)`
- Defined: `apex34.py:315`

### __init__ (method) `def __init__(self, device, output_dir)`
- Defined: `apex34.py:344`

### load_data (method) `def load_data(self, cycle, batch_size)`
- Defined: `apex34.py:367`

### train_model (method) `def train_model(self, model, cycle, chain_type, feature_extractor)`
- Defined: `apex34.py:385`

### detect_phase_state (method) `def detect_phase_state(self, ratio_history, phase_window, phase_std_dev_limit)`
- Defined: `apex34.py:570`

### compute_topology_ratio (method) `def compute_topology_ratio(self, model, extractor, chain_type)`
- Defined: `apex34.py:580`

### run_refinement (method) `def run_refinement(self, num_iterations, num_seeds, early_stop_patience)`
- Defined: `apex34.py:592`

### _save_results (method) `def _save_results(self, all_results, best_overall, hierarchy_delta)`
- Defined: `apex34.py:712`

### _plot_results (method) `def _plot_results(self, all_results, best_overall)`
- Defined: `apex34.py:738`

### __getitem__ (method) `def __getitem__(self, index)`
- Defined: `apex34.py:803`

### evaluate (method) `def evaluate(model, extractor)`
- Defined: `apex34.py:814`

### get_singular_values (method) `def get_singular_values(model, extractor)`
- Defined: `apex34.py:851`

## apex35.py

### set_seed (function) `def set_seed(seed)`
- Defined: `apex35.py:36`

### main (method) `def main()`
- Defined: `apex35.py:444`

### __init__ (method) `def __init__(self, num_patches, embed_dim)`
- Defined: `apex35.py:68`

### _init_weights (method) `def _init_weights(self)`
- Defined: `apex35.py:86`

### forward (method) `def forward(self, x)`
- Defined: `apex35.py:104`

### __init__ (method) `def __init__(self, embed_dim, num_heads)`
- Defined: `apex35.py:118`

### forward (method) `def forward(self, x)`
- Defined: `apex35.py:136`

### __init__ (method) `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)`
- Defined: `apex35.py:157`

### freeze (method) `def freeze(self)`
- Defined: `apex35.py:173`

### unfreeze_mixer_only (method) `def unfreeze_mixer_only(self)`
- Defined: `apex35.py:176`

### forward (method) `def forward(self, x)`
- Defined: `apex35.py:181`

### __init__ (method) `def __init__(self, input_dim, hidden_dim, num_classes, num_superclasses)`
- Defined: `apex35.py:189`

### apply_masks (method) `def apply_masks(self)`
- Defined: `apex35.py:204`

### get_sparsity (method) `def get_sparsity(self)`
- Defined: `apex35.py:210`

### forward (method) `def forward(self, x)`
- Defined: `apex35.py:215`

### __init__ (method) `def __init__(self, epsilon)`
- Defined: `apex35.py:229`

### inspect (method) `def inspect(self, weight)`
- Defined: `apex35.py:232`

### __init__ (method) `def __init__(self, device, output_dir)`
- Defined: `apex35.py:254`

### load_data (method) `def load_data(self, cycle, batch_size)`
- Defined: `apex35.py:274`

### train_model (method) `def train_model(self, model, cycle, chain_type, feature_extractor)`
- Defined: `apex35.py:292`

### __getitem__ (method) `def __getitem__(self, index)`
- Defined: `apex35.py:440`

### evaluate_safe (method) `def evaluate_safe(model, extractor)`
- Defined: `apex35.py:483`

## app.py

### main (method) `def main()`
- Defined: `app.py:561`

### __init__ (method) `def __init__(self, epsilon_c)`
- Defined: `app.py:52`

### compute_L (method) `def compute_L(self, weight)`
- Defined: `app.py:55`

### __init__ (method) `def __init__(self, sparsity_target)`
- Defined: `app.py:77`

### apply_to_model (method) `def apply_to_model(self, model)`
- Defined: `app.py:81`

### enforce_during_training (method) `def enforce_during_training(self, model)`
- Defined: `app.py:91`

### __init__ (method) `def __init__(self, input_dim, hidden_dim, num_classes)`
- Defined: `app.py:99`

### reduce_input (method) `def reduce_input(self, x)`
- Defined: `app.py:107`

### forward (method) `def forward(self, x)`
- Defined: `app.py:113`

### __init__ (method) `def __init__(self, device, base_target_acc)`
- Defined: `app.py:122`

### load_best_legacy_model (method) `def load_best_legacy_model(self, cycle)`
- Defined: `app.py:137`
- Doc: Load the best model from previous cycle, with fallback to initial seed

### train_base_model_to_target (method) `def train_base_model_to_target(self, hidden_dim, target_acc, test_loader)`
- Defined: `app.py:175`
- Doc: Train a base model to target accuracy

### extract_seed_from_checkpoint (method) `def extract_seed_from_checkpoint(self, checkpoint, expected_hidden_dim)`
- Defined: `app.py:228`
- Doc: Extract seed weights from checkpoint, handling different formats

### extract_seed_weights (method) `def extract_seed_weights(self, model)`
- Defined: `app.py:254`

### inoculate_seed_adaptive (method) `def inoculate_seed_adaptive(self, large_model, seed_weights)`
- Defined: `app.py:260`
- Doc: Adaptive inoculation that handles dimension mismatches

### measure_functional_alignment (method) `def measure_functional_alignment(self, model1, model2, test_loader)`
- Defined: `app.py:288`
- Doc: Measure functional alignment via logit cosine similarity

### progressive_pruning_with_target (method) `def progressive_pruning_with_target(self, model, target_acc, test_loader, max_density)`
- Defined: `app.py:309`
- Doc: Prune while maintaining target accuracy, with density constraint

### execute_resonance_cycle (method) `def execute_resonance_cycle(self, cycle, test_loader, expansion_factor)`
- Defined: `app.py:351`

### run_evolutionary_experiment (method) `def run_evolutionary_experiment(self, num_cycles)`
- Defined: `app.py:484`

## plank.py

### main (method) `def main()`
- Defined: `plank.py:178`

### __init__ (method) `def __init__(self, epsilon_c)`
- Defined: `plank.py:33`

### inspect (method) `def inspect(self, weights)`
- Defined: `plank.py:36`

### __init__ (method) `def __init__(self, in_features, out_features, sparsity_target)`
- Defined: `plank.py:63`

### forward (method) `def forward(self, x, inject_lies)`
- Defined: `plank.py:70`

### apply_black_swan_refraction (method) `def apply_black_swan_refraction(self)`
- Defined: `plank.py:87`
- Doc: Purificación extrema: sparsity 0.0004%

### __init__ (method) `def __init__(self, sparsity_target)`
- Defined: `plank.py:109`

### forward (method) `def forward(self, x, inject_lies)`
- Defined: `plank.py:117`

### __init__ (method) `def __init__(self, model, device)`
- Defined: `plank.py:133`

### train_epoch (method) `def train_epoch(self, dataloader, epoch)`
- Defined: `plank.py:139`

## plank10.py

### main (method) `def main()`
- Defined: `plank10.py:354`

### __init__ (method) `def __init__(self, img_size, patch_size, in_chans, embed_dim)`
- Defined: `plank10.py:39`

### freeze (method) `def freeze(self)`
- Defined: `plank10.py:67`

### unfreeze (method) `def unfreeze(self)`
- Defined: `plank10.py:71`

### forward (method) `def forward(self, x)`
- Defined: `plank10.py:75`

### __init__ (method) `def __init__(self, input_dim, hidden_dim, num_classes)`
- Defined: `plank10.py:92`

### apply_masks (method) `def apply_masks(self)`
- Defined: `plank10.py:104`

### get_sparsity (method) `def get_sparsity(self)`
- Defined: `plank10.py:109`

### forward (method) `def forward(self, x)`
- Defined: `plank10.py:114`

### __init__ (method) `def __init__(self, input_dim, hidden_dim)`
- Defined: `plank10.py:123`

### forward (method) `def forward(self, x)`
- Defined: `plank10.py:127`

### compute_L (method) `def compute_L(self, weight)`
- Defined: `plank10.py:134`

### __init__ (method) `def __init__(self, device)`
- Defined: `plank10.py:149`

### _gradient_nudge_inheritance (method) `def _gradient_nudge_inheritance(self, child_model, elk_state, data_loader, feature_extractor, nudge_lr)`
- Defined: `plank10.py:154`

### _apply_dynamic_spectral_shock (method) `def _apply_dynamic_spectral_shock(self, model, layer_name)`
- Defined: `plank10.py:188`

### create_apex_offspring (method) `def create_apex_offspring(self, elk_state, new_hidden_dim, cycle, data_loader, feature_extractor, parent_gap)`
- Defined: `plank10.py:211`

### __init__ (method) `def __init__(self, device, feature_extractor)`
- Defined: `plank10.py:248`

### _preprocess_batch (method) `def _preprocess_batch(self, x)`
- Defined: `plank10.py:255`

### get_curriculum_dataset (method) `def get_curriculum_dataset(self, cycle)`
- Defined: `plank10.py:259`

### train_model (method) `def train_model(self, model, cycle, is_baseline)`
- Defined: `plank10.py:265`

## plank11.py

### main (method) `def main()`
- Defined: `plank11.py:379`

### __init__ (method) `def __init__(self, num_tokens)`
- Defined: `plank11.py:41`

### forward (method) `def forward(self, x)`
- Defined: `plank11.py:50`

### __init__ (method) `def __init__(self, img_size, patch_size, in_chans, embed_dim)`
- Defined: `plank11.py:61`

### freeze (method) `def freeze(self)`
- Defined: `plank11.py:75`

### unfreeze (method) `def unfreeze(self)`
- Defined: `plank11.py:79`

### forward (method) `def forward(self, x)`
- Defined: `plank11.py:83`

### __init__ (method) `def __init__(self, input_dim, hidden_dim, num_classes)`
- Defined: `plank11.py:98`

### apply_masks (method) `def apply_masks(self)`
- Defined: `plank11.py:109`

### get_sparsity (method) `def get_sparsity(self)`
- Defined: `plank11.py:114`

### forward (method) `def forward(self, x)`
- Defined: `plank11.py:119`

### __init__ (method) `def __init__(self, input_dim, hidden_dim)`
- Defined: `plank11.py:127`

### forward (method) `def forward(self, x)`
- Defined: `plank11.py:131`

### compute_L (method) `def compute_L(self, weight)`
- Defined: `plank11.py:138`

### __init__ (method) `def __init__(self, device)`
- Defined: `plank11.py:153`

### _gradient_nudge_inheritance (method) `def _gradient_nudge_inheritance(self, child_model, elk_state, data_loader, feature_extractor, nudge_lr)`
- Defined: `plank11.py:158`

### _apply_minimalistic_shock (method) `def _apply_minimalistic_shock(self, model, layer_name, target_rank_ratio)`
- Defined: `plank11.py:190`
- Doc: Rank Capping: Cortamos singular values débiles y NO renormalizamos.

### create_orthogonal_offspring (method) `def create_orthogonal_offspring(self, elk_state, new_hidden_dim, cycle, data_loader, feature_extractor, parent_gap)`
- Defined: `plank11.py:223`

### __init__ (method) `def __init__(self, device, feature_extractor)`
- Defined: `plank11.py:261`

### _preprocess_batch (method) `def _preprocess_batch(self, x)`
- Defined: `plank11.py:268`

### get_curriculum_dataset (method) `def get_curriculum_dataset(self, cycle)`
- Defined: `plank11.py:272`

### train_model (method) `def train_model(self, model, cycle, is_baseline)`
- Defined: `plank11.py:278`

## plank12.py

### main (method) `def main()`
- Defined: `plank12.py:363`

### __init__ (method) `def __init__(self, num_tokens)`
- Defined: `plank12.py:37`

### forward (method) `def forward(self, x)`
- Defined: `plank12.py:44`

### __init__ (method) `def __init__(self, img_size, patch_size, in_chans, embed_dim)`
- Defined: `plank12.py:52`

### freeze (method) `def freeze(self)`
- Defined: `plank12.py:66`

### unfreeze (method) `def unfreeze(self)`
- Defined: `plank12.py:70`

### forward (method) `def forward(self, x)`
- Defined: `plank12.py:74`

### __init__ (method) `def __init__(self, input_dim, hidden_dim, num_classes)`
- Defined: `plank12.py:88`

### apply_masks (method) `def apply_masks(self)`
- Defined: `plank12.py:99`

### get_sparsity (method) `def get_sparsity(self)`
- Defined: `plank12.py:104`

### forward (method) `def forward(self, x)`
- Defined: `plank12.py:109`

### __init__ (method) `def __init__(self, input_dim, hidden_dim)`
- Defined: `plank12.py:117`

### forward (method) `def forward(self, x)`
- Defined: `plank12.py:121`

### compute_metrics (method) `def compute_metrics(self, weight)`
- Defined: `plank12.py:128`
- Doc: Returns: (L, Rank_Efficient, S_vN)

### __init__ (method) `def __init__(self, device)`
- Defined: `plank12.py:155`

### _gradient_nudge_inheritance (method) `def _gradient_nudge_inheritance(self, child_model, elk_state, data_loader, feature_extractor, nudge_lr)`
- Defined: `plank12.py:160`

### _apply_rank_capping_shock (method) `def _apply_rank_capping_shock(self, model, layer_name, keep_ratio)`
- Defined: `plank12.py:192`
- Doc: Minimalistic Shock: Zero out weak singular values without renormalizing.

### create_refined_offspring (method) `def create_refined_offspring(self, elk_state, data_loader, feature_extractor)`
- Defined: `plank12.py:224`
- Doc: Crea un hijo de las MISMAS dimensiones (Fixed Width).

### __init__ (method) `def __init__(self, device, feature_extractor)`
- Defined: `plank12.py:249`

### _preprocess_batch (method) `def _preprocess_batch(self, x)`
- Defined: `plank12.py:256`

### get_curriculum_dataset (method) `def get_curriculum_dataset(self, cycle)`
- Defined: `plank12.py:260`

### train_model (method) `def train_model(self, model, cycle, is_baseline)`
- Defined: `plank12.py:266`

## plank13.py

### main (method) `def main()`
- Defined: `plank13.py:329`

### __init__ (method) `def __init__(self, num_tokens)`
- Defined: `plank13.py:34`

### forward (method) `def forward(self, x)`
- Defined: `plank13.py:41`

### __init__ (method) `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)`
- Defined: `plank13.py:53`

### freeze (method) `def freeze(self)`
- Defined: `plank13.py:70`

### unfreeze (method) `def unfreeze(self)`
- Defined: `plank13.py:74`

### forward (method) `def forward(self, x)`
- Defined: `plank13.py:78`

### __init__ (method) `def __init__(self, input_dim, hidden_dim, num_classes)`
- Defined: `plank13.py:93`

### apply_masks (method) `def apply_masks(self)`
- Defined: `plank13.py:104`

### get_sparsity (method) `def get_sparsity(self)`
- Defined: `plank13.py:109`

### forward (method) `def forward(self, x)`
- Defined: `plank13.py:114`

### compute_metrics (method) `def compute_metrics(self, weight)`
- Defined: `plank13.py:124`
- Doc: Returns: (L, Rank_Efficient, S_vN)

### __init__ (method) `def __init__(self, device)`
- Defined: `plank13.py:150`

### _gradient_nudge_inheritance (method) `def _gradient_nudge_inheritance(self, child_model, elk_state, data_loader, feature_extractor, nudge_lr)`
- Defined: `plank13.py:155`

### _apply_rank_capping_shock (method) `def _apply_rank_capping_shock(self, model, layer_name, keep_ratio)`
- Defined: `plank13.py:187`

### create_refined_offspring (method) `def create_refined_offspring(self, elk_state, data_loader, feature_extractor)`
- Defined: `plank13.py:207`

### __init__ (method) `def __init__(self, device, extractor_apex, extractor_blind)`
- Defined: `plank13.py:225`

### _preprocess_batch (method) `def _preprocess_batch(self, x, extractor)`
- Defined: `plank13.py:233`

### get_curriculum_dataset (method) `def get_curriculum_dataset(self, cycle)`
- Defined: `plank13.py:237`

### train_single_chain (method) `def train_single_chain(self, model, cycle, chain_type)`
- Defined: `plank13.py:243`
- Doc: Entrena una cadena específica (Apex o Blind).

## plank2.py

### train_condition (method) `def train_condition(condition_name, config, device, seed)`
- Defined: `plank2.py:173`
- Doc: Train one condition and return full log as DataFrame.

### main (method) `def main()`
- Defined: `plank2.py:281`

### __init__ (method) `def __init__(self, epsilon_c)`
- Defined: `plank2.py:46`

### compute_L (method) `def compute_L(self, weight)`
- Defined: `plank2.py:49`
- Doc: Returns: (L, S_vN, rank_eff, regime)

### __init__ (method) `def __init__(self, sparsity_target)`
- Defined: `plank2.py:87`

### apply_to_model (method) `def apply_to_model(self, model)`
- Defined: `plank2.py:91`
- Doc: Apply pruning mask and register backward hook to zero gradients.

### enforce_during_training (method) `def enforce_during_training(self, model)`
- Defined: `plank2.py:105`
- Doc: Call this after every optimizer.step()

### __init__ (method) `def __init__(self, input_dim, hidden_dim, num_classes)`
- Defined: `plank2.py:119`

### reduce_input (method) `def reduce_input(self, x)`
- Defined: `plank2.py:128`
- Doc: Reduce CIFAR-10 (32x32x3) to 32D for focus

### forward (method) `def forward(self, x)`
- Defined: `plank2.py:135`

## plank3.py

### train_dense_to_target (method) `def train_dense_to_target(device, target_acc)`
- Defined: `plank3.py:110`
- Doc: Train dense model until it reaches target accuracy.

### progressive_pruning_search (method) `def progressive_pruning_search(model, device, target_acc)`
- Defined: `plank3.py:188`
- Doc: Progressively prune model and find critical density threshold.

### find_critical_threshold (method) `def find_critical_threshold(pruning_df, target_acc)`
- Defined: `plank3.py:264`
- Doc: Find the minimum density where accuracy >= target_acc.

### main (method) `def main()`
- Defined: `plank3.py:294`

### __init__ (method) `def __init__(self, epsilon_c)`
- Defined: `plank3.py:35`

### compute_L (method) `def compute_L(self, weight)`
- Defined: `plank3.py:38`

### __init__ (method) `def __init__(self, sparsity_target)`
- Defined: `plank3.py:63`

### apply_to_model (method) `def apply_to_model(self, model)`
- Defined: `plank3.py:67`

### enforce_during_training (method) `def enforce_during_training(self, model)`
- Defined: `plank3.py:77`

### __init__ (method) `def __init__(self, input_dim, hidden_dim, num_classes)`
- Defined: `plank3.py:88`

### reduce_input (method) `def reduce_input(self, x)`
- Defined: `plank3.py:96`

### forward (method) `def forward(self, x)`
- Defined: `plank3.py:102`

## plank4.py

### main (method) `def main()`
- Defined: `plank4.py:329`

### __init__ (method) `def __init__(self, epsilon_c)`
- Defined: `plank4.py:42`

### compute_L (method) `def compute_L(self, weight)`
- Defined: `plank4.py:45`

### __init__ (method) `def __init__(self, sparsity_target)`
- Defined: `plank4.py:67`

### apply_to_model (method) `def apply_to_model(self, model)`
- Defined: `plank4.py:71`

### enforce_during_training (method) `def enforce_during_training(self, model)`
- Defined: `plank4.py:81`

### __init__ (method) `def __init__(self, input_dim, hidden_dim, num_classes)`
- Defined: `plank4.py:89`

### reduce_input (method) `def reduce_input(self, x)`
- Defined: `plank4.py:97`

### forward (method) `def forward(self, x)`
- Defined: `plank4.py:103`

### __init__ (method) `def __init__(self, device, base_target_acc)`
- Defined: `plank4.py:112`

### train_dense_model (method) `def train_dense_model(self, hidden_dim, target_acc)`
- Defined: `plank4.py:123`

### extract_seed_weights (method) `def extract_seed_weights(self, model)`
- Defined: `plank4.py:168`

### inoculate_seed (method) `def inoculate_seed(self, large_model, seed_weights)`
- Defined: `plank4.py:174`
- Doc: Embed seed into larger architecture

### progressive_pruning (method) `def progressive_pruning(self, model, target_acc)`
- Defined: `plank4.py:192`

### execute_cycle (method) `def execute_cycle(self, cycle, base_hidden_dim, expansion_factor)`
- Defined: `plank4.py:236`

### run_experiment (method) `def run_experiment(self, num_cycles)`
- Defined: `plank4.py:281`

## plank5.py

### main (method) `def main()`
- Defined: `plank5.py:556`

### __init__ (method) `def __init__(self, epsilon_c)`
- Defined: `plank5.py:31`

### compute_L (method) `def compute_L(self, weight)`
- Defined: `plank5.py:34`

### __init__ (method) `def __init__(self, sparsity_target)`
- Defined: `plank5.py:51`

### apply_to_model (method) `def apply_to_model(self, model)`
- Defined: `plank5.py:55`

### __init__ (method) `def __init__(self, input_dim, hidden_dim, num_classes)`
- Defined: `plank5.py:66`

### reduce_input (method) `def reduce_input(self, x)`
- Defined: `plank5.py:74`

### forward (method) `def forward(self, x)`
- Defined: `plank5.py:80`

### __init__ (method) `def __init__(self, patience, gap_threshold)`
- Defined: `plank5.py:89`

### update (method) `def update(self, train_acc, test_acc, epoch)`
- Defined: `plank5.py:94`

### detect_grokking (method) `def detect_grokking(self)`
- Defined: `plank5.py:103`

### __init__ (method) `def __init__(self, device, target_acc, min_L)`
- Defined: `plank5.py:122`

### generate (method) `def generate(self, hidden_dim)`
- Defined: `plank5.py:128`
- Doc: Genera cisne negro sintético si no existe legacy

### __init__ (method) `def __init__(self, device, base_acc)`
- Defined: `plank5.py:212`

### inoculate_dna (method) `def inoculate_dna(self, large_model, seed_weights, noise_scale)`
- Defined: `plank5.py:218`
- Doc: Inocula ADN del cisne anterior con mutación controlada

### train_with_grokking (method) `def train_with_grokking(self, model, seed_model, target_acc)`
- Defined: `plank5.py:243`
- Doc: Entrena modelo induciendo grokking y monitoreando transición de fase

### distill_sparse_model (method) `def distill_sparse_model(self, model, target_acc)`
- Defined: `plank5.py:337`
- Doc: Pruning progresivo para extraer nuevo cisne negro

### __init__ (method) `def __init__(self, device, num_cycles, base_acc)`
- Defined: `plank5.py:393`

### load_legacy_or_generate_seed (method) `def load_legacy_or_generate_seed(self)`
- Defined: `plank5.py:402`
- Doc: Carga legacy seed o genera uno sintético

### run_evolutionary_chain (method) `def run_evolutionary_chain(self)`
- Defined: `plank5.py:419`
- Doc: Ejecuta la cadena evolutiva completa

### save_chain_results (method) `def save_chain_results(self)`
- Defined: `plank5.py:501`
- Doc: Guarda resultados completos de la cadena evolutiva

### print_evolution_summary (method) `def print_evolution_summary(self)`
- Defined: `plank5.py:519`
- Doc: Imprime resumen ejecutivo de la cadena evolutiva

## plank6.py

### main (method) `def main()`
- Defined: `plank6.py:293`

### __init__ (method) `def __init__(self, epsilon_c)`
- Defined: `plank6.py:31`

### compute_L (method) `def compute_L(self, weight)`
- Defined: `plank6.py:34`

### __init__ (method) `def __init__(self, input_dim, hidden_dim, num_classes)`
- Defined: `plank6.py:53`

### reduce_input (method) `def reduce_input(self, x)`
- Defined: `plank6.py:63`

### forward (method) `def forward(self, x)`
- Defined: `plank6.py:70`

### __init__ (method) `def __init__(self, patience, gap_threshold)`
- Defined: `plank6.py:79`

### update (method) `def update(self, train_acc, test_acc, epoch)`
- Defined: `plank6.py:84`

### detect_grokking (method) `def detect_grokking(self)`
- Defined: `plank6.py:87`

### __init__ (method) `def __init__(self, device)`
- Defined: `plank6.py:106`

### _guided_elk_mutation (method) `def _guided_elk_mutation(self, old_weight, target_shape, noise_scale, refinement_steps)`
- Defined: `plank6.py:110`
- Doc: Evoluciona los pesos del Elk a una dimensión mayor manteniendo coherencia.

### _apply_spectral_refinement (method) `def _apply_spectral_refinement(self, W)`
- Defined: `plank6.py:146`
- Doc: Filtra componentes de baja energía y reconstruye

### create_offspring_from_elk (method) `def create_offspring_from_elk(self, elk_state, new_hidden_dim, generation)`
- Defined: `plank6.py:158`
- Doc: Crea un nuevo modelo (Cisne Negro) basado en el Elk (mejor modelo previo).

### __init__ (method) `def __init__(self, device)`
- Defined: `plank6.py:200`

### train_phase (method) `def train_phase(self, model, cycle_id)`
- Defined: `plank6.py:204`

## plank7.py

### main (method) `def main()`
- Defined: `plank7.py:283`

### __init__ (method) `def __init__(self, epsilon_c)`
- Defined: `plank7.py:30`

### compute_L (method) `def compute_L(self, weight)`
- Defined: `plank7.py:33`

### __init__ (method) `def __init__(self, input_dim, hidden_dim, num_classes)`
- Defined: `plank7.py:46`

### reduce_input (method) `def reduce_input(self, x)`
- Defined: `plank7.py:54`

### forward (method) `def forward(self, x)`
- Defined: `plank7.py:60`

### __init__ (method) `def __init__(self, device)`
- Defined: `plank7.py:72`

### _gradient_nudge_inheritance (method) `def _gradient_nudge_inheritance(self, child_model, elk_state, data_loader, nudge_lr)`
- Defined: `plank7.py:76`
- Doc: Antes de entrenar, hacemos 1 paso de gradiente del Elk sobre los nuevos datos.

### _apply_spectral_shock (method) `def _apply_spectral_shock(self, W, shock_intensity)`
- Defined: `plank7.py:107`
- Doc: Aplica una perturbación no-lineal a los valores singulares.

### create_advanced_offspring (method) `def create_advanced_offspring(self, elk_state, new_hidden_dim, cycle, data_loader)`
- Defined: `plank7.py:124`
- Doc: Crea un hijo combinando:

### __init__ (method) `def __init__(self, device)`
- Defined: `plank7.py:178`

### get_curriculum_dataset (method) `def get_curriculum_dataset(self, cycle)`
- Defined: `plank7.py:186`
- Doc: Estrategia de Curriculum:

### train_phase (method) `def train_phase(self, model, cycle)`
- Defined: `plank7.py:204`

## plank8.py

### main (method) `def main()`
- Defined: `plank8.py:342`

### __init__ (method) `def __init__(self, img_size, patch_size, in_chans, embed_dim)`
- Defined: `plank8.py:39`

### freeze (method) `def freeze(self)`
- Defined: `plank8.py:57`

### unfreeze (method) `def unfreeze(self)`
- Defined: `plank8.py:61`

### forward (method) `def forward(self, x)`
- Defined: `plank8.py:65`

### __init__ (method) `def __init__(self, input_dim, hidden_dim, num_classes)`
- Defined: `plank8.py:79`

### apply_masks (method) `def apply_masks(self)`
- Defined: `plank8.py:92`

### get_sparsity (method) `def get_sparsity(self)`
- Defined: `plank8.py:97`

### forward (method) `def forward(self, x)`
- Defined: `plank8.py:102`

### __init__ (method) `def __init__(self, input_dim, hidden_dim)`
- Defined: `plank8.py:111`

### forward (method) `def forward(self, x)`
- Defined: `plank8.py:115`

### compute_L (method) `def compute_L(self, weight)`
- Defined: `plank8.py:122`

### __init__ (method) `def __init__(self, device)`
- Defined: `plank8.py:137`

### _gradient_nudge_inheritance (method) `def _gradient_nudge_inheritance(self, child_model, elk_state, data_loader, feature_extractor, nudge_lr)`
- Defined: `plank8.py:142`

### _apply_dynamic_spectral_shock (method) `def _apply_dynamic_spectral_shock(self, model, layer_name)`
- Defined: `plank8.py:176`

### create_apex_offspring (method) `def create_apex_offspring(self, elk_state, new_hidden_dim, cycle, data_loader, feature_extractor, parent_gap)`
- Defined: `plank8.py:199`

### __init__ (method) `def __init__(self, device, feature_extractor)`
- Defined: `plank8.py:236`

### _preprocess_batch (method) `def _preprocess_batch(self, x)`
- Defined: `plank8.py:243`

### get_curriculum_dataset (method) `def get_curriculum_dataset(self, cycle)`
- Defined: `plank8.py:247`

### train_model (method) `def train_model(self, model, cycle, is_baseline)`
- Defined: `plank8.py:253`

## resmav2_1.py

### load_elliptic_data (method) `def load_elliptic_data()`
- Defined: `resmav2_1.py:207`

### train_and_evaluate (method) `def train_and_evaluate(model, X, y, edge_index, train_idx, val_idx, epochs, lr, name, fold)`
- Defined: `resmav2_1.py:264`

### cross_validate_model (method) `def cross_validate_model(model_class, X, y, edge_index, num_nodes, n_splits, seed, name)`
- Defined: `resmav2_1.py:321`

### __init__ (method) `def __init__(self, in_features, out_features, edge_index, num_nodes)`
- Defined: `resmav2_1.py:25`

### forward (method) `def forward(self, x)`
- Defined: `resmav2_1.py:40`

### __init__ (method) `def __init__(self, input_dim, hidden_dim, edge_index, num_nodes, dropout)`
- Defined: `resmav2_1.py:50`

### forward (method) `def forward(self, x, edge_index)`
- Defined: `resmav2_1.py:73`

### __init__ (method) `def __init__(self, input_dim, hidden_dim, edge_index, num_nodes, dropout)`
- Defined: `resmav2_1.py:88`

### forward (method) `def forward(self, x, edge_index)`
- Defined: `resmav2_1.py:113`

### __init__ (method) `def __init__(self, input_dim, hidden_dim, edge_index, num_nodes, dropout)`
- Defined: `resmav2_1.py:136`

### forward (method) `def forward(self, x, edge_index)`
- Defined: `resmav2_1.py:168`

### __init__ (method) `def __init__(self, input_dim, hidden_dim, dropout)`
- Defined: `resmav2_1.py:184`

### forward (method) `def forward(self, x, edge_index)`
- Defined: `resmav2_1.py:194`
