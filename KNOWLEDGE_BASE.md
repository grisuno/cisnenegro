# Polyglot Codebase Knowledge Graph

> Generated offline by **readmenator**. Supports C, C++, Python, Go, Rust, JS/TS, Java, C#, Shell, PHP, Dart, GDScript, Nim, ASM.
> No LLMs. No tokens. Pure static analysis.

**Total Files Parsed:** 37 | **Total Symbols Extracted:** 1059 | **Total Imports:** 418

## Structural Knowledge Map
```mermaid
graph TD
    classDef mod fill:#1e1e1e,stroke:#ff6666,stroke-width:2px,color:#fff;
    classDef cls fill:#2d2d2d,stroke:#4ec9b0,stroke-width:2px,color:#fff;
    classDef fn fill:#333,stroke:#dcdcaa,stroke-width:1px,color:#dcdcaa;
    classDef ext fill:#111,stroke:#666,stroke-dasharray: 5 5,color:#aaa;
    apex27_py["apex27.py (py)"]
    class apex27_py mod;
    apex27_py_set_seed["set_seed"]
    class apex27_py_set_seed fn;
    apex27_py --> apex27_py_set_seed
    apex27_py_GatedTokenMixer["GatedTokenMixer"]
    class apex27_py_GatedTokenMixer cls;
    apex27_py --> apex27_py_GatedTokenMixer
    apex27_py_PatchFeatureExtractor["PatchFeatureExtractor"]
    class apex27_py_PatchFeatureExtractor cls;
    apex27_py --> apex27_py_PatchFeatureExtractor
    apex27_py_TaxonomicMLP["TaxonomicMLP"]
    class apex27_py_TaxonomicMLP cls;
    apex27_py --> apex27_py_TaxonomicMLP
    apex27_py_compute_spectral_loss["compute_spectral_loss"]
    class apex27_py_compute_spectral_loss fn;
    apex27_py --> apex27_py_compute_spectral_loss
    resmav2_1_py["resmav2_1.py (py)"]
    class resmav2_1_py mod;
    resmav2_1_py_OptimizedE8Layer["OptimizedE8Layer"]
    class resmav2_1_py_OptimizedE8Layer cls;
    resmav2_1_py --> resmav2_1_py_OptimizedE8Layer
    resmav2_1_py_RESMAv2Fast["RESMAv2Fast"]
    class resmav2_1_py_RESMAv2Fast cls;
    resmav2_1_py --> resmav2_1_py_RESMAv2Fast
    resmav2_1_py_RESMAv2Standard["RESMAv2Standard"]
    class resmav2_1_py_RESMAv2Standard cls;
    resmav2_1_py --> resmav2_1_py_RESMAv2Standard
    resmav2_1_py_RESMAv2Deep["RESMAv2Deep"]
    class resmav2_1_py_RESMAv2Deep cls;
    resmav2_1_py --> resmav2_1_py_RESMAv2Deep
    resmav2_1_py_GAT_Baseline["GAT_Baseline"]
    class resmav2_1_py_GAT_Baseline cls;
    resmav2_1_py --> resmav2_1_py_GAT_Baseline
    apex28_py["apex28.py (py)"]
    class apex28_py mod;
    apex28_py_set_seed["set_seed"]
    class apex28_py_set_seed fn;
    apex28_py --> apex28_py_set_seed
    apex28_py_GatedTokenMixer["GatedTokenMixer"]
    class apex28_py_GatedTokenMixer cls;
    apex28_py --> apex28_py_GatedTokenMixer
    apex28_py_PatchFeatureExtractor["PatchFeatureExtractor"]
    class apex28_py_PatchFeatureExtractor cls;
    apex28_py --> apex28_py_PatchFeatureExtractor
    apex28_py_TaxonomicMLP["TaxonomicMLP"]
    class apex28_py_TaxonomicMLP cls;
    apex28_py --> apex28_py_TaxonomicMLP
    apex28_py_compute_spectral_loss["compute_spectral_loss"]
    class apex28_py_compute_spectral_loss fn;
    apex28_py --> apex28_py_compute_spectral_loss
    apex29_py["apex29.py (py)"]
    class apex29_py mod;
    apex29_py_set_seed["set_seed"]
    class apex29_py_set_seed fn;
    apex29_py --> apex29_py_set_seed
    apex29_py_GatedTokenMixer["GatedTokenMixer"]
    class apex29_py_GatedTokenMixer cls;
    apex29_py --> apex29_py_GatedTokenMixer
    apex29_py_PatchFeatureExtractor["PatchFeatureExtractor"]
    class apex29_py_PatchFeatureExtractor cls;
    apex29_py --> apex29_py_PatchFeatureExtractor
    apex29_py_TaxonomicMLP["TaxonomicMLP"]
    class apex29_py_TaxonomicMLP cls;
    apex29_py --> apex29_py_TaxonomicMLP
    apex29_py_compute_spectral_loss["compute_spectral_loss"]
    class apex29_py_compute_spectral_loss fn;
    apex29_py --> apex29_py_compute_spectral_loss
    apex30_py["apex30.py (py)"]
    class apex30_py mod;
    apex30_py_set_seed["set_seed"]
    class apex30_py_set_seed fn;
    apex30_py --> apex30_py_set_seed
    apex30_py_GatedTokenMixer["GatedTokenMixer"]
    class apex30_py_GatedTokenMixer cls;
    apex30_py --> apex30_py_GatedTokenMixer
    apex30_py_PatchFeatureExtractor["PatchFeatureExtractor"]
    class apex30_py_PatchFeatureExtractor cls;
    apex30_py --> apex30_py_PatchFeatureExtractor
    apex30_py_TaxonomicMLP["TaxonomicMLP"]
    class apex30_py_TaxonomicMLP cls;
    apex30_py --> apex30_py_TaxonomicMLP
    apex30_py_compute_spectral_loss["compute_spectral_loss"]
    class apex30_py_compute_spectral_loss fn;
    apex30_py --> apex30_py_compute_spectral_loss
    apex31_py["apex31.py (py)"]
    class apex31_py mod;
    apex31_py_set_seed["set_seed"]
    class apex31_py_set_seed fn;
    apex31_py --> apex31_py_set_seed
    apex31_py_GatedTokenMixer["GatedTokenMixer"]
    class apex31_py_GatedTokenMixer cls;
    apex31_py --> apex31_py_GatedTokenMixer
    apex31_py_PatchFeatureExtractor["PatchFeatureExtractor"]
    class apex31_py_PatchFeatureExtractor cls;
    apex31_py --> apex31_py_PatchFeatureExtractor
    apex31_py_TaxonomicMLP["TaxonomicMLP"]
    class apex31_py_TaxonomicMLP cls;
    apex31_py --> apex31_py_TaxonomicMLP
    apex31_py_compute_spectral_loss["compute_spectral_loss"]
    class apex31_py_compute_spectral_loss fn;
    apex31_py --> apex31_py_compute_spectral_loss
    apex26_py["apex26.py (py)"]
    class apex26_py mod;
    apex26_py_set_seed["set_seed"]
    class apex26_py_set_seed fn;
    apex26_py --> apex26_py_set_seed
    apex26_py_GatedTokenMixer["GatedTokenMixer"]
    class apex26_py_GatedTokenMixer cls;
    apex26_py --> apex26_py_GatedTokenMixer
    apex26_py_PatchFeatureExtractor["PatchFeatureExtractor"]
    class apex26_py_PatchFeatureExtractor cls;
    apex26_py --> apex26_py_PatchFeatureExtractor
    apex26_py_TaxonomicMLP["TaxonomicMLP"]
    class apex26_py_TaxonomicMLP cls;
    apex26_py --> apex26_py_TaxonomicMLP
    apex26_py_compute_spectral_loss["compute_spectral_loss"]
    class apex26_py_compute_spectral_loss fn;
    apex26_py --> apex26_py_compute_spectral_loss
    apex33_py["apex33.py (py)"]
    class apex33_py mod;
    apex33_py_set_seed["set_seed"]
    class apex33_py_set_seed fn;
    apex33_py --> apex33_py_set_seed
    apex33_py_GatedTokenMixer["GatedTokenMixer"]
    class apex33_py_GatedTokenMixer cls;
    apex33_py --> apex33_py_GatedTokenMixer
    apex33_py_PatchFeatureExtractor["PatchFeatureExtractor"]
    class apex33_py_PatchFeatureExtractor cls;
    apex33_py --> apex33_py_PatchFeatureExtractor
    apex33_py_TaxonomicMLP["TaxonomicMLP"]
    class apex33_py_TaxonomicMLP cls;
    apex33_py --> apex33_py_TaxonomicMLP
    apex33_py_compute_spectral_loss["compute_spectral_loss"]
    class apex33_py_compute_spectral_loss fn;
    apex33_py --> apex33_py_compute_spectral_loss
    apex34_py["apex34.py (py)"]
    class apex34_py mod;
    apex34_py_set_seed["set_seed"]
    class apex34_py_set_seed fn;
    apex34_py --> apex34_py_set_seed
    apex34_py_GatedTokenMixer["GatedTokenMixer"]
    class apex34_py_GatedTokenMixer cls;
    apex34_py --> apex34_py_GatedTokenMixer
    apex34_py_PatchFeatureExtractor["PatchFeatureExtractor"]
    class apex34_py_PatchFeatureExtractor cls;
    apex34_py --> apex34_py_PatchFeatureExtractor
    apex34_py_TaxonomicMLP["TaxonomicMLP"]
    class apex34_py_TaxonomicMLP cls;
    apex34_py --> apex34_py_TaxonomicMLP
    apex34_py_compute_spectral_loss["compute_spectral_loss"]
    class apex34_py_compute_spectral_loss fn;
    apex34_py --> apex34_py_compute_spectral_loss
    apex35_py["apex35.py (py)"]
    class apex35_py mod;
    apex35_py_set_seed["set_seed"]
    class apex35_py_set_seed fn;
    apex35_py --> apex35_py_set_seed
    apex35_py_GatedTokenMixer["GatedTokenMixer"]
    class apex35_py_GatedTokenMixer cls;
    apex35_py --> apex35_py_GatedTokenMixer
    apex35_py_E8FusionLayer["E8FusionLayer"]
    class apex35_py_E8FusionLayer cls;
    apex35_py --> apex35_py_E8FusionLayer
    apex35_py_PatchFeatureExtractor["PatchFeatureExtractor"]
    class apex35_py_PatchFeatureExtractor cls;
    apex35_py --> apex35_py_PatchFeatureExtractor
    apex35_py_TaxonomicMLP["TaxonomicMLP"]
    class apex35_py_TaxonomicMLP cls;
    apex35_py --> apex35_py_TaxonomicMLP
    plank5_py["plank5.py (py)"]
    class plank5_py mod;
    plank5_py_SpectralMonitor["SpectralMonitor"]
    class plank5_py_SpectralMonitor cls;
    plank5_py --> plank5_py_SpectralMonitor
    plank5_py_PersistentPruner["PersistentPruner"]
    class plank5_py_PersistentPruner cls;
    plank5_py --> plank5_py_PersistentPruner
    plank5_py_SpectralMLP["SpectralMLP"]
    class plank5_py_SpectralMLP cls;
    plank5_py --> plank5_py_SpectralMLP
    plank5_py_GrokkingDetector["GrokkingDetector"]
    class plank5_py_GrokkingDetector cls;
    plank5_py --> plank5_py_GrokkingDetector
    plank5_py_SyntheticBlackSwanGenerator["SyntheticBlackSwanGenerator"]
    class plank5_py_SyntheticBlackSwanGenerator cls;
    plank5_py --> plank5_py_SyntheticBlackSwanGenerator
    app_py["app.py (py)"]
    class app_py mod;
    app_py_SpectralMonitor["SpectralMonitor"]
    class app_py_SpectralMonitor cls;
    app_py --> app_py_SpectralMonitor
    app_py_PersistentPruner["PersistentPruner"]
    class app_py_PersistentPruner cls;
    app_py --> app_py_PersistentPruner
    app_py_SpectralMLP["SpectralMLP"]
    class app_py_SpectralMLP cls;
    app_py --> app_py_SpectralMLP
    app_py_EvolutionaryResonanceEngine["EvolutionaryResonanceEngine"]
    class app_py_EvolutionaryResonanceEngine cls;
    app_py --> app_py_EvolutionaryResonanceEngine
    app_py_main["main"]
    class app_py_main fn;
    app_py --> app_py_main
    plank4_py["plank4.py (py)"]
    class plank4_py mod;
    plank4_py_SpectralMonitor["SpectralMonitor"]
    class plank4_py_SpectralMonitor cls;
    plank4_py --> plank4_py_SpectralMonitor
    plank4_py_PersistentPruner["PersistentPruner"]
    class plank4_py_PersistentPruner cls;
    plank4_py --> plank4_py_PersistentPruner
    plank4_py_SpectralMLP["SpectralMLP"]
    class plank4_py_SpectralMLP cls;
    plank4_py --> plank4_py_SpectralMLP
    plank4_py_FractalSovereigntyEngine["FractalSovereigntyEngine"]
    class plank4_py_FractalSovereigntyEngine cls;
    plank4_py --> plank4_py_FractalSovereigntyEngine
    plank4_py_main["main"]
    class plank4_py_main fn;
    plank4_py --> plank4_py_main
    plank6_py["plank6.py (py)"]
    class plank6_py mod;
    plank6_py_SpectralMonitor["SpectralMonitor"]
    class plank6_py_SpectralMonitor cls;
    plank6_py --> plank6_py_SpectralMonitor
    plank6_py_SpectralMLP["SpectralMLP"]
    class plank6_py_SpectralMLP cls;
    plank6_py --> plank6_py_SpectralMLP
    plank6_py_GrokkingDetector["GrokkingDetector"]
    class plank6_py_GrokkingDetector cls;
    plank6_py --> plank6_py_GrokkingDetector
    plank6_py_GuidedElkHuntingEngine["GuidedElkHuntingEngine"]
    class plank6_py_GuidedElkHuntingEngine cls;
    plank6_py --> plank6_py_GuidedElkHuntingEngine
    plank6_py_TrainingCycle["TrainingCycle"]
    class plank6_py_TrainingCycle cls;
    plank6_py --> plank6_py_TrainingCycle
    plank7_py["plank7.py (py)"]
    class plank7_py mod;
    plank7_py_SpectralMonitor["SpectralMonitor"]
    class plank7_py_SpectralMonitor cls;
    plank7_py --> plank7_py_SpectralMonitor
    plank7_py_SpectralMLP["SpectralMLP"]
    class plank7_py_SpectralMLP cls;
    plank7_py --> plank7_py_SpectralMLP
    plank7_py_AdvancedEvolutionEngine["AdvancedEvolutionEngine"]
    class plank7_py_AdvancedEvolutionEngine cls;
    plank7_py --> plank7_py_AdvancedEvolutionEngine
    plank7_py_CurriculumTrainingCycle["CurriculumTrainingCycle"]
    class plank7_py_CurriculumTrainingCycle cls;
    plank7_py --> plank7_py_CurriculumTrainingCycle
    plank7_py_main["main"]
    class plank7_py_main fn;
    plank7_py --> plank7_py_main
    plank3_py["plank3.py (py)"]
    class plank3_py mod;
    plank3_py_SpectralMonitor["SpectralMonitor"]
    class plank3_py_SpectralMonitor cls;
    plank3_py --> plank3_py_SpectralMonitor
    plank3_py_PersistentPruner["PersistentPruner"]
    class plank3_py_PersistentPruner cls;
    plank3_py --> plank3_py_PersistentPruner
    plank3_py_SpectralMLP["SpectralMLP"]
    class plank3_py_SpectralMLP cls;
    plank3_py --> plank3_py_SpectralMLP
    plank3_py_train_dense_to_target["train_dense_to_target"]
    class plank3_py_train_dense_to_target fn;
    plank3_py --> plank3_py_train_dense_to_target
    plank3_py_progressive_pruning_search["progressive_pruning_search"]
    class plank3_py_progressive_pruning_search fn;
    plank3_py --> plank3_py_progressive_pruning_search
    plank2_py["plank2.py (py)"]
    class plank2_py mod;
    plank2_py_SpectralMonitor["SpectralMonitor"]
    class plank2_py_SpectralMonitor cls;
    plank2_py --> plank2_py_SpectralMonitor
    plank2_py_PersistentPruner["PersistentPruner"]
    class plank2_py_PersistentPruner cls;
    plank2_py --> plank2_py_PersistentPruner
    plank2_py_SpectralMLP["SpectralMLP"]
    class plank2_py_SpectralMLP cls;
    plank2_py --> plank2_py_SpectralMLP
    plank2_py_train_condition["train_condition"]
    class plank2_py_train_condition fn;
    plank2_py --> plank2_py_train_condition
    plank2_py_main["main"]
    class plank2_py_main fn;
    plank2_py --> plank2_py_main
    apex21_py["apex21.py (py)"]
    class apex21_py mod;
    apex21_py_compute_spectral_loss["compute_spectral_loss"]
    class apex21_py_compute_spectral_loss fn;
    apex21_py --> apex21_py_compute_spectral_loss
    apex21_py_GatedTokenMixer["GatedTokenMixer"]
    class apex21_py_GatedTokenMixer cls;
    apex21_py --> apex21_py_GatedTokenMixer
    apex21_py_PatchFeatureExtractor["PatchFeatureExtractor"]
    class apex21_py_PatchFeatureExtractor cls;
    apex21_py --> apex21_py_PatchFeatureExtractor
    apex21_py_TaxonomicMLP["TaxonomicMLP"]
    class apex21_py_TaxonomicMLP cls;
    apex21_py --> apex21_py_TaxonomicMLP
    apex21_py_SpectralMonitor["SpectralMonitor"]
    class apex21_py_SpectralMonitor cls;
    apex21_py --> apex21_py_SpectralMonitor
    apex23_py["apex23.py (py)"]
    class apex23_py mod;
    apex23_py_compute_spectral_loss["compute_spectral_loss"]
    class apex23_py_compute_spectral_loss fn;
    apex23_py --> apex23_py_compute_spectral_loss
    apex23_py_GatedTokenMixer["GatedTokenMixer"]
    class apex23_py_GatedTokenMixer cls;
    apex23_py --> apex23_py_GatedTokenMixer
    apex23_py_PatchFeatureExtractor["PatchFeatureExtractor"]
    class apex23_py_PatchFeatureExtractor cls;
    apex23_py --> apex23_py_PatchFeatureExtractor
    apex23_py_TaxonomicMLP["TaxonomicMLP"]
    class apex23_py_TaxonomicMLP cls;
    apex23_py --> apex23_py_TaxonomicMLP
    apex23_py_SpectralMonitor["SpectralMonitor"]
    class apex23_py_SpectralMonitor cls;
    apex23_py --> apex23_py_SpectralMonitor
    apex24_py["apex24.py (py)"]
    class apex24_py mod;
    apex24_py_compute_spectral_loss["compute_spectral_loss"]
    class apex24_py_compute_spectral_loss fn;
    apex24_py --> apex24_py_compute_spectral_loss
    apex24_py_GatedTokenMixer["GatedTokenMixer"]
    class apex24_py_GatedTokenMixer cls;
    apex24_py --> apex24_py_GatedTokenMixer
    apex24_py_PatchFeatureExtractor["PatchFeatureExtractor"]
    class apex24_py_PatchFeatureExtractor cls;
    apex24_py --> apex24_py_PatchFeatureExtractor
    apex24_py_TaxonomicMLP["TaxonomicMLP"]
    class apex24_py_TaxonomicMLP cls;
    apex24_py --> apex24_py_TaxonomicMLP
    apex24_py_SpectralMonitor["SpectralMonitor"]
    class apex24_py_SpectralMonitor cls;
    apex24_py --> apex24_py_SpectralMonitor
    apex25_py["apex25.py (py)"]
    class apex25_py mod;
    apex25_py_compute_spectral_loss["compute_spectral_loss"]
    class apex25_py_compute_spectral_loss fn;
    apex25_py --> apex25_py_compute_spectral_loss
    apex25_py_GatedTokenMixer["GatedTokenMixer"]
    class apex25_py_GatedTokenMixer cls;
    apex25_py --> apex25_py_GatedTokenMixer
    apex25_py_PatchFeatureExtractor["PatchFeatureExtractor"]
    class apex25_py_PatchFeatureExtractor cls;
    apex25_py --> apex25_py_PatchFeatureExtractor
    apex25_py_TaxonomicMLP["TaxonomicMLP"]
    class apex25_py_TaxonomicMLP cls;
    apex25_py --> apex25_py_TaxonomicMLP
    apex25_py_SpectralMonitor["SpectralMonitor"]
    class apex25_py_SpectralMonitor cls;
    apex25_py --> apex25_py_SpectralMonitor
    apex22_py["apex22.py (py)"]
    class apex22_py mod;
    apex22_py_compute_spectral_loss["compute_spectral_loss"]
    class apex22_py_compute_spectral_loss fn;
    apex22_py --> apex22_py_compute_spectral_loss
    apex22_py_GatedTokenMixer["GatedTokenMixer"]
    class apex22_py_GatedTokenMixer cls;
    apex22_py --> apex22_py_GatedTokenMixer
    apex22_py_PatchFeatureExtractor["PatchFeatureExtractor"]
    class apex22_py_PatchFeatureExtractor cls;
    apex22_py --> apex22_py_PatchFeatureExtractor
    apex22_py_TaxonomicMLP["TaxonomicMLP"]
    class apex22_py_TaxonomicMLP cls;
    apex22_py --> apex22_py_TaxonomicMLP
    apex22_py_SpectralMonitor["SpectralMonitor"]
    class apex22_py_SpectralMonitor cls;
    apex22_py --> apex22_py_SpectralMonitor
    plank11_py["plank11.py (py)"]
    class plank11_py mod;
    plank11_py_TokenMixer["TokenMixer"]
    class plank11_py_TokenMixer cls;
    plank11_py --> plank11_py_TokenMixer
    plank11_py_PatchFeatureExtractor["PatchFeatureExtractor"]
    class plank11_py_PatchFeatureExtractor cls;
    plank11_py --> plank11_py_PatchFeatureExtractor
    plank11_py_LotteryMLP["LotteryMLP"]
    class plank11_py_LotteryMLP cls;
    plank11_py --> plank11_py_LotteryMLP
    plank11_py_StandardBaseline["StandardBaseline"]
    class plank11_py_StandardBaseline cls;
    plank11_py --> plank11_py_StandardBaseline
    plank11_py_SpectralMonitor["SpectralMonitor"]
    class plank11_py_SpectralMonitor cls;
    plank11_py --> plank11_py_SpectralMonitor
    plank12_py["plank12.py (py)"]
    class plank12_py mod;
    plank12_py_TokenMixer["TokenMixer"]
    class plank12_py_TokenMixer cls;
    plank12_py --> plank12_py_TokenMixer
    plank12_py_PatchFeatureExtractor["PatchFeatureExtractor"]
    class plank12_py_PatchFeatureExtractor cls;
    plank12_py --> plank12_py_PatchFeatureExtractor
    plank12_py_LotteryMLP["LotteryMLP"]
    class plank12_py_LotteryMLP cls;
    plank12_py --> plank12_py_LotteryMLP
    plank12_py_StandardBaseline["StandardBaseline"]
    class plank12_py_StandardBaseline cls;
    plank12_py --> plank12_py_StandardBaseline
    plank12_py_SpectralMonitor["SpectralMonitor"]
    class plank12_py_SpectralMonitor cls;
    plank12_py --> plank12_py_SpectralMonitor
    apex20_py["apex20.py (py)"]
    class apex20_py mod;
    apex20_py_compute_spectral_loss["compute_spectral_loss"]
    class apex20_py_compute_spectral_loss fn;
    apex20_py --> apex20_py_compute_spectral_loss
    apex20_py_GatedTokenMixer["GatedTokenMixer"]
    class apex20_py_GatedTokenMixer cls;
    apex20_py --> apex20_py_GatedTokenMixer
    apex20_py_PatchFeatureExtractor["PatchFeatureExtractor"]
    class apex20_py_PatchFeatureExtractor cls;
    apex20_py --> apex20_py_PatchFeatureExtractor
    apex20_py_TaxonomicMLP["TaxonomicMLP"]
    class apex20_py_TaxonomicMLP cls;
    apex20_py --> apex20_py_TaxonomicMLP
    apex20_py_SpectralMonitor["SpectralMonitor"]
    class apex20_py_SpectralMonitor cls;
    apex20_py --> apex20_py_SpectralMonitor
    apex32_py["apex32.py (py)"]
    class apex32_py mod;
    apex32_py_compute_spectral_loss["compute_spectral_loss"]
    class apex32_py_compute_spectral_loss fn;
    apex32_py --> apex32_py_compute_spectral_loss
    apex32_py_GatedTokenMixer["GatedTokenMixer"]
    class apex32_py_GatedTokenMixer cls;
    apex32_py --> apex32_py_GatedTokenMixer
    apex32_py_PatchFeatureExtractor["PatchFeatureExtractor"]
    class apex32_py_PatchFeatureExtractor cls;
    apex32_py --> apex32_py_PatchFeatureExtractor
    apex32_py_TaxonomicMLP["TaxonomicMLP"]
    class apex32_py_TaxonomicMLP cls;
    apex32_py --> apex32_py_TaxonomicMLP
    apex32_py_SpectralMonitor["SpectralMonitor"]
    class apex32_py_SpectralMonitor cls;
    apex32_py --> apex32_py_SpectralMonitor
    apex19_py["apex19.py (py)"]
    class apex19_py mod;
    apex19_py_compute_spectral_loss["compute_spectral_loss"]
    class apex19_py_compute_spectral_loss fn;
    apex19_py --> apex19_py_compute_spectral_loss
    apex19_py_GatedTokenMixer["GatedTokenMixer"]
    class apex19_py_GatedTokenMixer cls;
    apex19_py --> apex19_py_GatedTokenMixer
    apex19_py_PatchFeatureExtractor["PatchFeatureExtractor"]
    class apex19_py_PatchFeatureExtractor cls;
    apex19_py --> apex19_py_PatchFeatureExtractor
    apex19_py_TaxonomicMLP["TaxonomicMLP"]
    class apex19_py_TaxonomicMLP cls;
    apex19_py --> apex19_py_TaxonomicMLP
    apex19_py_SpectralMonitor["SpectralMonitor"]
    class apex19_py_SpectralMonitor cls;
    apex19_py --> apex19_py_SpectralMonitor
    apex14_py["apex14.py (py)"]
    class apex14_py mod;
    apex14_py_GatedTokenMixer["GatedTokenMixer"]
    class apex14_py_GatedTokenMixer cls;
    apex14_py --> apex14_py_GatedTokenMixer
    apex14_py_PatchFeatureExtractor["PatchFeatureExtractor"]
    class apex14_py_PatchFeatureExtractor cls;
    apex14_py --> apex14_py_PatchFeatureExtractor
    apex14_py_LotteryMLP["LotteryMLP"]
    class apex14_py_LotteryMLP cls;
    apex14_py --> apex14_py_LotteryMLP
    apex14_py_SpectralMonitor["SpectralMonitor"]
    class apex14_py_SpectralMonitor cls;
    apex14_py --> apex14_py_SpectralMonitor
    apex14_py_OrthogonalEvolutionEngine["OrthogonalEvolutionEngine"]
    class apex14_py_OrthogonalEvolutionEngine cls;
    apex14_py --> apex14_py_OrthogonalEvolutionEngine
    apex17_py["apex17.py (py)"]
    class apex17_py mod;
    apex17_py_compute_spectral_loss["compute_spectral_loss"]
    class apex17_py_compute_spectral_loss fn;
    apex17_py --> apex17_py_compute_spectral_loss
    apex17_py_GatedTokenMixer["GatedTokenMixer"]
    class apex17_py_GatedTokenMixer cls;
    apex17_py --> apex17_py_GatedTokenMixer
    apex17_py_PatchFeatureExtractor["PatchFeatureExtractor"]
    class apex17_py_PatchFeatureExtractor cls;
    apex17_py --> apex17_py_PatchFeatureExtractor
    apex17_py_TaxonomicMLP["TaxonomicMLP"]
    class apex17_py_TaxonomicMLP cls;
    apex17_py --> apex17_py_TaxonomicMLP
    apex17_py_SpectralMonitor["SpectralMonitor"]
    class apex17_py_SpectralMonitor cls;
    apex17_py --> apex17_py_SpectralMonitor
    apex18_py["apex18.py (py)"]
    class apex18_py mod;
    apex18_py_compute_spectral_loss["compute_spectral_loss"]
    class apex18_py_compute_spectral_loss fn;
    apex18_py --> apex18_py_compute_spectral_loss
    apex18_py_GatedTokenMixer["GatedTokenMixer"]
    class apex18_py_GatedTokenMixer cls;
    apex18_py --> apex18_py_GatedTokenMixer
    apex18_py_PatchFeatureExtractor["PatchFeatureExtractor"]
    class apex18_py_PatchFeatureExtractor cls;
    apex18_py --> apex18_py_PatchFeatureExtractor
    apex18_py_TaxonomicMLP["TaxonomicMLP"]
    class apex18_py_TaxonomicMLP cls;
    apex18_py --> apex18_py_TaxonomicMLP
    apex18_py_SpectralMonitor["SpectralMonitor"]
    class apex18_py_SpectralMonitor cls;
    apex18_py --> apex18_py_SpectralMonitor
    plank10_py["plank10.py (py)"]
    class plank10_py mod;
    plank10_py_PatchFeatureExtractor["PatchFeatureExtractor"]
    class plank10_py_PatchFeatureExtractor cls;
    plank10_py --> plank10_py_PatchFeatureExtractor
    plank10_py_LotteryMLP["LotteryMLP"]
    class plank10_py_LotteryMLP cls;
    plank10_py --> plank10_py_LotteryMLP
    plank10_py_StandardBaseline["StandardBaseline"]
    class plank10_py_StandardBaseline cls;
    plank10_py --> plank10_py_StandardBaseline
    plank10_py_SpectralMonitor["SpectralMonitor"]
    class plank10_py_SpectralMonitor cls;
    plank10_py --> plank10_py_SpectralMonitor
    plank10_py_ApexEvolutionEngine["ApexEvolutionEngine"]
    class plank10_py_ApexEvolutionEngine cls;
    plank10_py --> plank10_py_ApexEvolutionEngine
    plank13_py["plank13.py (py)"]
    class plank13_py mod;
    plank13_py_TokenMixer["TokenMixer"]
    class plank13_py_TokenMixer cls;
    plank13_py --> plank13_py_TokenMixer
    plank13_py_PatchFeatureExtractor["PatchFeatureExtractor"]
    class plank13_py_PatchFeatureExtractor cls;
    plank13_py --> plank13_py_PatchFeatureExtractor
    plank13_py_LotteryMLP["LotteryMLP"]
    class plank13_py_LotteryMLP cls;
    plank13_py --> plank13_py_LotteryMLP
    plank13_py_SpectralMonitor["SpectralMonitor"]
    class plank13_py_SpectralMonitor cls;
    plank13_py --> plank13_py_SpectralMonitor
    plank13_py_OrthogonalEvolutionEngine["OrthogonalEvolutionEngine"]
    class plank13_py_OrthogonalEvolutionEngine cls;
    plank13_py --> plank13_py_OrthogonalEvolutionEngine
    plank8_py["plank8.py (py)"]
    class plank8_py mod;
    plank8_py_PatchFeatureExtractor["PatchFeatureExtractor"]
    class plank8_py_PatchFeatureExtractor cls;
    plank8_py --> plank8_py_PatchFeatureExtractor
    plank8_py_LotteryMLP["LotteryMLP"]
    class plank8_py_LotteryMLP cls;
    plank8_py --> plank8_py_LotteryMLP
    plank8_py_StandardBaseline["StandardBaseline"]
    class plank8_py_StandardBaseline cls;
    plank8_py --> plank8_py_StandardBaseline
    plank8_py_SpectralMonitor["SpectralMonitor"]
    class plank8_py_SpectralMonitor cls;
    plank8_py --> plank8_py_SpectralMonitor
    plank8_py_ApexEvolutionEngine["ApexEvolutionEngine"]
    class plank8_py_ApexEvolutionEngine cls;
    plank8_py --> plank8_py_ApexEvolutionEngine
    apex15_py["apex15.py (py)"]
    class apex15_py mod;
    apex15_py_compute_spectral_loss["compute_spectral_loss"]
    class apex15_py_compute_spectral_loss fn;
    apex15_py --> apex15_py_compute_spectral_loss
    apex15_py_GatedTokenMixer["GatedTokenMixer"]
    class apex15_py_GatedTokenMixer cls;
    apex15_py --> apex15_py_GatedTokenMixer
    apex15_py_PatchFeatureExtractor["PatchFeatureExtractor"]
    class apex15_py_PatchFeatureExtractor cls;
    apex15_py --> apex15_py_PatchFeatureExtractor
    apex15_py_TaxonomicMLP["TaxonomicMLP"]
    class apex15_py_TaxonomicMLP cls;
    apex15_py --> apex15_py_TaxonomicMLP
    apex15_py_SpectralMonitor["SpectralMonitor"]
    class apex15_py_SpectralMonitor cls;
    apex15_py --> apex15_py_SpectralMonitor
    apex16_py["apex16.py (py)"]
    class apex16_py mod;
    apex16_py_compute_spectral_loss["compute_spectral_loss"]
    class apex16_py_compute_spectral_loss fn;
    apex16_py --> apex16_py_compute_spectral_loss
    apex16_py_GatedTokenMixer["GatedTokenMixer"]
    class apex16_py_GatedTokenMixer cls;
    apex16_py --> apex16_py_GatedTokenMixer
    apex16_py_PatchFeatureExtractor["PatchFeatureExtractor"]
    class apex16_py_PatchFeatureExtractor cls;
    apex16_py --> apex16_py_PatchFeatureExtractor
    apex16_py_TaxonomicMLP["TaxonomicMLP"]
    class apex16_py_TaxonomicMLP cls;
    apex16_py --> apex16_py_TaxonomicMLP
    apex16_py_SpectralMonitor["SpectralMonitor"]
    class apex16_py_SpectralMonitor cls;
    apex16_py --> apex16_py_SpectralMonitor
    plank_py["plank.py (py)"]
    class plank_py mod;
    plank_py_BlackMirrorMonitor["BlackMirrorMonitor"]
    class plank_py_BlackMirrorMonitor cls;
    plank_py --> plank_py_BlackMirrorMonitor
    plank_py_SovereignNeuron["SovereignNeuron"]
    class plank_py_SovereignNeuron cls;
    plank_py --> plank_py_SovereignNeuron
    plank_py_NeuroSovereign["NeuroSovereign"]
    class plank_py_NeuroSovereign cls;
    plank_py --> plank_py_NeuroSovereign
    plank_py_SovereignTrainer["SovereignTrainer"]
    class plank_py_SovereignTrainer cls;
    plank_py --> plank_py_SovereignTrainer
    plank_py_main["main"]
    class plank_py_main fn;
    plank_py --> plank_py_main
    install_sh["install.sh (sh)"]
    class install_sh mod;
    ext_torch["torch"]
    class ext_torch ext;
    apex14_py -.->|imports| ext_torch
    ext_torch_nn["torch.nn"]
    class ext_torch_nn ext;
    apex14_py -.->|imports| ext_torch_nn
    ext_torch_nn_functional["torch.nn.functional"]
    class ext_torch_nn_functional ext;
    apex14_py -.->|imports| ext_torch_nn_functional
    ext_torchvision["torchvision"]
    class ext_torchvision ext;
    apex14_py -.->|imports| ext_torchvision
    ext_torchvision_transforms["torchvision.transforms"]
    class ext_torchvision_transforms ext;
    apex14_py -.->|imports| ext_torchvision_transforms
    ext_numpy["numpy"]
    class ext_numpy ext;
    apex14_py -.->|imports| ext_numpy
    ext_pandas["pandas"]
    class ext_pandas ext;
    apex14_py -.->|imports| ext_pandas
    ext_os["os"]
    class ext_os ext;
    apex14_py -.->|imports| ext_os
    ext_warnings["warnings"]
    class ext_warnings ext;
    apex14_py -.->|imports| ext_warnings
    ext_typing["typing"]
    class ext_typing ext;
    apex14_py -.->|imports| ext_typing
    apex15_py -.->|imports| ext_torch
    apex15_py -.->|imports| ext_torch_nn
    apex15_py -.->|imports| ext_torch_nn_functional
    apex15_py -.->|imports| ext_torchvision
    apex15_py -.->|imports| ext_torchvision_transforms
    apex15_py -.->|imports| ext_numpy
    apex15_py -.->|imports| ext_pandas
    apex15_py -.->|imports| ext_os
    apex15_py -.->|imports| ext_warnings
    apex15_py -.->|imports| ext_typing
    apex16_py -.->|imports| ext_torch
    apex16_py -.->|imports| ext_torch_nn
    apex16_py -.->|imports| ext_torch_nn_functional
    apex16_py -.->|imports| ext_torchvision
    apex16_py -.->|imports| ext_torchvision_transforms
    apex16_py -.->|imports| ext_numpy
    apex16_py -.->|imports| ext_pandas
    apex16_py -.->|imports| ext_os
    apex16_py -.->|imports| ext_warnings
    apex16_py -.->|imports| ext_typing
    apex17_py -.->|imports| ext_torch
    apex17_py -.->|imports| ext_torch_nn
    apex17_py -.->|imports| ext_torch_nn_functional
    apex17_py -.->|imports| ext_torchvision
    apex17_py -.->|imports| ext_torchvision_transforms
    apex17_py -.->|imports| ext_numpy
    apex17_py -.->|imports| ext_pandas
    apex17_py -.->|imports| ext_os
    apex17_py -.->|imports| ext_warnings
    apex17_py -.->|imports| ext_typing
    apex18_py -.->|imports| ext_torch
    apex18_py -.->|imports| ext_torch_nn
    apex18_py -.->|imports| ext_torch_nn_functional
    apex18_py -.->|imports| ext_torchvision
    apex18_py -.->|imports| ext_torchvision_transforms
    apex18_py -.->|imports| ext_numpy
    apex18_py -.->|imports| ext_pandas
    apex18_py -.->|imports| ext_os
    apex18_py -.->|imports| ext_warnings
    apex18_py -.->|imports| ext_typing
    apex19_py -.->|imports| ext_torch
    apex19_py -.->|imports| ext_torch_nn
    apex19_py -.->|imports| ext_torch_nn_functional
    apex19_py -.->|imports| ext_torchvision
    apex19_py -.->|imports| ext_torchvision_transforms
    apex19_py -.->|imports| ext_numpy
    apex19_py -.->|imports| ext_pandas
    apex19_py -.->|imports| ext_os
    apex19_py -.->|imports| ext_warnings
    apex19_py -.->|imports| ext_typing
    apex20_py -.->|imports| ext_torch
    apex20_py -.->|imports| ext_torch_nn
    apex20_py -.->|imports| ext_torch_nn_functional
    apex20_py -.->|imports| ext_torchvision
    apex20_py -.->|imports| ext_torchvision_transforms
    apex20_py -.->|imports| ext_numpy
    apex20_py -.->|imports| ext_pandas
    apex20_py -.->|imports| ext_os
    apex20_py -.->|imports| ext_warnings
    apex20_py -.->|imports| ext_typing
    apex21_py -.->|imports| ext_torch
    apex21_py -.->|imports| ext_torch_nn
    apex21_py -.->|imports| ext_torch_nn_functional
    apex21_py -.->|imports| ext_torchvision
    apex21_py -.->|imports| ext_torchvision_transforms
    apex21_py -.->|imports| ext_numpy
    apex21_py -.->|imports| ext_pandas
    apex21_py -.->|imports| ext_os
    apex21_py -.->|imports| ext_warnings
    apex21_py -.->|imports| ext_typing
    apex22_py -.->|imports| ext_torch
    apex22_py -.->|imports| ext_torch_nn
    apex22_py -.->|imports| ext_torch_nn_functional
    apex22_py -.->|imports| ext_torchvision
    apex22_py -.->|imports| ext_torchvision_transforms
    apex22_py -.->|imports| ext_numpy
    apex22_py -.->|imports| ext_pandas
    apex22_py -.->|imports| ext_os
    apex22_py -.->|imports| ext_warnings
    apex22_py -.->|imports| ext_typing
    apex23_py -.->|imports| ext_torch
    apex23_py -.->|imports| ext_torch_nn
    apex23_py -.->|imports| ext_torch_nn_functional
    apex23_py -.->|imports| ext_torchvision
    apex23_py -.->|imports| ext_torchvision_transforms
    apex23_py -.->|imports| ext_numpy
    apex23_py -.->|imports| ext_pandas
    apex23_py -.->|imports| ext_os
    apex23_py -.->|imports| ext_warnings
    apex23_py -.->|imports| ext_typing
    apex24_py -.->|imports| ext_torch
    apex24_py -.->|imports| ext_torch_nn
    apex24_py -.->|imports| ext_torch_nn_functional
    apex24_py -.->|imports| ext_torchvision
    apex24_py -.->|imports| ext_torchvision_transforms
    apex24_py -.->|imports| ext_numpy
    apex24_py -.->|imports| ext_pandas
    apex24_py -.->|imports| ext_os
    apex24_py -.->|imports| ext_warnings
    apex24_py -.->|imports| ext_typing
    apex25_py -.->|imports| ext_torch
    apex25_py -.->|imports| ext_torch_nn
    apex25_py -.->|imports| ext_torch_nn_functional
    apex25_py -.->|imports| ext_torchvision
    apex25_py -.->|imports| ext_torchvision_transforms
    apex25_py -.->|imports| ext_numpy
    apex25_py -.->|imports| ext_pandas
    apex25_py -.->|imports| ext_os
    apex25_py -.->|imports| ext_warnings
    apex25_py -.->|imports| ext_typing
    apex26_py -.->|imports| ext_torch
    apex26_py -.->|imports| ext_torch_nn
    apex26_py -.->|imports| ext_torch_nn_functional
    apex26_py -.->|imports| ext_torchvision
    apex26_py -.->|imports| ext_torchvision_transforms
    apex26_py -.->|imports| ext_numpy
    apex26_py -.->|imports| ext_pandas
    ext_matplotlib_pyplot["matplotlib.pyplot"]
    class ext_matplotlib_pyplot ext;
    apex26_py -.->|imports| ext_matplotlib_pyplot
    apex26_py -.->|imports| ext_os
    ext_time["time"]
    class ext_time ext;
    apex26_py -.->|imports| ext_time
    ext_json["json"]
    class ext_json ext;
    apex26_py -.->|imports| ext_json
    ext_random["random"]
    class ext_random ext;
    apex26_py -.->|imports| ext_random
    ext_argparse["argparse"]
    class ext_argparse ext;
    apex26_py -.->|imports| ext_argparse
    apex26_py -.->|imports| ext_typing
    apex26_py -.->|imports| ext_warnings
    apex27_py -.->|imports| ext_torch
    apex27_py -.->|imports| ext_torch_nn
    apex27_py -.->|imports| ext_torch_nn_functional
    apex27_py -.->|imports| ext_torchvision
    apex27_py -.->|imports| ext_torchvision_transforms
    apex27_py -.->|imports| ext_numpy
    apex27_py -.->|imports| ext_pandas
    apex27_py -.->|imports| ext_matplotlib_pyplot
    apex27_py -.->|imports| ext_os
    apex27_py -.->|imports| ext_time
    apex27_py -.->|imports| ext_json
    apex27_py -.->|imports| ext_random
    apex27_py -.->|imports| ext_argparse
    apex27_py -.->|imports| ext_typing
    ext_collections["collections"]
    class ext_collections ext;
    apex27_py -.->|imports| ext_collections
    apex27_py -.->|imports| ext_warnings
    apex28_py -.->|imports| ext_torch
    apex28_py -.->|imports| ext_torch_nn
    apex28_py -.->|imports| ext_torch_nn_functional
    apex28_py -.->|imports| ext_torchvision
    apex28_py -.->|imports| ext_torchvision_transforms
    apex28_py -.->|imports| ext_numpy
    apex28_py -.->|imports| ext_pandas
    apex28_py -.->|imports| ext_matplotlib_pyplot
    apex28_py -.->|imports| ext_os
    apex28_py -.->|imports| ext_time
    apex28_py -.->|imports| ext_json
    apex28_py -.->|imports| ext_random
    apex28_py -.->|imports| ext_argparse
    apex28_py -.->|imports| ext_typing
    apex28_py -.->|imports| ext_warnings
    apex29_py -.->|imports| ext_torch
    apex29_py -.->|imports| ext_torch_nn
    apex29_py -.->|imports| ext_torch_nn_functional
    apex29_py -.->|imports| ext_torchvision
    apex29_py -.->|imports| ext_torchvision_transforms
    apex29_py -.->|imports| ext_numpy
    apex29_py -.->|imports| ext_pandas
    apex29_py -.->|imports| ext_matplotlib_pyplot
    apex29_py -.->|imports| ext_os
    apex29_py -.->|imports| ext_time
    apex29_py -.->|imports| ext_json
    apex29_py -.->|imports| ext_random
    apex29_py -.->|imports| ext_argparse
    apex29_py -.->|imports| ext_typing
    apex29_py -.->|imports| ext_warnings
    apex30_py -.->|imports| ext_torch
    apex30_py -.->|imports| ext_torch_nn
    apex30_py -.->|imports| ext_torch_nn_functional
    apex30_py -.->|imports| ext_torchvision
    apex30_py -.->|imports| ext_torchvision_transforms
    apex30_py -.->|imports| ext_numpy
    apex30_py -.->|imports| ext_pandas
    apex30_py -.->|imports| ext_matplotlib_pyplot
    apex30_py -.->|imports| ext_os
    apex30_py -.->|imports| ext_time
    apex30_py -.->|imports| ext_json
    apex30_py -.->|imports| ext_random
    apex30_py -.->|imports| ext_argparse
    apex30_py -.->|imports| ext_typing
    apex30_py -.->|imports| ext_warnings
    apex31_py -.->|imports| ext_torch
    apex31_py -.->|imports| ext_torch_nn
    apex31_py -.->|imports| ext_torch_nn_functional
    apex31_py -.->|imports| ext_torchvision
    apex31_py -.->|imports| ext_torchvision_transforms
    apex31_py -.->|imports| ext_numpy
    apex31_py -.->|imports| ext_pandas
    apex31_py -.->|imports| ext_matplotlib_pyplot
    apex31_py -.->|imports| ext_os
    apex31_py -.->|imports| ext_time
    apex31_py -.->|imports| ext_json
    apex31_py -.->|imports| ext_random
    apex31_py -.->|imports| ext_argparse
    apex31_py -.->|imports| ext_typing
    apex31_py -.->|imports| ext_warnings
    apex32_py -.->|imports| ext_torch
    apex32_py -.->|imports| ext_torch_nn
    apex32_py -.->|imports| ext_torch_nn_functional
    apex32_py -.->|imports| ext_torchvision
    apex32_py -.->|imports| ext_torchvision_transforms
    apex32_py -.->|imports| ext_numpy
    apex32_py -.->|imports| ext_pandas
    apex32_py -.->|imports| ext_os
    apex32_py -.->|imports| ext_warnings
    apex32_py -.->|imports| ext_typing
    apex33_py -.->|imports| ext_torch
    apex33_py -.->|imports| ext_torch_nn
    apex33_py -.->|imports| ext_torch_nn_functional
    apex33_py -.->|imports| ext_torchvision
    apex33_py -.->|imports| ext_torchvision_transforms
    apex33_py -.->|imports| ext_numpy
    apex33_py -.->|imports| ext_pandas
    apex33_py -.->|imports| ext_matplotlib_pyplot
    apex33_py -.->|imports| ext_os
    apex33_py -.->|imports| ext_json
    apex33_py -.->|imports| ext_random
    apex33_py -.->|imports| ext_argparse
    apex33_py -.->|imports| ext_typing
    apex33_py -.->|imports| ext_warnings
    apex34_py -.->|imports| ext_torch
    apex34_py -.->|imports| ext_torch_nn
    apex34_py -.->|imports| ext_torch_nn_functional
    apex34_py -.->|imports| ext_torchvision
    apex34_py -.->|imports| ext_torchvision_transforms
    apex34_py -.->|imports| ext_numpy
    apex34_py -.->|imports| ext_pandas
    apex34_py -.->|imports| ext_matplotlib_pyplot
    apex34_py -.->|imports| ext_os
    apex34_py -.->|imports| ext_json
    apex34_py -.->|imports| ext_random
    apex34_py -.->|imports| ext_argparse
    apex34_py -.->|imports| ext_typing
    apex34_py -.->|imports| ext_warnings
    apex35_py -.->|imports| ext_torch
    apex35_py -.->|imports| ext_torch_nn
    apex35_py -.->|imports| ext_torch_nn_functional
    apex35_py -.->|imports| ext_torchvision
    apex35_py -.->|imports| ext_torchvision_transforms
    apex35_py -.->|imports| ext_numpy
    apex35_py -.->|imports| ext_pandas
    apex35_py -.->|imports| ext_matplotlib_pyplot
    apex35_py -.->|imports| ext_os
    apex35_py -.->|imports| ext_json
    apex35_py -.->|imports| ext_random
    apex35_py -.->|imports| ext_argparse
    apex35_py -.->|imports| ext_typing
    apex35_py -.->|imports| ext_warnings
    app_py -.->|imports| ext_torch
    app_py -.->|imports| ext_torch_nn
    app_py -.->|imports| ext_torch_nn_functional
    app_py -.->|imports| ext_torchvision
    app_py -.->|imports| ext_torchvision_transforms
    app_py -.->|imports| ext_numpy
    app_py -.->|imports| ext_pandas
    app_py -.->|imports| ext_json
    app_py -.->|imports| ext_os
    app_py -.->|imports| ext_time
    app_py -.->|imports| ext_typing
    ext_sklearn_metrics_pairwise["sklearn.metrics.pairwise"]
    class ext_sklearn_metrics_pairwise ext;
    app_py -.->|imports| ext_sklearn_metrics_pairwise
    app_py -.->|imports| ext_warnings
    plank_py -.->|imports| ext_torch
    plank_py -.->|imports| ext_torch_nn
    plank_py -.->|imports| ext_torch_nn_functional
    plank_py -.->|imports| ext_torchvision
    plank_py -.->|imports| ext_torchvision_transforms
    plank_py -.->|imports| ext_numpy
    plank_py -.->|imports| ext_warnings
    plank10_py -.->|imports| ext_torch
    plank10_py -.->|imports| ext_torch_nn
    plank10_py -.->|imports| ext_torch_nn_functional
    plank10_py -.->|imports| ext_torchvision
    plank10_py -.->|imports| ext_torchvision_transforms
    plank10_py -.->|imports| ext_numpy
    plank10_py -.->|imports| ext_pandas
    plank10_py -.->|imports| ext_os
    plank10_py -.->|imports| ext_warnings
    plank10_py -.->|imports| ext_typing
    plank11_py -.->|imports| ext_torch
    plank11_py -.->|imports| ext_torch_nn
    plank11_py -.->|imports| ext_torch_nn_functional
    plank11_py -.->|imports| ext_torchvision
    plank11_py -.->|imports| ext_torchvision_transforms
    plank11_py -.->|imports| ext_numpy
    plank11_py -.->|imports| ext_pandas
    plank11_py -.->|imports| ext_os
    plank11_py -.->|imports| ext_warnings
    plank11_py -.->|imports| ext_typing
    plank12_py -.->|imports| ext_torch
    plank12_py -.->|imports| ext_torch_nn
    plank12_py -.->|imports| ext_torch_nn_functional
    plank12_py -.->|imports| ext_torchvision
    plank12_py -.->|imports| ext_torchvision_transforms
    plank12_py -.->|imports| ext_numpy
    plank12_py -.->|imports| ext_pandas
    plank12_py -.->|imports| ext_os
    plank12_py -.->|imports| ext_warnings
    plank12_py -.->|imports| ext_typing
    plank13_py -.->|imports| ext_torch
    plank13_py -.->|imports| ext_torch_nn
    plank13_py -.->|imports| ext_torch_nn_functional
    plank13_py -.->|imports| ext_torchvision
    plank13_py -.->|imports| ext_torchvision_transforms
    plank13_py -.->|imports| ext_numpy
    plank13_py -.->|imports| ext_pandas
    plank13_py -.->|imports| ext_os
    plank13_py -.->|imports| ext_warnings
    plank13_py -.->|imports| ext_typing
    plank2_py -.->|imports| ext_torch
    plank2_py -.->|imports| ext_torch_nn
    plank2_py -.->|imports| ext_torch_nn_functional
    plank2_py -.->|imports| ext_torchvision
    plank2_py -.->|imports| ext_torchvision_transforms
    plank2_py -.->|imports| ext_numpy
    plank2_py -.->|imports| ext_pandas
    plank2_py -.->|imports| ext_os
    plank2_py -.->|imports| ext_time
    plank2_py -.->|imports| ext_typing
    plank2_py -.->|imports| ext_warnings
    plank3_py -.->|imports| ext_torch
    plank3_py -.->|imports| ext_torch_nn
    plank3_py -.->|imports| ext_torch_nn_functional
    plank3_py -.->|imports| ext_torchvision
    plank3_py -.->|imports| ext_torchvision_transforms
    plank3_py -.->|imports| ext_numpy
    plank3_py -.->|imports| ext_pandas
    plank3_py -.->|imports| ext_os
    plank3_py -.->|imports| ext_time
    plank3_py -.->|imports| ext_typing
    plank3_py -.->|imports| ext_warnings
    plank4_py -.->|imports| ext_torch
    plank4_py -.->|imports| ext_torch_nn
    plank4_py -.->|imports| ext_torch_nn_functional
    plank4_py -.->|imports| ext_torchvision
    plank4_py -.->|imports| ext_torchvision_transforms
    plank4_py -.->|imports| ext_numpy
    plank4_py -.->|imports| ext_pandas
    plank4_py -.->|imports| ext_json
    plank4_py -.->|imports| ext_os
    plank4_py -.->|imports| ext_time
    plank4_py -.->|imports| ext_typing
    plank4_py -.->|imports| ext_warnings
    plank5_py -.->|imports| ext_torch
    plank5_py -.->|imports| ext_torch_nn
    plank5_py -.->|imports| ext_torch_nn_functional
    plank5_py -.->|imports| ext_torchvision
    plank5_py -.->|imports| ext_torchvision_transforms
    plank5_py -.->|imports| ext_numpy
    plank5_py -.->|imports| ext_pandas
    plank5_py -.->|imports| ext_json
    plank5_py -.->|imports| ext_os
    plank5_py -.->|imports| ext_warnings
    plank5_py -.->|imports| ext_typing
    plank5_py -.->|imports| ext_sklearn_metrics_pairwise
    plank5_py -.->|imports| ext_warnings
    plank6_py -.->|imports| ext_torch
    plank6_py -.->|imports| ext_torch_nn
    plank6_py -.->|imports| ext_torch_nn_functional
    plank6_py -.->|imports| ext_torchvision
    plank6_py -.->|imports| ext_torchvision_transforms
    plank6_py -.->|imports| ext_numpy
    plank6_py -.->|imports| ext_pandas
    plank6_py -.->|imports| ext_json
    plank6_py -.->|imports| ext_os
    plank6_py -.->|imports| ext_warnings
    plank6_py -.->|imports| ext_typing
    plank7_py -.->|imports| ext_torch
    plank7_py -.->|imports| ext_torch_nn
    plank7_py -.->|imports| ext_torch_nn_functional
    plank7_py -.->|imports| ext_torchvision
    plank7_py -.->|imports| ext_torchvision_transforms
    plank7_py -.->|imports| ext_numpy
    plank7_py -.->|imports| ext_pandas
    plank7_py -.->|imports| ext_json
    plank7_py -.->|imports| ext_os
    plank7_py -.->|imports| ext_warnings
    plank7_py -.->|imports| ext_typing
    plank8_py -.->|imports| ext_torch
    plank8_py -.->|imports| ext_torch_nn
    plank8_py -.->|imports| ext_torch_nn_functional
    plank8_py -.->|imports| ext_torchvision
    plank8_py -.->|imports| ext_torchvision_transforms
    plank8_py -.->|imports| ext_numpy
    plank8_py -.->|imports| ext_pandas
    plank8_py -.->|imports| ext_os
    plank8_py -.->|imports| ext_warnings
    plank8_py -.->|imports| ext_typing
    resmav2_1_py -.->|imports| ext_os
    ext_glob["glob"]
    class ext_glob ext;
    resmav2_1_py -.->|imports| ext_glob
    resmav2_1_py -.->|imports| ext_torch
    ext_zipfile["zipfile"]
    class ext_zipfile ext;
    resmav2_1_py -.->|imports| ext_zipfile
    ext_kagglehub["kagglehub"]
    class ext_kagglehub ext;
    resmav2_1_py -.->|imports| ext_kagglehub
    resmav2_1_py -.->|imports| ext_numpy
    resmav2_1_py -.->|imports| ext_pandas
    resmav2_1_py -.->|imports| ext_torch_nn
    resmav2_1_py -.->|imports| ext_torch_nn_functional
    ext_torch_geometric_nn["torch_geometric.nn"]
    class ext_torch_geometric_nn ext;
    resmav2_1_py -.->|imports| ext_torch_geometric_nn
    ext_torch_geometric_utils["torch_geometric.utils"]
    class ext_torch_geometric_utils ext;
    resmav2_1_py -.->|imports| ext_torch_geometric_utils
    ext_sklearn_preprocessing["sklearn.preprocessing"]
    class ext_sklearn_preprocessing ext;
    resmav2_1_py -.->|imports| ext_sklearn_preprocessing
    ext_sklearn_model_selection["sklearn.model_selection"]
    class ext_sklearn_model_selection ext;
    resmav2_1_py -.->|imports| ext_sklearn_model_selection
    ext_sklearn_metrics["sklearn.metrics"]
    class ext_sklearn_metrics ext;
    resmav2_1_py -.->|imports| ext_sklearn_metrics
    resmav2_1_py -.->|imports| ext_time
    resmav2_1_py -.->|imports| ext_warnings
```

---

## Architecture Reference

### PY (36 files)

#### `apex14.py`
**Path:** `apex14.py`

**Classs:**
- `GatedTokenMixer` (line 36)
- `PatchFeatureExtractor` (line 66) - *Extractor configurable para CIFAR-100.
use_mixer=True -> Apex (Syntactic)
use_mixer=False -> Blind Structural Baseline*
- `LotteryMLP` (line 111)
- `SpectralMonitor` (line 142)
- `OrthogonalEvolutionEngine` (line 158)
- `HierarchicalTrainer` (line 231)

**Functions:**
- `main` (line 336)
- `__init__` (line 37)
- `forward` (line 54)
- `__init__` (line 72)
- `freeze` (line 89)
- `unfreeze` (line 93)
- `forward` (line 97)
- `__init__` (line 112)
- `apply_masks` (line 123)
- `get_sparsity` (line 128)
- `forward` (line 133)
- `compute_metrics` (line 143)
- `__init__` (line 159)
- `_gradient_nudge_inheritance` (line 164)
- `_apply_rank_capping_shock` (line 198)
- `create_refined_offspring` (line 217)
- `__init__` (line 232)
- `_preprocess_batch` (line 240)
- `get_curriculum_dataset` (line 244)
- `train_single_chain` (line 250)

#### `apex15.py`
**Path:** `apex15.py`

**Classs:**
- `GatedTokenMixer` (line 88)
- `PatchFeatureExtractor` (line 105)
- `TaxonomicMLP` (line 143)
- `SpectralMonitor` (line 179)
- `TaxonomicTrainer` (line 195)

**Functions:**
- `compute_spectral_loss` (line 63) - *Penaliza la desalineación entre Entropía Espectral y Rango Efectivo.
Esta es la versión 'activa' de la métrica L.*
- `main` (line 367)
- `__init__` (line 89)
- `forward` (line 98)
- `__init__` (line 106)
- `freeze` (line 121)
- `unfreeze_mixer_only` (line 125)
- `forward` (line 131)
- `__init__` (line 144)
- `apply_masks` (line 158)
- `get_sparsity` (line 164)
- `forward` (line 169)
- `compute_metrics` (line 180)
- `__init__` (line 196)
- `_preprocess_batch` (line 203)
- `get_curriculum_dataset` (line 207)
- `train_single_chain` (line 213)

#### `apex16.py`
**Path:** `apex16.py`

**Classs:**
- `GatedTokenMixer` (line 82)
- `PatchFeatureExtractor` (line 99)
- `TaxonomicMLP` (line 137)
- `SpectralMonitor` (line 173)
- `TaxonomicTrainer` (line 189)

**Functions:**
- `compute_spectral_loss` (line 63) - *Penaliza la desalineación entre Entropía Espectral y Rango Efectivo.
v14.5: Se aplicará a pesos del MLP y del Mixer.*
- `main` (line 377)
- `__init__` (line 83)
- `forward` (line 92)
- `__init__` (line 100)
- `freeze` (line 115)
- `unfreeze_mixer_only` (line 119)
- `forward` (line 125)
- `__init__` (line 138)
- `apply_masks` (line 152)
- `get_sparsity` (line 158)
- `forward` (line 163)
- `compute_metrics` (line 174)
- `__init__` (line 190)
- `_preprocess_batch` (line 197)
- `get_curriculum_dataset` (line 201)
- `train_single_chain` (line 207)

#### `apex17.py`
**Path:** `apex17.py`

**Classs:**
- `GatedTokenMixer` (line 81)
- `PatchFeatureExtractor` (line 98)
- `TaxonomicMLP` (line 136)
- `SpectralMonitor` (line 172)
- `TaxonomicTrainer` (line 190)
- `CoarseCIFAR100` (line 368) - *Wrapper que convierte CIFAR100 en un problema de clasificación pura de 20 clases (Superclases).
Se usa para validar el inductive bias aprendido.*

**Functions:**
- `compute_spectral_loss` (line 65) - *v15.0: Optimization Objective for Spectral Control.*
- `run_hierarchy_benchmark` (line 378)
- `main` (line 428)
- `__init__` (line 82)
- `forward` (line 91)
- `__init__` (line 99)
- `freeze` (line 114)
- `unfreeze_mixer_only` (line 118)
- `forward` (line 124)
- `__init__` (line 137)
- `apply_masks` (line 151)
- `get_sparsity` (line 157)
- `forward` (line 162)
- `compute_metrics` (line 173) - *L_mon: Used for plotting and historical reporting, not optimization.*
- `__init__` (line 191)
- `_preprocess_batch` (line 198)
- `get_curriculum_dataset` (line 202)
- `train_single_chain` (line 208)
- `__getitem__` (line 373)
- `evaluate` (line 391)

#### `apex18.py`
**Path:** `apex18.py`

**Classs:**
- `GatedTokenMixer` (line 81)
- `PatchFeatureExtractor` (line 98)
- `TaxonomicMLP` (line 136)
- `SpectralMonitor` (line 172)
- `TaxonomicTrainer` (line 188)
- `CoarseCIFAR100` (line 369)

**Functions:**
- `compute_spectral_loss` (line 65) - *v15.1: Optimization Objective for Spectral Control (Applied to both APEX and BLIND).*
- `run_hierarchy_benchmark` (line 374)
- `main` (line 420)
- `__init__` (line 82)
- `forward` (line 91)
- `__init__` (line 99)
- `freeze` (line 114)
- `unfreeze_mixer_only` (line 118)
- `forward` (line 124)
- `__init__` (line 137)
- `apply_masks` (line 151)
- `get_sparsity` (line 157)
- `forward` (line 162)
- `compute_metrics` (line 173)
- `__init__` (line 189)
- `_preprocess_batch` (line 196)
- `get_curriculum_dataset` (line 200)
- `train_single_chain` (line 206)
- `__getitem__` (line 370)
- `evaluate` (line 386)

#### `apex19.py`
**Path:** `apex19.py`

**Classs:**
- `GatedTokenMixer` (line 84)
- `PatchFeatureExtractor` (line 101)
- `TaxonomicMLP` (line 139)
- `SpectralMonitor` (line 175)
- `TaxonomicTrainer` (line 211)
- `CoarseCIFAR100` (line 394)

**Functions:**
- `compute_spectral_loss` (line 68) - *L_opt: Optimization Objective for Structural Control.*
- `run_hierarchy_benchmark` (line 399)
- `main` (line 445)
- `__init__` (line 85)
- `forward` (line 94)
- `__init__` (line 102)
- `freeze` (line 117)
- `unfreeze_mixer_only` (line 121)
- `forward` (line 127)
- `__init__` (line 140)
- `apply_masks` (line 154)
- `get_sparsity` (line 160)
- `forward` (line 165)
- `compute_metrics` (line 176)
- `compute_topology_ratio` (line 188) - *v15.2: Calcula el ratio R = L_opt / L_mon.
Valores bajos indican alineación estable.
Valores altos o erráticos indican transición de fase (Grokking/Collapse).*
- `__init__` (line 212)
- `_preprocess_batch` (line 219)
- `get_curriculum_dataset` (line 223)
- `train_single_chain` (line 229)
- `__getitem__` (line 395)
- `evaluate` (line 411)

#### `apex20.py`
**Path:** `apex20.py`

**Classs:**
- `GatedTokenMixer` (line 85)
- `PatchFeatureExtractor` (line 102)
- `TaxonomicMLP` (line 140)
- `SpectralMonitor` (line 176)
- `TaxonomicTrainer` (line 242)
- `CoarseCIFAR100` (line 426)

**Functions:**
- `compute_spectral_loss` (line 69) - *L_opt: Optimization Objective for Structural Control.*
- `run_hierarchy_benchmark` (line 431)
- `main` (line 490)
- `__init__` (line 86)
- `forward` (line 95)
- `__init__` (line 103)
- `freeze` (line 118)
- `unfreeze_mixer_only` (line 122)
- `forward` (line 128)
- `__init__` (line 141)
- `apply_masks` (line 155)
- `get_sparsity` (line 161)
- `forward` (line 166)
- `compute_metrics` (line 177)
- `detect_phase_state` (line 190) - *v15.3: Detects phase state based on relative deviation, not absolute value.
Returns: 'STABLE', 'SHIFTING', or 'INIT'*
- `compute_topology_ratio` (line 216) - *v15.3: Returns L_opt components and Total Ratio.
Ratio = L_opt_Total / L_mon(Downstream)*
- `__init__` (line 243)
- `_preprocess_batch` (line 250)
- `get_curriculum_dataset` (line 254)
- `train_single_chain` (line 260)
- `__getitem__` (line 427)
- `evaluate` (line 443)

#### `apex21.py`
**Path:** `apex21.py`

**Classs:**
- `GatedTokenMixer` (line 88)
- `PatchFeatureExtractor` (line 105)
- `TaxonomicMLP` (line 143)
- `SpectralMonitor` (line 179)
- `TopologyController` (line 228) - *v15.4: Manages Active Interventions to break stagnation.*
- `TaxonomicTrainer` (line 270)
- `CoarseCIFAR100` (line 459)

**Functions:**
- `compute_spectral_loss` (line 73)
- `run_hierarchy_benchmark` (line 464)
- `main` (line 513)
- `__init__` (line 89)
- `forward` (line 98)
- `__init__` (line 106)
- `freeze` (line 121)
- `unfreeze_mixer_only` (line 125)
- `forward` (line 131)
- `__init__` (line 144)
- `apply_masks` (line 158)
- `get_sparsity` (line 164)
- `forward` (line 169)
- `compute_metrics` (line 180)
- `detect_phase_state` (line 192)
- `compute_topology_ratio` (line 204) - *v15.3: Returns L_opt components and Total Ratio.
Ratio = L_opt_Total / L_mon(Downstream)*
- `__init__` (line 230)
- `check_intervention` (line 233) - *Decides whether to intervene.
Returns 'INTERVENE' if action is taken, 'NONE' otherwise.*
- `perturb_mixer` (line 255) - *Causal Intervention: Inject topological noise to force phase shift.*
- `__init__` (line 271)
- `_preprocess_batch` (line 279)
- `get_curriculum_dataset` (line 283)
- `train_single_chain` (line 289)
- `__getitem__` (line 460)
- `evaluate` (line 476)

#### `apex22.py`
**Path:** `apex22.py`

**Classs:**
- `GatedTokenMixer` (line 88)
- `PatchFeatureExtractor` (line 105)
- `TaxonomicMLP` (line 143)
- `SpectralMonitor` (line 179)
- `TopologyController` (line 204) - *v15.5: Manages Targeted Spectral Interventions (Surgery).*
- `TaxonomicTrainer` (line 273)
- `CoarseCIFAR100` (line 459)

**Functions:**
- `compute_spectral_loss` (line 73)
- `run_hierarchy_benchmark` (line 464)
- `main` (line 513)
- `__init__` (line 89)
- `forward` (line 98)
- `__init__` (line 106)
- `freeze` (line 121)
- `unfreeze_mixer_only` (line 125)
- `forward` (line 131)
- `__init__` (line 144)
- `apply_masks` (line 158)
- `get_sparsity` (line 164)
- `forward` (line 169)
- `compute_metrics` (line 180)
- `detect_phase_state` (line 192)
- `__init__` (line 206)
- `check_intervention` (line 209)
- `perturb_mixer_targeted` (line 226) - *v15.5: Targeted Phase Surgery.
Injects noise ONLY in the nullspace of the dominant spectral subspace.
Preserves existing structure while forcing exploration of latent dimensions.*
- `__init__` (line 274)
- `_preprocess_batch` (line 282)
- `get_curriculum_dataset` (line 286)
- `train_single_chain` (line 292)
- `__getitem__` (line 460)
- `evaluate` (line 476)

#### `apex23.py`
**Path:** `apex23.py`

**Classs:**
- `GatedTokenMixer` (line 91)
- `PatchFeatureExtractor` (line 108)
- `TaxonomicMLP` (line 146)
- `SpectralMonitor` (line 182)
- `TopologyController` (line 228)
- `TaxonomicTrainer` (line 306)
- `CoarseCIFAR100` (line 493)

**Functions:**
- `compute_spectral_loss` (line 74) - *L_opt: Computes the discrepancy between spectral entropy and effective rank.*
- `run_hierarchy_benchmark` (line 498)
- `main` (line 544)
- `__init__` (line 92)
- `forward` (line 101)
- `__init__` (line 109)
- `freeze` (line 124)
- `unfreeze_mixer_only` (line 128)
- `forward` (line 134)
- `__init__` (line 147)
- `apply_masks` (line 161)
- `get_sparsity` (line 167)
- `forward` (line 172)
- `compute_metrics` (line 183) - *L_mon: Legacy reporting metric.*
- `detect_phase_state` (line 196)
- `compute_topology_ratio` (line 208) - *Calculates Topo_R = L_opt / L_mon.
L_opt is the active optimization energy (FC1 + Mixer).
L_mon is the passive structural metric.*
- `__init__` (line 229)
- `check_intervention` (line 232) - *Decides if intervention is needed based on Phase and Performance.*
- `perturb_mixer_targeted` (line 250) - *v15.5: Targeted Spectral Surgery.
Injects noise in the nullspace of the dominant spectral subspace to explore
latent modes without destroying learned hierarchy.*
- `__init__` (line 307)
- `_preprocess_batch` (line 315)
- `get_curriculum_dataset` (line 319)
- `train_single_chain` (line 325)
- `__getitem__` (line 494)
- `evaluate` (line 510)

#### `apex24.py`
**Path:** `apex24.py`

**Classs:**
- `GatedTokenMixer` (line 92)
- `PatchFeatureExtractor` (line 109)
- `TaxonomicMLP` (line 147)
- `SpectralMonitor` (line 183)
- `TopologyController` (line 229)
- `TaxonomicTrainer` (line 339)
- `CoarseCIFAR100` (line 527)

**Functions:**
- `compute_spectral_loss` (line 75) - *L_opt: Computes the discrepancy between spectral entropy and effective rank.*
- `run_hierarchy_benchmark` (line 532)
- `main` (line 578)
- `__init__` (line 93)
- `forward` (line 102)
- `__init__` (line 110)
- `freeze` (line 125)
- `unfreeze_mixer_only` (line 129)
- `forward` (line 135)
- `__init__` (line 148)
- `apply_masks` (line 162)
- `get_sparsity` (line 168)
- `forward` (line 173)
- `compute_metrics` (line 184) - *L_mon: Legacy reporting metric.*
- `detect_phase_state` (line 197)
- `compute_topology_ratio` (line 209) - *Calculates Topo_R = L_opt / L_mon.
L_opt is the active optimization energy (FC1 + Mixer).
L_mon is the passive structural metric.*
- `__init__` (line 230)
- `check_intervention` (line 235) - *v15.5 Final: Geometric Mismatch Detection.
Triggers intervention if Topo_R (Structure) changes but Coarse Acc (Semantics) does not.*
- `perturb_mixer_targeted` (line 284) - *v15.5: Targeted Spectral Surgery.
Injects noise in the nullspace of the dominant spectral subspace to explore
latent modes without destroying learned hierarchy.*
- `__init__` (line 340)
- `_preprocess_batch` (line 348)
- `get_curriculum_dataset` (line 352)
- `train_single_chain` (line 358)
- `__getitem__` (line 528)
- `evaluate` (line 544)

#### `apex25.py`
**Path:** `apex25.py`

**Classs:**
- `GatedTokenMixer` (line 89)
- `PatchFeatureExtractor` (line 106)
- `TaxonomicMLP` (line 144)
- `SpectralMonitor` (line 180)
- `TopologyController` (line 224)
- `TaxonomicTrainer` (line 316)
- `CoarseCIFAR100` (line 504)

**Functions:**
- `compute_spectral_loss` (line 75) - *L_opt: Computes the discrepancy between spectral entropy and effective rank.*
- `run_hierarchy_benchmark` (line 509)
- `main` (line 555)
- `__init__` (line 90)
- `forward` (line 99)
- `__init__` (line 107)
- `freeze` (line 122)
- `unfreeze_mixer_only` (line 126)
- `forward` (line 132)
- `__init__` (line 145)
- `apply_masks` (line 159)
- `get_sparsity` (line 165)
- `forward` (line 170)
- `compute_metrics` (line 181) - *L_mon: Legacy reporting metric.*
- `detect_phase_state` (line 194)
- `compute_topology_ratio` (line 206) - *Calculates Topo_R = L_opt / L_mon.*
- `__init__` (line 225)
- `check_intervention` (line 230) - *v15.5 Final: Geometric Mismatch Detection.
Triggers intervention if Topo_R (Structure) changes but Coarse Acc (Semantics) does not.*
- `perturb_mixer_targeted` (line 277) - *v15.5: Targeted Spectral Surgery.
Injects noise in the nullspace of the dominant spectral subspace.*
- `__init__` (line 317)
- `_preprocess_batch` (line 325)
- `get_curriculum_dataset` (line 329)
- `train_single_chain` (line 335)
- `__getitem__` (line 505)
- `evaluate` (line 521)

#### `apex26.py`
**Path:** `apex26.py`

**Classs:**
- `GatedTokenMixer` (line 89) - *Efficient token mixer with gating mechanism for feature interaction*
- `PatchFeatureExtractor` (line 146) - *Efficient patch-based feature extractor with optional token mixing*
- `TaxonomicMLP` (line 207) - *Sparse MLP with taxonomic heads for hierarchical learning*
- `SpectralMonitor` (line 281) - *Monitor spectral properties of weight matrices for evolutionary guidance*
- `TopologyController` (line 302) - *Advanced controller for targeted spectral surgery*
- `EvolutionaryEngine` (line 410) - *Engine for evolving neural networks through spectral refinement*
- `CoarseCIFAR100` (line 480) - *Wrapper that converts CIFAR100 into a pure 20-class classification problem (Superclasses).
Used to validate learned inductive bias.*
- `EvolutionaryTrainer` (line 544) - *Framework for evolutionary training with statistical validation*

**Functions:**
- `set_seed` (line 43) - *Ensure full reproducibility across runs*
- `compute_spectral_loss` (line 265) - *Optimization Objective for Spectral Control (L_opt)*
- `run_hierarchy_benchmark` (line 490) - *Run hierarchy stress test to validate inductive bias transfer*
- `parse_args` (line 1170)
- `main` (line 1181) - *Main execution function*
- `__init__` (line 91)
- `_init_weights` (line 114) - *Initialize weights for stable training*
- `forward` (line 128) - *Input:  [B, num_patches, embed_dim]
Output: [B, num_patches, embed_dim]*
- `__init__` (line 148)
- `freeze` (line 177) - *Freeze all parameters for transfer learning*
- `unfreeze_mixer_only` (line 183) - *Unfreeze only the mixer parameters for fine-tuning*
- `forward` (line 190) - *Input:  [B, C, H, W]
Output: [B, embed_dim]*
- `__init__` (line 209)
- `apply_masks` (line 232) - *Apply sparsity masks to weights*
- `get_sparsity` (line 239) - *Calculate overall sparsity percentage*
- `forward` (line 245) - *Input:  [B, input_dim]
Output: ([B, num_classes], [B, num_superclasses])*
- `__init__` (line 283)
- `compute_metrics` (line 286) - *Compute spectral coherence metrics*
- `__init__` (line 304)
- `detect_phase_state` (line 314) - *Detect phase state based on topology ratio history*
- `check_intervention` (line 332) - *Check if intervention is needed based on geometric mismatch detection*
- `perturb_mixer_targeted` (line 377) - *Targeted Spectral Surgery: Inject noise in the nullspace of dominant subspace*
- `__init__` (line 412)
- `apply_rank_capping` (line 417) - *Apply rank capping shock to prevent over-specialization*
- `create_offspring` (line 434) - *Create refined offspring through gradient-based inheritance*
- `__getitem__` (line 485)
- `evaluate` (line 503)
- `__init__` (line 546)
- `load_data` (line 572) - *Load curriculum dataset based on evolutionary cycle*
- `train_model` (line 614) - *Train model with evolutionary pressure and hierarchical learning*
- `compute_topology_ratio` (line 837) - *Calculates Topo_R = L_opt / L_mon.*
- `run_evolution` (line 855) - *Run full evolutionary experiment with statistical validation*
- `_save_results` (line 1027) - *Save results to files*
- `_plot_results` (line 1081) - *Create publication-quality plots*

#### `apex27.py`
**Path:** `apex27.py`

**Classs:**
- `GatedTokenMixer` (line 83) - *Efficient token mixer with gating mechanism*
- `PatchFeatureExtractor` (line 124) - *Efficient patch-based feature extractor*
- `TaxonomicMLP` (line 170) - *Sparse MLP with taxonomic heads*
- `DynamicThresholdController` (line 233) - *v18 Improvement: Replaces fixed TARGET_COARSE_V with adaptive logic.
Triggers intervention if current performance stagnates relative to its own history.*
- `SpectralMonitor` (line 255) - *Monitor spectral properties*
- `TopologyController` (line 275) - *v18 Improvement: Advanced controller with Adaptive Thresholding.
Implements Targeted Spectral Surgery with Nullspace Injection.*
- `IterativeRefinementEngine` (line 369) - *v18: Engine for iterative refinement (formerly Evolutionary)*
- `IterativeTrainer` (line 416) - *Framework for Iterative Refinement with v18 Adaptive Control*

**Functions:**
- `set_seed` (line 38) - *Ensure full reproducibility across runs*
- `compute_spectral_loss` (line 217) - *Optimization Objective for Spectral Control (L_opt)*
- `parse_args` (line 900)
- `main` (line 914)
- `__init__` (line 85)
- `_init_weights` (line 104)
- `forward` (line 116)
- `__init__` (line 126)
- `freeze` (line 151)
- `unfreeze_mixer_only` (line 156)
- `forward` (line 162)
- `__init__` (line 172)
- `apply_masks` (line 192)
- `get_sparsity` (line 198)
- `forward` (line 203)
- `__init__` (line 238)
- `update` (line 243)
- `is_stagnant` (line 246)
- `__init__` (line 257)
- `compute_metrics` (line 260)
- `__init__` (line 280)
- `check_intervention` (line 291)
- `perturb_mixer_targeted` (line 336) - *Targeted Spectral Surgery: Inject noise in the nullspace of dominant subspace*
- `__init__` (line 371)
- `create_offspring` (line 374) - *Create refined offspring through gradient-based inheritance*
- `__init__` (line 418)
- `load_data` (line 441)
- `train_model` (line 479)
- `compute_topology_ratio` (line 683)
- `run_refinement` (line 696)
- `_save_results` (line 809)
- `_plot_results_v18` (line 815)

#### `apex28.py`
**Path:** `apex28.py`

**Classs:**
- `GatedTokenMixer` (line 93) - *Efficient token mixer with gating mechanism for feature interaction*
- `PatchFeatureExtractor` (line 150) - *Efficient patch-based feature extractor with optional token mixing*
- `TaxonomicMLP` (line 211) - *Sparse MLP with taxonomic heads for hierarchical learning*
- `SpectralMonitor` (line 285) - *Monitor spectral properties with adaptive analysis*
- `AdaptiveTopologyController` (line 316) - *Self-referential controller using semantic-plasticity ratio*
- `IterativeRefinementEngine` (line 415) - *Engine for iterative refinement through spectral control*
- `CoarseCIFAR100` (line 484) - *Wrapper that converts CIFAR100 into a pure 20-class classification problem (Superclasses).
Used to validate learned inductive bias.*
- `IterativeRefinementTrainer` (line 611) - *Framework for iterative refinement with statistical validation*

**Functions:**
- `set_seed` (line 47) - *Ensure full reproducibility across runs*
- `compute_spectral_loss` (line 269) - *Optimization Objective for Spectral Control (L_opt)*
- `run_hierarchy_benchmark` (line 494) - *Run hierarchy stress test to validate inductive bias transfer*
- `visualize_singular_values` (line 548) - *Generate publication-quality singular value visualizations*
- `parse_args` (line 1270)
- `main` (line 1282) - *Main execution function*
- `__init__` (line 95)
- `_init_weights` (line 118) - *Initialize weights for stable training*
- `forward` (line 132) - *Input:  [B, num_patches, embed_dim]
Output: [B, num_patches, embed_dim]*
- `__init__` (line 152)
- `freeze` (line 181) - *Freeze all parameters for transfer learning*
- `unfreeze_mixer_only` (line 187) - *Unfreeze only the mixer parameters for fine-tuning*
- `forward` (line 194) - *Input:  [B, C, H, W]
Output: [B, embed_dim]*
- `__init__` (line 213)
- `apply_masks` (line 236) - *Apply sparsity masks to weights*
- `get_sparsity` (line 243) - *Calculate overall sparsity percentage*
- `forward` (line 249) - *Input:  [B, input_dim]
Output: ([B, num_classes], [B, num_superclasses])*
- `__init__` (line 287)
- `compute_metrics` (line 290) - *Compute spectral coherence metrics*
- `get_singular_values` (line 306) - *Get singular values for visualization*
- `__init__` (line 318)
- `compute_semantic_plasticity_ratio` (line 330) - *Compute ratio of semantic gain to structural change*
- `detect_intervention_need` (line 343) - *Determine if intervention is needed using adaptive criteria*
- `update_history` (line 365) - *Update history for adaptive control*
- `perturb_mixer_targeted` (line 382) - *Targeted Spectral Surgery: Inject noise in the nullspace of dominant subspace*
- `__init__` (line 417)
- `apply_rank_capping` (line 421) - *Apply rank capping shock to prevent over-specialization*
- `create_refined_model` (line 438) - *Create refined model through gradient-based inheritance*
- `__getitem__` (line 489)
- `evaluate` (line 507)
- `get_singular_values` (line 556) - *Get singular values from model weights*
- `__init__` (line 613)
- `load_data` (line 639) - *Load curriculum dataset based on refinement cycle*
- `train_model` (line 681) - *Train model with iterative refinement and hierarchical learning*
- `detect_phase_state` (line 908) - *Detect phase state based on topology ratio history*
- `compute_topology_ratio` (line 926) - *Calculates Topo_R = L_opt / L_mon.*
- `run_refinement` (line 944) - *Run full iterative refinement experiment with statistical validation*
- `_save_results` (line 1126) - *Save results to files*
- `_plot_results` (line 1181) - *Create publication-quality plots*

#### `apex29.py`
**Path:** `apex29.py`

**Classs:**
- `GatedTokenMixer` (line 86) - *Efficient token mixer with gating mechanism for feature interaction*
- `PatchFeatureExtractor` (line 143) - *Efficient patch-based feature extractor with optional token mixing*
- `TaxonomicMLP` (line 204) - *Sparse MLP with taxonomic heads for hierarchical learning*
- `SpectralMonitor` (line 278) - *Monitor spectral properties with adaptive analysis*
- `AdaptiveTopologyController` (line 309) - *Self-referential controller using semantic-plasticity ratio*
- `IterativeRefinementEngine` (line 408) - *Engine for iterative refinement through spectral control*
- `CoarseCIFAR100` (line 476) - *Wrapper that converts CIFAR100 into a pure 20-class classification problem (Superclasses).
Used to validate learned inductive bias.*
- `IterativeRefinementTrainer` (line 606) - *Framework for iterative refinement with statistical validation*

**Functions:**
- `set_seed` (line 43) - *Ensure full reproducibility across runs*
- `compute_spectral_loss` (line 262) - *Optimization Objective for Spectral Control (L_opt)*
- `run_hierarchy_benchmark` (line 486) - *Run hierarchy stress test to validate inductive bias transfer*
- `visualize_singular_values` (line 543) - *Generate publication-quality singular value visualizations*
- `parse_args` (line 1244)
- `main` (line 1255) - *Main execution function*
- `__init__` (line 88)
- `_init_weights` (line 111) - *Initialize weights for stable training*
- `forward` (line 125) - *Input:  [B, num_patches, embed_dim]
Output: [B, num_patches, embed_dim]*
- `__init__` (line 145)
- `freeze` (line 174) - *Freeze all parameters for transfer learning*
- `unfreeze_mixer_only` (line 180) - *Unfreeze only the mixer parameters for fine-tuning*
- `forward` (line 187) - *Input:  [B, C, H, W]
Output: [B, embed_dim]*
- `__init__` (line 206)
- `apply_masks` (line 229) - *Apply sparsity masks to weights*
- `get_sparsity` (line 236) - *Calculate overall sparsity percentage*
- `forward` (line 242) - *Input:  [B, input_dim]
Output: ([B, num_classes], [B, num_superclasses])*
- `__init__` (line 280)
- `compute_metrics` (line 283) - *Compute spectral coherence metrics*
- `get_singular_values` (line 299) - *Get singular values for visualization*
- `__init__` (line 311)
- `compute_semantic_plasticity_ratio` (line 323) - *Compute ratio of semantic gain to structural change*
- `detect_intervention_need` (line 336) - *Determine if intervention is needed using adaptive criteria*
- `update_history` (line 358) - *Update history for adaptive control*
- `perturb_mixer_targeted` (line 375) - *Targeted Spectral Surgery: Inject noise in the nullspace of dominant subspace*
- `__init__` (line 410)
- `apply_rank_capping` (line 414) - *Apply rank capping shock to prevent over-specialization*
- `create_refined_model` (line 430) - *Create refined model through gradient-based inheritance*
- `__getitem__` (line 481)
- `evaluate` (line 499)
- `get_singular_values` (line 551) - *Get singular values from model weights*
- `__init__` (line 608)
- `load_data` (line 634) - *Load curriculum dataset based on refinement cycle*
- `train_model` (line 674) - *Train model with iterative refinement and hierarchical learning*
- `detect_phase_state` (line 894) - *Detect phase state based on topology ratio history*
- `compute_topology_ratio` (line 912) - *Calculates Topo_R = L_opt / L_mon.*
- `run_refinement` (line 931) - *Run full iterative refinement experiment with statistical validation*
- `_save_results` (line 1109) - *Save results to files*
- `_plot_results` (line 1164) - *Create publication-quality plots*

#### `apex30.py`
**Path:** `apex30.py`

**Classs:**
- `GatedTokenMixer` (line 87) - *Efficient token mixer with gating mechanism for feature interaction*
- `PatchFeatureExtractor` (line 142)
- `TaxonomicMLP` (line 185) - *Sparse MLP with taxonomic heads for hierarchical learning*
- `SpectralMonitor` (line 248) - *Monitor spectral properties with adaptive analysis*
- `AdaptiveTopologyController` (line 277) - *Self-referential controller using semantic-plasticity ratio*
- `IterativeRefinementEngine` (line 397) - *Engine for iterative refinement through spectral control*
- `CoarseCIFAR100` (line 459)
- `IterativeRefinementTrainer` (line 570)

**Functions:**
- `set_seed` (line 43) - *Ensure full reproducibility across runs*
- `compute_spectral_loss` (line 232) - *Optimization Objective for Spectral Control (L_opt)*
- `run_hierarchy_benchmark` (line 464)
- `visualize_singular_values` (line 510) - *v18.1 FIX: Now accepts 'monitor' explicitly.
Generate publication-quality singular value visualizations.*
- `parse_args` (line 1062)
- `main` (line 1071)
- `__init__` (line 89)
- `_init_weights` (line 111)
- `forward` (line 134)
- `__init__` (line 143)
- `freeze` (line 164)
- `unfreeze_mixer_only` (line 169)
- `forward` (line 175)
- `__init__` (line 187)
- `apply_masks` (line 207)
- `get_sparsity` (line 213)
- `forward` (line 218)
- `__init__` (line 250)
- `compute_metrics` (line 253)
- `get_singular_values` (line 268)
- `__init__` (line 279)
- `compute_semantic_plasticity_ratio` (line 295) - *Compute ratio of semantic gain to structural change*
- `detect_intervention_need` (line 308) - *Determine if intervention is needed using adaptive criteria.
Implements hysteresis to avoid intervention during active SHIFTING phases.*
- `update_history` (line 348) - *Update history for adaptive control*
- `perturb_mixer_targeted` (line 363) - *Targeted Spectral Surgery: Inject noise in the nullspace of dominant subspace*
- `__init__` (line 399)
- `apply_rank_capping` (line 403)
- `create_refined_model` (line 419)
- `__getitem__` (line 460)
- `evaluate` (line 471)
- `get_singular_values` (line 521)
- `__init__` (line 571)
- `load_data` (line 593)
- `train_model` (line 617)
- `detect_phase_state` (line 797)
- `compute_topology_ratio` (line 813)
- `run_refinement` (line 828)
- `_save_results` (line 956)
- `_plot_results` (line 983) - *v18.1 FIX: Explicitly accepts best_overall to fix scope bug.
Create publication-quality plots.*

#### `apex31.py`
**Path:** `apex31.py`

**Classs:**
- `GatedTokenMixer` (line 89) - *Efficient token mixer with gating mechanism for feature interaction*
- `PatchFeatureExtractor` (line 144)
- `TaxonomicMLP` (line 187) - *Sparse MLP with taxonomic heads for hierarchical learning*
- `SpectralMonitor` (line 250) - *Monitor spectral properties with adaptive analysis*
- `AdaptiveTopologyController` (line 279) - *Self-referential controller using semantic-plasticity ratio*
- `IterativeRefinementEngine` (line 404) - *Engine for iterative refinement through spectral control*
- `CoarseCIFAR100` (line 470)
- `IterativeRefinementTrainer` (line 581)

**Functions:**
- `set_seed` (line 45) - *Ensure full reproducibility across runs*
- `compute_spectral_loss` (line 234) - *Optimization Objective for Spectral Control (L_opt)*
- `run_hierarchy_benchmark` (line 475)
- `visualize_singular_values` (line 521) - *v18.1 FIX: Now accepts 'monitor' explicitly.
Generate publication-quality singular value visualizations.*
- `parse_args` (line 1073)
- `main` (line 1082)
- `__init__` (line 91)
- `_init_weights` (line 113)
- `forward` (line 136)
- `__init__` (line 145)
- `freeze` (line 166)
- `unfreeze_mixer_only` (line 171)
- `forward` (line 177)
- `__init__` (line 189)
- `apply_masks` (line 209)
- `get_sparsity` (line 215)
- `forward` (line 220)
- `__init__` (line 252)
- `compute_metrics` (line 255)
- `get_singular_values` (line 270)
- `__init__` (line 281)
- `compute_semantic_plasticity_ratio` (line 297) - *Compute ratio of semantic gain to structural change*
- `detect_intervention_need` (line 310) - *Determine if intervention is needed using adaptive criteria.
Implements hysteresis to avoid intervention during active SHIFTING phases.*
- `update_history` (line 354) - *Update history for adaptive control*
- `perturb_mixer_targeted` (line 369) - *Targeted Spectral Surgery: Inject noise in the nullspace of dominant subspace*
- `__init__` (line 406)
- `apply_rank_capping` (line 410)
- `create_refined_model` (line 430)
- `__getitem__` (line 471)
- `evaluate` (line 482)
- `get_singular_values` (line 532)
- `__init__` (line 582)
- `load_data` (line 604)
- `train_model` (line 628)
- `detect_phase_state` (line 808)
- `compute_topology_ratio` (line 824)
- `run_refinement` (line 839)
- `_save_results` (line 967)
- `_plot_results` (line 994) - *v18.1 FIX: Explicitly accepts best_overall to fix scope bug.
Create publication-quality plots.*

#### `apex32.py`
**Path:** `apex32.py`

**Classs:**
- `GatedTokenMixer` (line 92)
- `PatchFeatureExtractor` (line 109)
- `TaxonomicMLP` (line 148)
- `SpectralMonitor` (line 187)
- `TaxonomicTrainer` (line 226)
- `CoarseCIFAR100` (line 404)

**Functions:**
- `compute_spectral_loss` (line 76) - *L_opt: Optimization Objective for Structural Control.*
- `run_hierarchy_benchmark` (line 409)
- `main` (line 455)
- `__init__` (line 93)
- `forward` (line 102)
- `__init__` (line 110)
- `freeze` (line 126)
- `unfreeze_mixer_only` (line 130)
- `forward` (line 136)
- `__init__` (line 149)
- `apply_masks` (line 164) - *Zero out weights based on masks. Runs on device (CUDA).*
- `get_sparsity` (line 171)
- `forward` (line 176)
- `compute_metrics` (line 188)
- `detect_phase_state` (line 200)
- `compute_topology_ratio` (line 211)
- `__init__` (line 227)
- `_preprocess_batch` (line 234)
- `get_curriculum_dataset` (line 238)
- `train_single_chain` (line 244)
- `__getitem__` (line 405)
- `evaluate` (line 422)

#### `apex33.py`
**Path:** `apex33.py`

**Classs:**
- `GatedTokenMixer` (line 72) - *Efficient token mixer with high-std initialization for exploration*
- `PatchFeatureExtractor` (line 118)
- `TaxonomicMLP` (line 150) - *Sparse MLP with taxonomic heads*
- `SpectralMonitor` (line 201)
- `AdaptiveTopologyController` (line 227) - *Self-referential controller with Hysteresis*
- `IterativeRefinementEngine` (line 310)
- `IterativeRefinementTrainer` (line 343)
- `CoarseCIFAR100` (line 802)

**Functions:**
- `set_seed` (line 41) - *Ensure full reproducibility across runs*
- `compute_spectral_loss` (line 190)
- `run_hierarchy_benchmark` (line 807)
- `visualize_singular_values` (line 845)
- `parse_args` (line 894)
- `main` (line 903)
- `__init__` (line 74)
- `_init_weights` (line 92)
- `forward` (line 110)
- `__init__` (line 119)
- `freeze` (line 135)
- `unfreeze_mixer_only` (line 138)
- `forward` (line 143)
- `__init__` (line 152)
- `apply_masks` (line 167)
- `get_sparsity` (line 173)
- `forward` (line 178)
- `__init__` (line 202)
- `compute_metrics` (line 205)
- `get_singular_values` (line 219)
- `__init__` (line 229)
- `compute_semantic_plasticity_ratio` (line 242)
- `detect_intervention_need` (line 250)
- `update_history` (line 273)
- `perturb_mixer_targeted` (line 284)
- `__init__` (line 311)
- `create_refined_model` (line 315)
- `__init__` (line 344)
- `load_data` (line 367)
- `train_model` (line 385)
- `detect_phase_state` (line 570)
- `compute_topology_ratio` (line 580)
- `run_refinement` (line 592)
- `_save_results` (line 712)
- `_plot_results` (line 738)
- `__getitem__` (line 803)
- `evaluate` (line 814)
- `get_singular_values` (line 851)

#### `apex34.py`
**Path:** `apex34.py`

**Classs:**
- `GatedTokenMixer` (line 72) - *Efficient token mixer with high-std initialization for exploration*
- `PatchFeatureExtractor` (line 118)
- `TaxonomicMLP` (line 150) - *Sparse MLP with taxonomic heads*
- `SpectralMonitor` (line 201)
- `AdaptiveTopologyController` (line 227) - *Self-referential controller with Hysteresis*
- `IterativeRefinementEngine` (line 310)
- `IterativeRefinementTrainer` (line 343)
- `CoarseCIFAR100` (line 802)

**Functions:**
- `set_seed` (line 41) - *Ensure full reproducibility across runs*
- `compute_spectral_loss` (line 190)
- `run_hierarchy_benchmark` (line 807)
- `visualize_singular_values` (line 845)
- `main` (line 894)
- `__init__` (line 74)
- `_init_weights` (line 92)
- `forward` (line 110)
- `__init__` (line 119)
- `freeze` (line 135)
- `unfreeze_mixer_only` (line 138)
- `forward` (line 143)
- `__init__` (line 152)
- `apply_masks` (line 167)
- `get_sparsity` (line 173)
- `forward` (line 178)
- `__init__` (line 202)
- `compute_metrics` (line 205)
- `get_singular_values` (line 219)
- `__init__` (line 229)
- `compute_semantic_plasticity_ratio` (line 242)
- `detect_intervention_need` (line 250)
- `update_history` (line 273)
- `perturb_mixer_targeted` (line 284)
- `__init__` (line 311)
- `create_refined_model` (line 315)
- `__init__` (line 344)
- `load_data` (line 367)
- `train_model` (line 385)
- `detect_phase_state` (line 570)
- `compute_topology_ratio` (line 580)
- `run_refinement` (line 592)
- `_save_results` (line 712)
- `_plot_results` (line 738)
- `__getitem__` (line 803)
- `evaluate` (line 814)
- `get_singular_values` (line 851)

#### `apex35.py`
**Path:** `apex35.py`

**Classs:**
- `GatedTokenMixer` (line 66) - *Chaotic Mixer for Emergent Regime*
- `E8FusionLayer` (line 112) - *🕸️ E8 Lattice Fusion (Synergy Component E)
Optimized version from Suite v4.0.
Fuses geometric structure (Orthogonal Proj) with attention.*
- `PatchFeatureExtractor` (line 156)
- `TaxonomicMLP` (line 188)
- `BlackMirrorMonitor` (line 227) - *Passive Ontological Monitor*
- `IterativeRefinementTrainer` (line 253)
- `CoarseCIFAR100` (line 439)

**Functions:**
- `set_seed` (line 36)
- `main` (line 444)
- `__init__` (line 68)
- `_init_weights` (line 86)
- `forward` (line 104)
- `__init__` (line 118)
- `forward` (line 136)
- `__init__` (line 157)
- `freeze` (line 173)
- `unfreeze_mixer_only` (line 176)
- `forward` (line 181)
- `__init__` (line 189)
- `apply_masks` (line 204)
- `get_sparsity` (line 210)
- `forward` (line 215)
- `__init__` (line 229)
- `inspect` (line 232)
- `__init__` (line 254)
- `load_data` (line 274)
- `train_model` (line 292)
- `__getitem__` (line 440)
- `evaluate_safe` (line 483)

#### `app.py`
**Path:** `app.py`

**Classs:**
- `SpectralMonitor` (line 51)
- `PersistentPruner` (line 76)
- `SpectralMLP` (line 98)
- `EvolutionaryResonanceEngine` (line 121)

**Functions:**
- `main` (line 561)
- `__init__` (line 52)
- `compute_L` (line 55)
- `__init__` (line 77)
- `apply_to_model` (line 81)
- `enforce_during_training` (line 91)
- `__init__` (line 99)
- `reduce_input` (line 107)
- `forward` (line 113)
- `__init__` (line 122)
- `load_best_legacy_model` (line 137) - *Load the best model from previous cycle, with fallback to initial seed*
- `train_base_model_to_target` (line 175) - *Train a base model to target accuracy*
- `extract_seed_from_checkpoint` (line 228) - *Extract seed weights from checkpoint, handling different formats*
- `extract_seed_weights` (line 254)
- `inoculate_seed_adaptive` (line 260) - *Adaptive inoculation that handles dimension mismatches*
- `measure_functional_alignment` (line 288) - *Measure functional alignment via logit cosine similarity*
- `progressive_pruning_with_target` (line 309) - *Prune while maintaining target accuracy, with density constraint*
- `execute_resonance_cycle` (line 351)
- `run_evolutionary_experiment` (line 484)

#### `plank.py`
**Path:** `plank.py`

**Classs:**
- `BlackMirrorMonitor` (line 28) - *Calcula el Lagrangiano de Verdad L usando entropía de von Neumann y rango efectivo.
Umbrales calibrados empíricamente para detectar mentiras estructurales (10% ruido).*
- `SovereignNeuron` (line 62)
- `NeuroSovereign` (line 108)
- `SovereignTrainer` (line 132)

**Functions:**
- `main` (line 178)
- `__init__` (line 33)
- `inspect` (line 36)
- `__init__` (line 63)
- `forward` (line 70)
- `apply_black_swan_refraction` (line 87) - *Purificación extrema: sparsity 0.0004%*
- `__init__` (line 109)
- `forward` (line 117)
- `__init__` (line 133)
- `train_epoch` (line 139)

#### `plank10.py`
**Path:** `plank10.py`

**Classs:**
- `PatchFeatureExtractor` (line 34) - *Extrae características mediante Patch Embedding y añade una capa de mezcla (Mixer).
Esto permite al modelo aprender relaciones espaciales entre parches antes de la clasificación.*
- `LotteryMLP` (line 91)
- `StandardBaseline` (line 121) - *Baseline moderno (Patch + Mixer + MLP simple) sin evolución.*
- `SpectralMonitor` (line 133)
- `ApexEvolutionEngine` (line 148)
- `ApexTrainer` (line 247)

**Functions:**
- `main` (line 354)
- `__init__` (line 39)
- `freeze` (line 67)
- `unfreeze` (line 71)
- `forward` (line 75)
- `__init__` (line 92)
- `apply_masks` (line 104)
- `get_sparsity` (line 109)
- `forward` (line 114)
- `__init__` (line 123)
- `forward` (line 127)
- `compute_L` (line 134)
- `__init__` (line 149)
- `_gradient_nudge_inheritance` (line 154)
- `_apply_dynamic_spectral_shock` (line 188)
- `create_apex_offspring` (line 211)
- `__init__` (line 248)
- `_preprocess_batch` (line 255)
- `get_curriculum_dataset` (line 259)
- `train_model` (line 265)

#### `plank11.py`
**Path:** `plank11.py`

**Classs:**
- `TokenMixer` (line 36) - *Mezcla tokens entre sí.
Input: (B, T, D) -> Transpose -> (B, D, T) -> Linear -> (B, D, T) -> Transpose*
- `PatchFeatureExtractor` (line 57) - *ViT-Lite + True Token Mixing.*
- `LotteryMLP` (line 97)
- `StandardBaseline` (line 126)
- `SpectralMonitor` (line 137)
- `OrthogonalEvolutionEngine` (line 152)
- `OrthogonalTrainer` (line 260)

**Functions:**
- `main` (line 379)
- `__init__` (line 41)
- `forward` (line 50)
- `__init__` (line 61)
- `freeze` (line 75)
- `unfreeze` (line 79)
- `forward` (line 83)
- `__init__` (line 98)
- `apply_masks` (line 109)
- `get_sparsity` (line 114)
- `forward` (line 119)
- `__init__` (line 127)
- `forward` (line 131)
- `compute_L` (line 138)
- `__init__` (line 153)
- `_gradient_nudge_inheritance` (line 158)
- `_apply_minimalistic_shock` (line 190) - *Rank Capping: Cortamos singular values débiles y NO renormalizamos.
Esto fuerza la minimización (Energy Decay).*
- `create_orthogonal_offspring` (line 223)
- `__init__` (line 261)
- `_preprocess_batch` (line 268)
- `get_curriculum_dataset` (line 272)
- `train_model` (line 278)

#### `plank12.py`
**Path:** `plank12.py`

**Classs:**
- `TokenMixer` (line 35) - *Mezcla tokens entre sí (eje T).*
- `PatchFeatureExtractor` (line 51)
- `LotteryMLP` (line 87)
- `StandardBaseline` (line 116)
- `SpectralMonitor` (line 127)
- `OrthogonalEvolutionEngine` (line 154)
- `OrthogonalTrainer` (line 248)

**Functions:**
- `main` (line 363)
- `__init__` (line 37)
- `forward` (line 44)
- `__init__` (line 52)
- `freeze` (line 66)
- `unfreeze` (line 70)
- `forward` (line 74)
- `__init__` (line 88)
- `apply_masks` (line 99)
- `get_sparsity` (line 104)
- `forward` (line 109)
- `__init__` (line 117)
- `forward` (line 121)
- `compute_metrics` (line 128) - *Returns: (L, Rank_Efficient, S_vN)
Used for logging and decision making (NOT for backprop).*
- `__init__` (line 155)
- `_gradient_nudge_inheritance` (line 160)
- `_apply_rank_capping_shock` (line 192) - *Minimalistic Shock: Zero out weak singular values without renormalizing.
Force energy decay and minimality.*
- `create_refined_offspring` (line 224) - *Crea un hijo de las MISMAS dimensiones (Fixed Width).
Evoluciona mediante Nudge (Aprendizaje) + Shock (Poda).*
- `__init__` (line 249)
- `_preprocess_batch` (line 256)
- `get_curriculum_dataset` (line 260)
- `train_model` (line 266)

#### `plank13.py`
**Path:** `plank13.py`

**Classs:**
- `TokenMixer` (line 32) - *Mezcla tokens entre sí (Solo para Apex).*
- `PatchFeatureExtractor` (line 47) - *Extractor configurable.
use_mixer=True -> Apex (Syntactic)
use_mixer=False -> Blind Structural Baseline*
- `LotteryMLP` (line 92)
- `SpectralMonitor` (line 123)
- `OrthogonalEvolutionEngine` (line 149)
- `DualTrainer` (line 224)

**Functions:**
- `main` (line 329)
- `__init__` (line 34)
- `forward` (line 41)
- `__init__` (line 53)
- `freeze` (line 70)
- `unfreeze` (line 74)
- `forward` (line 78)
- `__init__` (line 93)
- `apply_masks` (line 104)
- `get_sparsity` (line 109)
- `forward` (line 114)
- `compute_metrics` (line 124) - *Returns: (L, Rank_Efficient, S_vN)*
- `__init__` (line 150)
- `_gradient_nudge_inheritance` (line 155)
- `_apply_rank_capping_shock` (line 187)
- `create_refined_offspring` (line 207)
- `__init__` (line 225)
- `_preprocess_batch` (line 233)
- `get_curriculum_dataset` (line 237)
- `train_single_chain` (line 243) - *Entrena una cadena específica (Apex o Blind).*

#### `plank2.py`
**Path:** `plank2.py`

**Classs:**
- `SpectralMonitor` (line 41) - *Computes L = 1 / (|S_vN - log(rank_eff + 1)| + ε)
Used purely as a diagnostic—never to modify training.*
- `PersistentPruner` (line 82) - *Applies and ENFORCES magnitude-based pruning across training.
Unlike transient pruning, this modifies the parameter mask permanently.*
- `SpectralMLP` (line 117) - *Small MLP (1504 params) for clean spectral analysis.*

**Functions:**
- `train_condition` (line 173) - *Train one condition and return full log as DataFrame.*
- `main` (line 281)
- `__init__` (line 46)
- `compute_L` (line 49) - *Returns: (L, S_vN, rank_eff, regime)*
- `__init__` (line 87)
- `apply_to_model` (line 91) - *Apply pruning mask and register backward hook to zero gradients.*
- `enforce_during_training` (line 105) - *Call this after every optimizer.step()*
- `__init__` (line 119)
- `reduce_input` (line 128) - *Reduce CIFAR-10 (32x32x3) to 32D for focus*
- `forward` (line 135)

#### `plank3.py`
**Path:** `plank3.py`

**Classs:**
- `SpectralMonitor` (line 34)
- `PersistentPruner` (line 62)
- `SpectralMLP` (line 87)

**Functions:**
- `train_dense_to_target` (line 110) - *Train dense model until it reaches target accuracy.*
- `progressive_pruning_search` (line 188) - *Progressively prune model and find critical density threshold.*
- `find_critical_threshold` (line 264) - *Find the minimum density where accuracy >= target_acc.*
- `main` (line 294)
- `__init__` (line 35)
- `compute_L` (line 38)
- `__init__` (line 63)
- `apply_to_model` (line 67)
- `enforce_during_training` (line 77)
- `__init__` (line 88)
- `reduce_input` (line 96)
- `forward` (line 102)

#### `plank4.py`
**Path:** `plank4.py`

**Classs:**
- `SpectralMonitor` (line 41)
- `PersistentPruner` (line 66)
- `SpectralMLP` (line 88)
- `FractalSovereigntyEngine` (line 111)

**Functions:**
- `main` (line 329)
- `__init__` (line 42)
- `compute_L` (line 45)
- `__init__` (line 67)
- `apply_to_model` (line 71)
- `enforce_during_training` (line 81)
- `__init__` (line 89)
- `reduce_input` (line 97)
- `forward` (line 103)
- `__init__` (line 112)
- `train_dense_model` (line 123)
- `extract_seed_weights` (line 168)
- `inoculate_seed` (line 174) - *Embed seed into larger architecture*
- `progressive_pruning` (line 192)
- `execute_cycle` (line 236)
- `run_experiment` (line 281)

#### `plank5.py`
**Path:** `plank5.py`

**Classs:**
- `SpectralMonitor` (line 30)
- `PersistentPruner` (line 50)
- `SpectralMLP` (line 65)
- `GrokkingDetector` (line 88)
- `SyntheticBlackSwanGenerator` (line 121)
- `EvolutionCycle` (line 211)
- `EvolutionaryBlackSwanChain` (line 392)

**Functions:**
- `main` (line 556)
- `__init__` (line 31)
- `compute_L` (line 34)
- `__init__` (line 51)
- `apply_to_model` (line 55)
- `__init__` (line 66)
- `reduce_input` (line 74)
- `forward` (line 80)
- `__init__` (line 89)
- `update` (line 94)
- `detect_grokking` (line 103)
- `__init__` (line 122)
- `generate` (line 128) - *Genera cisne negro sintético si no existe legacy*
- `__init__` (line 212)
- `inoculate_dna` (line 218) - *Inocula ADN del cisne anterior con mutación controlada*
- `train_with_grokking` (line 243) - *Entrena modelo induciendo grokking y monitoreando transición de fase*
- `distill_sparse_model` (line 337) - *Pruning progresivo para extraer nuevo cisne negro*
- `__init__` (line 393)
- `load_legacy_or_generate_seed` (line 402) - *Carga legacy seed o genera uno sintético*
- `run_evolutionary_chain` (line 419) - *Ejecuta la cadena evolutiva completa*
- `save_chain_results` (line 501) - *Guarda resultados completos de la cadena evolutiva*
- `print_evolution_summary` (line 519) - *Imprime resumen ejecutivo de la cadena evolutiva*

#### `plank6.py`
**Path:** `plank6.py`

**Classs:**
- `SpectralMonitor` (line 29) - *Calcula L (Coherencia Espectral) y Rank Efectivo*
- `SpectralMLP` (line 51) - *Red Neuronal Base para el experimento*
- `GrokkingDetector` (line 78)
- `GuidedElkHuntingEngine` (line 101) - *Motor que toma el mejor modelo anterior (Elk), 
muta sus pesos guiadamente y expande la arquitectura.*
- `TrainingCycle` (line 199)

**Functions:**
- `main` (line 293)
- `__init__` (line 31)
- `compute_L` (line 34)
- `__init__` (line 53)
- `reduce_input` (line 63)
- `forward` (line 70)
- `__init__` (line 79)
- `update` (line 84)
- `detect_grokking` (line 87)
- `__init__` (line 106)
- `_guided_elk_mutation` (line 110) - *Evoluciona los pesos del Elk a una dimensión mayor manteniendo coherencia.*
- `_apply_spectral_refinement` (line 146) - *Filtra componentes de baja energía y reconstruye*
- `create_offspring_from_elk` (line 158) - *Crea un nuevo modelo (Cisne Negro) basado en el Elk (mejor modelo previo).*
- `__init__` (line 200)
- `train_phase` (line 204)

#### `plank7.py`
**Path:** `plank7.py`

**Classs:**
- `SpectralMonitor` (line 29)
- `SpectralMLP` (line 45)
- `AdvancedEvolutionEngine` (line 68) - *Implementa Gradient Nudging y Espectral Shock.*
- `CurriculumTrainingCycle` (line 177)

**Functions:**
- `main` (line 283)
- `__init__` (line 30)
- `compute_L` (line 33)
- `__init__` (line 46)
- `reduce_input` (line 54)
- `forward` (line 60)
- `__init__` (line 72)
- `_gradient_nudge_inheritance` (line 76) - *Antes de entrenar, hacemos 1 paso de gradiente del Elk sobre los nuevos datos.
Esto 'pre-ajusta' el ADN al contexto actual.*
- `_apply_spectral_shock` (line 107) - *Aplica una perturbación no-lineal a los valores singulares.
Esto rompe mínimos locales planos sin destruir la estructura global.*
- `create_advanced_offspring` (line 124) - *Crea un hijo combinando:
1. Herencia de pesos
2. Gradient Nudge (context awareness)
3. Spectral Shock (ruptura de estancamiento)*
- `__init__` (line 178)
- `get_curriculum_dataset` (line 186) - *Estrategia de Curriculum:
Ciclos 1-3: Subset pequeño (Foco en estructura).
Ciclos 4+: Expansión progresiva (Foco en generalización).*
- `train_phase` (line 204)

#### `plank8.py`
**Path:** `plank8.py`

**Classs:**
- `PatchFeatureExtractor` (line 34) - *Extrae características mediante Patch Embedding.
Convierte imagen (B, 3, 32, 32) en secuencia de parches proyectados.*
- `LotteryMLP` (line 78)
- `StandardBaseline` (line 109) - *Baseline moderno (Patch + MLP simple) sin evolución.*
- `SpectralMonitor` (line 121)
- `ApexEvolutionEngine` (line 136)
- `ApexTrainer` (line 235)

**Functions:**
- `main` (line 342)
- `__init__` (line 39)
- `freeze` (line 57)
- `unfreeze` (line 61)
- `forward` (line 65)
- `__init__` (line 79)
- `apply_masks` (line 92)
- `get_sparsity` (line 97)
- `forward` (line 102)
- `__init__` (line 111)
- `forward` (line 115)
- `compute_L` (line 122)
- `__init__` (line 137)
- `_gradient_nudge_inheritance` (line 142)
- `_apply_dynamic_spectral_shock` (line 176)
- `create_apex_offspring` (line 199)
- `__init__` (line 236)
- `_preprocess_batch` (line 243)
- `get_curriculum_dataset` (line 247)
- `train_model` (line 253)

#### `resmav2_1.py`
**Path:** `resmav2_1.py`

**Classs:**
- `OptimizedE8Layer` (line 23) - *Optimized E8 with caching and efficiency improvements*
- `RESMAv2Fast` (line 48) - *Fast version: E8 + GAT fusion with minimal overhead*
- `RESMAv2Standard` (line 86) - *Standard version: 2 layers of E8 + GAT fusion*
- `RESMAv2Deep` (line 134) - *Deeper version with 3 layers*
- `GAT_Baseline` (line 182) - *Optimized GAT baseline*

**Functions:**
- `load_elliptic_data` (line 207)
- `train_and_evaluate` (line 264)
- `cross_validate_model` (line 321)
- `__init__` (line 25)
- `forward` (line 40)
- `__init__` (line 50)
- `forward` (line 73)
- `__init__` (line 88)
- `forward` (line 113)
- `__init__` (line 136)
- `forward` (line 168)
- `__init__` (line 184)
- `forward` (line 194)

### SH (1 files)

#### `install.sh`
**Path:** `install.sh`

*No symbols extracted*
