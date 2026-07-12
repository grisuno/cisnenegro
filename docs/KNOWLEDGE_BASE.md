# Polyglot Codebase Knowledge Graph

> Generated offline by **readmenator**. Supports C, C++, Python, Go, Rust, JS/TS, Java, C#, Shell, PHP, Dart, GDScript, Nim, ASM.
> No LLMs. No tokens. Pure static analysis. See more [here](https://github.com/grisuno/ReadMenator)

**Total Files Parsed:** 37 | **Total Symbols Extracted:** 1059 | **Total Imports:** 418

## Structural Knowledge Map
```mermaid
graph TD
    classDef mod fill:#1e1e1e,stroke:#ff6666,stroke-width:2px,color:#fff;
    classDef cls fill:#2d2d2d,stroke:#4ec9b0,stroke-width:2px,color:#fff;
    classDef fn fill:#333,stroke:#dcdcaa,stroke-width:1px,color:#dcdcaa;
    classDef ext fill:#111,stroke:#666,stroke-dasharray:5 5,color:#aaa;
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

**Classes:**
- `GatedTokenMixer` (line 36) `class GatedTokenMixer`
- `PatchFeatureExtractor` (line 66) `class PatchFeatureExtractor` - *Extractor configurable para CIFAR-100.
use_mixer=True -> Apex (Syntactic)
use_mixer=False -> Blind Structural Baseline*
- `LotteryMLP` (line 111) `class LotteryMLP`
- `SpectralMonitor` (line 142) `class SpectralMonitor`
- `OrthogonalEvolutionEngine` (line 158) `class OrthogonalEvolutionEngine`
- `HierarchicalTrainer` (line 231) `class HierarchicalTrainer`

**Functions:**
- `main` (line 336) `def main()`
- `__init__` (line 37) `def __init__(self, num_tokens, embed_dim)`
- `forward` (line 54) `def forward(self, x)`
- `__init__` (line 72) `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)`
- `freeze` (line 89) `def freeze(self)`
- `unfreeze` (line 93) `def unfreeze(self)`
- `forward` (line 97) `def forward(self, x)`
- `__init__` (line 112) `def __init__(self, input_dim, hidden_dim, num_classes)`
- `apply_masks` (line 123) `def apply_masks(self)`
- `get_sparsity` (line 128) `def get_sparsity(self)`
- `forward` (line 133) `def forward(self, x)`
- `compute_metrics` (line 143) `def compute_metrics(self, weight)`
- `__init__` (line 159) `def __init__(self, device)`
- `_gradient_nudge_inheritance` (line 164) `def _gradient_nudge_inheritance(self, child_model, elk_state, data_loader, feature_extractor, nudge_lr)`
- `_apply_rank_capping_shock` (line 198) `def _apply_rank_capping_shock(self, model, layer_name, keep_ratio)`
- `create_refined_offspring` (line 217) `def create_refined_offspring(self, elk_state, data_loader, feature_extractor)`
- `__init__` (line 232) `def __init__(self, device, extractor_apex, extractor_blind)`
- `_preprocess_batch` (line 240) `def _preprocess_batch(self, x, extractor)`
- `get_curriculum_dataset` (line 244) `def get_curriculum_dataset(self, cycle)`
- `train_single_chain` (line 250) `def train_single_chain(self, model, cycle, chain_type)`

#### `apex15.py`
**Path:** `apex15.py`

**Classes:**
- `GatedTokenMixer` (line 88) `class GatedTokenMixer`
- `PatchFeatureExtractor` (line 105) `class PatchFeatureExtractor`
- `TaxonomicMLP` (line 143) `class TaxonomicMLP`
- `SpectralMonitor` (line 179) `class SpectralMonitor`
- `TaxonomicTrainer` (line 195) `class TaxonomicTrainer`

**Functions:**
- `compute_spectral_loss` (line 63) `def compute_spectral_loss(W, target_rank_factor)` - *Penaliza la desalineación entre Entropía Espectral y Rango Efectivo.
Esta es la versión 'activa' de la métrica L.*
- `main` (line 367) `def main()`
- `__init__` (line 89) `def __init__(self, num_patches, embed_dim)`
- `forward` (line 98) `def forward(self, x)`
- `__init__` (line 106) `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)`
- `freeze` (line 121) `def freeze(self)`
- `unfreeze_mixer_only` (line 125) `def unfreeze_mixer_only(self)`
- `forward` (line 131) `def forward(self, x)`
- `__init__` (line 144) `def __init__(self, input_dim, hidden_dim, num_classes)`
- `apply_masks` (line 158) `def apply_masks(self)`
- `get_sparsity` (line 164) `def get_sparsity(self)`
- `forward` (line 169) `def forward(self, x)`
- `compute_metrics` (line 180) `def compute_metrics(self, weight)`
- `__init__` (line 196) `def __init__(self, device, extractor_apex, extractor_blind)`
- `_preprocess_batch` (line 203) `def _preprocess_batch(self, x, extractor)`
- `get_curriculum_dataset` (line 207) `def get_curriculum_dataset(self, cycle)`
- `train_single_chain` (line 213) `def train_single_chain(self, model, cycle, chain_type)`

#### `apex16.py`
**Path:** `apex16.py`

**Classes:**
- `GatedTokenMixer` (line 82) `class GatedTokenMixer`
- `PatchFeatureExtractor` (line 99) `class PatchFeatureExtractor`
- `TaxonomicMLP` (line 137) `class TaxonomicMLP`
- `SpectralMonitor` (line 173) `class SpectralMonitor`
- `TaxonomicTrainer` (line 189) `class TaxonomicTrainer`

**Functions:**
- `compute_spectral_loss` (line 63) `def compute_spectral_loss(W, target_rank_factor)` - *Penaliza la desalineación entre Entropía Espectral y Rango Efectivo.
v14.5: Se aplicará a pesos del MLP y del Mixer.*
- `main` (line 377) `def main()`
- `__init__` (line 83) `def __init__(self, num_patches, embed_dim)`
- `forward` (line 92) `def forward(self, x)`
- `__init__` (line 100) `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)`
- `freeze` (line 115) `def freeze(self)`
- `unfreeze_mixer_only` (line 119) `def unfreeze_mixer_only(self)`
- `forward` (line 125) `def forward(self, x)`
- `__init__` (line 138) `def __init__(self, input_dim, hidden_dim, num_classes)`
- `apply_masks` (line 152) `def apply_masks(self)`
- `get_sparsity` (line 158) `def get_sparsity(self)`
- `forward` (line 163) `def forward(self, x)`
- `compute_metrics` (line 174) `def compute_metrics(self, weight)`
- `__init__` (line 190) `def __init__(self, device, extractor_apex, extractor_blind)`
- `_preprocess_batch` (line 197) `def _preprocess_batch(self, x, extractor)`
- `get_curriculum_dataset` (line 201) `def get_curriculum_dataset(self, cycle)`
- `train_single_chain` (line 207) `def train_single_chain(self, model, cycle, chain_type)`

#### `apex17.py`
**Path:** `apex17.py`

**Classes:**
- `GatedTokenMixer` (line 81) `class GatedTokenMixer`
- `PatchFeatureExtractor` (line 98) `class PatchFeatureExtractor`
- `TaxonomicMLP` (line 136) `class TaxonomicMLP`
- `SpectralMonitor` (line 172) `class SpectralMonitor`
- `TaxonomicTrainer` (line 190) `class TaxonomicTrainer`
- `CoarseCIFAR100` (line 368) `class CoarseCIFAR100` - *Wrapper que convierte CIFAR100 en un problema de clasificación pura de 20 clases (Superclases).
Se usa para validar el inductive bias aprendido.*

**Functions:**
- `compute_spectral_loss` (line 65) `def compute_spectral_loss(W)` - *v15.0: Optimization Objective for Spectral Control.*
- `run_hierarchy_benchmark` (line 378) `def run_hierarchy_benchmark(model_apex, model_blind, device)`
- `main` (line 428) `def main()`
- `__init__` (line 82) `def __init__(self, num_patches, embed_dim)`
- `forward` (line 91) `def forward(self, x)`
- `__init__` (line 99) `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)`
- `freeze` (line 114) `def freeze(self)`
- `unfreeze_mixer_only` (line 118) `def unfreeze_mixer_only(self)`
- `forward` (line 124) `def forward(self, x)`
- `__init__` (line 137) `def __init__(self, input_dim, hidden_dim, num_classes)`
- `apply_masks` (line 151) `def apply_masks(self)`
- `get_sparsity` (line 157) `def get_sparsity(self)`
- `forward` (line 162) `def forward(self, x)`
- `compute_metrics` (line 173) `def compute_metrics(self, weight)` - *L_mon: Used for plotting and historical reporting, not optimization.*
- `__init__` (line 191) `def __init__(self, device, extractor_apex, extractor_blind)`
- `_preprocess_batch` (line 198) `def _preprocess_batch(self, x, extractor)`
- `get_curriculum_dataset` (line 202) `def get_curriculum_dataset(self, cycle)`
- `train_single_chain` (line 208) `def train_single_chain(self, model, cycle, chain_type)`
- `__getitem__` (line 373) `def __getitem__(self, index)`
- `evaluate` (line 391) `def evaluate(model, extractor, name)`

#### `apex18.py`
**Path:** `apex18.py`

**Classes:**
- `GatedTokenMixer` (line 81) `class GatedTokenMixer`
- `PatchFeatureExtractor` (line 98) `class PatchFeatureExtractor`
- `TaxonomicMLP` (line 136) `class TaxonomicMLP`
- `SpectralMonitor` (line 172) `class SpectralMonitor`
- `TaxonomicTrainer` (line 188) `class TaxonomicTrainer`
- `CoarseCIFAR100` (line 369) `class CoarseCIFAR100`

**Functions:**
- `compute_spectral_loss` (line 65) `def compute_spectral_loss(W)` - *v15.1: Optimization Objective for Spectral Control (Applied to both APEX and BLIND).*
- `run_hierarchy_benchmark` (line 374) `def run_hierarchy_benchmark(model_apex, model_blind, device)`
- `main` (line 420) `def main()`
- `__init__` (line 82) `def __init__(self, num_patches, embed_dim)`
- `forward` (line 91) `def forward(self, x)`
- `__init__` (line 99) `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)`
- `freeze` (line 114) `def freeze(self)`
- `unfreeze_mixer_only` (line 118) `def unfreeze_mixer_only(self)`
- `forward` (line 124) `def forward(self, x)`
- `__init__` (line 137) `def __init__(self, input_dim, hidden_dim, num_classes)`
- `apply_masks` (line 151) `def apply_masks(self)`
- `get_sparsity` (line 157) `def get_sparsity(self)`
- `forward` (line 162) `def forward(self, x)`
- `compute_metrics` (line 173) `def compute_metrics(self, weight)`
- `__init__` (line 189) `def __init__(self, device, extractor_apex, extractor_blind)`
- `_preprocess_batch` (line 196) `def _preprocess_batch(self, x, extractor)`
- `get_curriculum_dataset` (line 200) `def get_curriculum_dataset(self, cycle)`
- `train_single_chain` (line 206) `def train_single_chain(self, model, cycle, chain_type)`
- `__getitem__` (line 370) `def __getitem__(self, index)`
- `evaluate` (line 386) `def evaluate(model, extractor, name)`

#### `apex19.py`
**Path:** `apex19.py`

**Classes:**
- `GatedTokenMixer` (line 84) `class GatedTokenMixer`
- `PatchFeatureExtractor` (line 101) `class PatchFeatureExtractor`
- `TaxonomicMLP` (line 139) `class TaxonomicMLP`
- `SpectralMonitor` (line 175) `class SpectralMonitor`
- `TaxonomicTrainer` (line 211) `class TaxonomicTrainer`
- `CoarseCIFAR100` (line 394) `class CoarseCIFAR100`

**Functions:**
- `compute_spectral_loss` (line 68) `def compute_spectral_loss(W)` - *L_opt: Optimization Objective for Structural Control.*
- `run_hierarchy_benchmark` (line 399) `def run_hierarchy_benchmark(model_apex, model_blind, device)`
- `main` (line 445) `def main()`
- `__init__` (line 85) `def __init__(self, num_patches, embed_dim)`
- `forward` (line 94) `def forward(self, x)`
- `__init__` (line 102) `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)`
- `freeze` (line 117) `def freeze(self)`
- `unfreeze_mixer_only` (line 121) `def unfreeze_mixer_only(self)`
- `forward` (line 127) `def forward(self, x)`
- `__init__` (line 140) `def __init__(self, input_dim, hidden_dim, num_classes)`
- `apply_masks` (line 154) `def apply_masks(self)`
- `get_sparsity` (line 160) `def get_sparsity(self)`
- `forward` (line 165) `def forward(self, x)`
- `compute_metrics` (line 176) `def compute_metrics(self, weight)`
- `compute_topology_ratio` (line 188) `def compute_topology_ratio(self, model, extractor, chain_type)` - *v15.2: Calcula el ratio R = L_opt / L_mon.
Valores bajos indican alineación estable.
Valores altos o erráticos indican transición de fase (Grokking/Collapse).*
- `__init__` (line 212) `def __init__(self, device, extractor_apex, extractor_blind)`
- `_preprocess_batch` (line 219) `def _preprocess_batch(self, x, extractor)`
- `get_curriculum_dataset` (line 223) `def get_curriculum_dataset(self, cycle)`
- `train_single_chain` (line 229) `def train_single_chain(self, model, cycle, chain_type)`
- `__getitem__` (line 395) `def __getitem__(self, index)`
- `evaluate` (line 411) `def evaluate(model, extractor, name)`

#### `apex20.py`
**Path:** `apex20.py`

**Classes:**
- `GatedTokenMixer` (line 85) `class GatedTokenMixer`
- `PatchFeatureExtractor` (line 102) `class PatchFeatureExtractor`
- `TaxonomicMLP` (line 140) `class TaxonomicMLP`
- `SpectralMonitor` (line 176) `class SpectralMonitor`
- `TaxonomicTrainer` (line 242) `class TaxonomicTrainer`
- `CoarseCIFAR100` (line 426) `class CoarseCIFAR100`

**Functions:**
- `compute_spectral_loss` (line 69) `def compute_spectral_loss(W)` - *L_opt: Optimization Objective for Structural Control.*
- `run_hierarchy_benchmark` (line 431) `def run_hierarchy_benchmark(model_apex, model_blind, device)`
- `main` (line 490) `def main()`
- `__init__` (line 86) `def __init__(self, num_patches, embed_dim)`
- `forward` (line 95) `def forward(self, x)`
- `__init__` (line 103) `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)`
- `freeze` (line 118) `def freeze(self)`
- `unfreeze_mixer_only` (line 122) `def unfreeze_mixer_only(self)`
- `forward` (line 128) `def forward(self, x)`
- `__init__` (line 141) `def __init__(self, input_dim, hidden_dim, num_classes)`
- `apply_masks` (line 155) `def apply_masks(self)`
- `get_sparsity` (line 161) `def get_sparsity(self)`
- `forward` (line 166) `def forward(self, x)`
- `compute_metrics` (line 177) `def compute_metrics(self, weight)`
- `detect_phase_state` (line 190) `def detect_phase_state(self, ratio_history)` - *v15.3: Detects phase state based on relative deviation, not absolute value.
Returns: 'STABLE', 'SHIFTING', or 'INIT'*
- `compute_topology_ratio` (line 216) `def compute_topology_ratio(self, model, extractor, chain_type)` - *v15.3: Returns L_opt components and Total Ratio.
Ratio = L_opt_Total / L_mon(Downstream)*
- `__init__` (line 243) `def __init__(self, device, extractor_apex, extractor_blind)`
- `_preprocess_batch` (line 250) `def _preprocess_batch(self, x, extractor)`
- `get_curriculum_dataset` (line 254) `def get_curriculum_dataset(self, cycle)`
- `train_single_chain` (line 260) `def train_single_chain(self, model, cycle, chain_type)`
- `__getitem__` (line 427) `def __getitem__(self, index)`
- `evaluate` (line 443) `def evaluate(model, extractor, name)`

#### `apex21.py`
**Path:** `apex21.py`

**Classes:**
- `GatedTokenMixer` (line 88) `class GatedTokenMixer`
- `PatchFeatureExtractor` (line 105) `class PatchFeatureExtractor`
- `TaxonomicMLP` (line 143) `class TaxonomicMLP`
- `SpectralMonitor` (line 179) `class SpectralMonitor`
- `TopologyController` (line 228) `class TopologyController` - *v15.4: Manages Active Interventions to break stagnation.*
- `TaxonomicTrainer` (line 270) `class TaxonomicTrainer`
- `CoarseCIFAR100` (line 459) `class CoarseCIFAR100`

**Functions:**
- `compute_spectral_loss` (line 73) `def compute_spectral_loss(W)`
- `run_hierarchy_benchmark` (line 464) `def run_hierarchy_benchmark(model_apex, model_blind, device)`
- `main` (line 513) `def main()`
- `__init__` (line 89) `def __init__(self, num_patches, embed_dim)`
- `forward` (line 98) `def forward(self, x)`
- `__init__` (line 106) `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)`
- `freeze` (line 121) `def freeze(self)`
- `unfreeze_mixer_only` (line 125) `def unfreeze_mixer_only(self)`
- `forward` (line 131) `def forward(self, x)`
- `__init__` (line 144) `def __init__(self, input_dim, hidden_dim, num_classes)`
- `apply_masks` (line 158) `def apply_masks(self)`
- `get_sparsity` (line 164) `def get_sparsity(self)`
- `forward` (line 169) `def forward(self, x)`
- `compute_metrics` (line 180) `def compute_metrics(self, weight)`
- `detect_phase_state` (line 192) `def detect_phase_state(self, ratio_history)`
- `compute_topology_ratio` (line 204) `def compute_topology_ratio(self, model, extractor, chain_type)` - *v15.3: Returns L_opt components and Total Ratio.
Ratio = L_opt_Total / L_mon(Downstream)*
- `__init__` (line 230) `def __init__(self)`
- `check_intervention` (line 233) `def check_intervention(self, phase_state, coarse_acc, extractor)` - *Decides whether to intervene.
Returns 'INTERVENE' if action is taken, 'NONE' otherwise.*
- `perturb_mixer` (line 255) `def perturb_mixer(self, extractor)` - *Causal Intervention: Inject topological noise to force phase shift.*
- `__init__` (line 271) `def __init__(self, device, extractor_apex, extractor_blind)`
- `_preprocess_batch` (line 279) `def _preprocess_batch(self, x, extractor)`
- `get_curriculum_dataset` (line 283) `def get_curriculum_dataset(self, cycle)`
- `train_single_chain` (line 289) `def train_single_chain(self, model, cycle, chain_type)`
- `__getitem__` (line 460) `def __getitem__(self, index)`
- `evaluate` (line 476) `def evaluate(model, extractor, name)`

#### `apex22.py`
**Path:** `apex22.py`

**Classes:**
- `GatedTokenMixer` (line 88) `class GatedTokenMixer`
- `PatchFeatureExtractor` (line 105) `class PatchFeatureExtractor`
- `TaxonomicMLP` (line 143) `class TaxonomicMLP`
- `SpectralMonitor` (line 179) `class SpectralMonitor`
- `TopologyController` (line 204) `class TopologyController` - *v15.5: Manages Targeted Spectral Interventions (Surgery).*
- `TaxonomicTrainer` (line 273) `class TaxonomicTrainer`
- `CoarseCIFAR100` (line 459) `class CoarseCIFAR100`

**Functions:**
- `compute_spectral_loss` (line 73) `def compute_spectral_loss(W)`
- `run_hierarchy_benchmark` (line 464) `def run_hierarchy_benchmark(model_apex, model_blind, device)`
- `main` (line 513) `def main()`
- `__init__` (line 89) `def __init__(self, num_patches, embed_dim)`
- `forward` (line 98) `def forward(self, x)`
- `__init__` (line 106) `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)`
- `freeze` (line 121) `def freeze(self)`
- `unfreeze_mixer_only` (line 125) `def unfreeze_mixer_only(self)`
- `forward` (line 131) `def forward(self, x)`
- `__init__` (line 144) `def __init__(self, input_dim, hidden_dim, num_classes)`
- `apply_masks` (line 158) `def apply_masks(self)`
- `get_sparsity` (line 164) `def get_sparsity(self)`
- `forward` (line 169) `def forward(self, x)`
- `compute_metrics` (line 180) `def compute_metrics(self, weight)`
- `detect_phase_state` (line 192) `def detect_phase_state(self, ratio_history)`
- `__init__` (line 206) `def __init__(self)`
- `check_intervention` (line 209) `def check_intervention(self, phase_state, coarse_acc, extractor)`
- `perturb_mixer_targeted` (line 226) `def perturb_mixer_targeted(self, extractor)` - *v15.5: Targeted Phase Surgery.
Injects noise ONLY in the nullspace of the dominant spectral subspace.
Preserves existing structure while forcing exploration of latent dimensions.*
- `__init__` (line 274) `def __init__(self, device, extractor_apex, extractor_blind)`
- `_preprocess_batch` (line 282) `def _preprocess_batch(self, x, extractor)`
- `get_curriculum_dataset` (line 286) `def get_curriculum_dataset(self, cycle)`
- `train_single_chain` (line 292) `def train_single_chain(self, model, cycle, chain_type)`
- `__getitem__` (line 460) `def __getitem__(self, index)`
- `evaluate` (line 476) `def evaluate(model, extractor, name)`

#### `apex23.py`
**Path:** `apex23.py`

**Classes:**
- `GatedTokenMixer` (line 91) `class GatedTokenMixer`
- `PatchFeatureExtractor` (line 108) `class PatchFeatureExtractor`
- `TaxonomicMLP` (line 146) `class TaxonomicMLP`
- `SpectralMonitor` (line 182) `class SpectralMonitor`
- `TopologyController` (line 228) `class TopologyController`
- `TaxonomicTrainer` (line 306) `class TaxonomicTrainer`
- `CoarseCIFAR100` (line 493) `class CoarseCIFAR100`

**Functions:**
- `compute_spectral_loss` (line 74) `def compute_spectral_loss(W)` - *L_opt: Computes the discrepancy between spectral entropy and effective rank.*
- `run_hierarchy_benchmark` (line 498) `def run_hierarchy_benchmark(model_apex, model_blind, device)`
- `main` (line 544) `def main()`
- `__init__` (line 92) `def __init__(self, num_patches, embed_dim)`
- `forward` (line 101) `def forward(self, x)`
- `__init__` (line 109) `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)`
- `freeze` (line 124) `def freeze(self)`
- `unfreeze_mixer_only` (line 128) `def unfreeze_mixer_only(self)`
- `forward` (line 134) `def forward(self, x)`
- `__init__` (line 147) `def __init__(self, input_dim, hidden_dim, num_classes)`
- `apply_masks` (line 161) `def apply_masks(self)`
- `get_sparsity` (line 167) `def get_sparsity(self)`
- `forward` (line 172) `def forward(self, x)`
- `compute_metrics` (line 183) `def compute_metrics(self, weight)` - *L_mon: Legacy reporting metric.*
- `detect_phase_state` (line 196) `def detect_phase_state(self, ratio_history)`
- `compute_topology_ratio` (line 208) `def compute_topology_ratio(self, model, extractor, chain_type)` - *Calculates Topo_R = L_opt / L_mon.
L_opt is the active optimization energy (FC1 + Mixer).
L_mon is the passive structural metric.*
- `__init__` (line 229) `def __init__(self)`
- `check_intervention` (line 232) `def check_intervention(self, phase_state, coarse_acc, extractor)` - *Decides if intervention is needed based on Phase and Performance.*
- `perturb_mixer_targeted` (line 250) `def perturb_mixer_targeted(self, extractor)` - *v15.5: Targeted Spectral Surgery.
Injects noise in the nullspace of the dominant spectral subspace to explore
latent modes without destroying learned hierarchy.*
- `__init__` (line 307) `def __init__(self, device, extractor_apex, extractor_blind)`
- `_preprocess_batch` (line 315) `def _preprocess_batch(self, x, extractor)`
- `get_curriculum_dataset` (line 319) `def get_curriculum_dataset(self, cycle)`
- `train_single_chain` (line 325) `def train_single_chain(self, model, cycle, chain_type)`
- `__getitem__` (line 494) `def __getitem__(self, index)`
- `evaluate` (line 510) `def evaluate(model, extractor, name)`

#### `apex24.py`
**Path:** `apex24.py`

**Classes:**
- `GatedTokenMixer` (line 92) `class GatedTokenMixer`
- `PatchFeatureExtractor` (line 109) `class PatchFeatureExtractor`
- `TaxonomicMLP` (line 147) `class TaxonomicMLP`
- `SpectralMonitor` (line 183) `class SpectralMonitor`
- `TopologyController` (line 229) `class TopologyController`
- `TaxonomicTrainer` (line 339) `class TaxonomicTrainer`
- `CoarseCIFAR100` (line 527) `class CoarseCIFAR100`

**Functions:**
- `compute_spectral_loss` (line 75) `def compute_spectral_loss(W)` - *L_opt: Computes the discrepancy between spectral entropy and effective rank.*
- `run_hierarchy_benchmark` (line 532) `def run_hierarchy_benchmark(model_apex, model_blind, device)`
- `main` (line 578) `def main()`
- `__init__` (line 93) `def __init__(self, num_patches, embed_dim)`
- `forward` (line 102) `def forward(self, x)`
- `__init__` (line 110) `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)`
- `freeze` (line 125) `def freeze(self)`
- `unfreeze_mixer_only` (line 129) `def unfreeze_mixer_only(self)`
- `forward` (line 135) `def forward(self, x)`
- `__init__` (line 148) `def __init__(self, input_dim, hidden_dim, num_classes)`
- `apply_masks` (line 162) `def apply_masks(self)`
- `get_sparsity` (line 168) `def get_sparsity(self)`
- `forward` (line 173) `def forward(self, x)`
- `compute_metrics` (line 184) `def compute_metrics(self, weight)` - *L_mon: Legacy reporting metric.*
- `detect_phase_state` (line 197) `def detect_phase_state(self, ratio_history)`
- `compute_topology_ratio` (line 209) `def compute_topology_ratio(self, model, extractor, chain_type)` - *Calculates Topo_R = L_opt / L_mon.
L_opt is the active optimization energy (FC1 + Mixer).
L_mon is the passive structural metric.*
- `__init__` (line 230) `def __init__(self)`
- `check_intervention` (line 235) `def check_intervention(self, phase_state, coarse_acc, extractor, current_topo_r)` - *v15.5 Final: Geometric Mismatch Detection.
Triggers intervention if Topo_R (Structure) changes but Coarse Acc (Semantics) does not.*
- `perturb_mixer_targeted` (line 284) `def perturb_mixer_targeted(self, extractor)` - *v15.5: Targeted Spectral Surgery.
Injects noise in the nullspace of the dominant spectral subspace to explore
latent modes without destroying learned hierarchy.*
- `__init__` (line 340) `def __init__(self, device, extractor_apex, extractor_blind)`
- `_preprocess_batch` (line 348) `def _preprocess_batch(self, x, extractor)`
- `get_curriculum_dataset` (line 352) `def get_curriculum_dataset(self, cycle)`
- `train_single_chain` (line 358) `def train_single_chain(self, model, cycle, chain_type)`
- `__getitem__` (line 528) `def __getitem__(self, index)`
- `evaluate` (line 544) `def evaluate(model, extractor, name)`

#### `apex25.py`
**Path:** `apex25.py`

**Classes:**
- `GatedTokenMixer` (line 89) `class GatedTokenMixer`
- `PatchFeatureExtractor` (line 106) `class PatchFeatureExtractor`
- `TaxonomicMLP` (line 144) `class TaxonomicMLP`
- `SpectralMonitor` (line 180) `class SpectralMonitor`
- `TopologyController` (line 224) `class TopologyController`
- `TaxonomicTrainer` (line 316) `class TaxonomicTrainer`
- `CoarseCIFAR100` (line 504) `class CoarseCIFAR100`

**Functions:**
- `compute_spectral_loss` (line 75) `def compute_spectral_loss(W)` - *L_opt: Computes the discrepancy between spectral entropy and effective rank.*
- `run_hierarchy_benchmark` (line 509) `def run_hierarchy_benchmark(model_apex, model_blind, device)`
- `main` (line 555) `def main()`
- `__init__` (line 90) `def __init__(self, num_patches, embed_dim)`
- `forward` (line 99) `def forward(self, x)`
- `__init__` (line 107) `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)`
- `freeze` (line 122) `def freeze(self)`
- `unfreeze_mixer_only` (line 126) `def unfreeze_mixer_only(self)`
- `forward` (line 132) `def forward(self, x)`
- `__init__` (line 145) `def __init__(self, input_dim, hidden_dim, num_classes)`
- `apply_masks` (line 159) `def apply_masks(self)`
- `get_sparsity` (line 165) `def get_sparsity(self)`
- `forward` (line 170) `def forward(self, x)`
- `compute_metrics` (line 181) `def compute_metrics(self, weight)` - *L_mon: Legacy reporting metric.*
- `detect_phase_state` (line 194) `def detect_phase_state(self, ratio_history)`
- `compute_topology_ratio` (line 206) `def compute_topology_ratio(self, model, extractor, chain_type)` - *Calculates Topo_R = L_opt / L_mon.*
- `__init__` (line 225) `def __init__(self)`
- `check_intervention` (line 230) `def check_intervention(self, phase_state, coarse_acc, extractor, current_topo_r)` - *v15.5 Final: Geometric Mismatch Detection.
Triggers intervention if Topo_R (Structure) changes but Coarse Acc (Semantics) does not.*
- `perturb_mixer_targeted` (line 277) `def perturb_mixer_targeted(self, extractor)` - *v15.5: Targeted Spectral Surgery.
Injects noise in the nullspace of the dominant spectral subspace.*
- `__init__` (line 317) `def __init__(self, device, extractor_apex, extractor_blind)`
- `_preprocess_batch` (line 325) `def _preprocess_batch(self, x, extractor)`
- `get_curriculum_dataset` (line 329) `def get_curriculum_dataset(self, cycle)`
- `train_single_chain` (line 335) `def train_single_chain(self, model, cycle, chain_type)`
- `__getitem__` (line 505) `def __getitem__(self, index)`
- `evaluate` (line 521) `def evaluate(model, extractor, name)`

#### `apex26.py`
**Path:** `apex26.py`

**Classes:**
- `GatedTokenMixer` (line 89) `class GatedTokenMixer` - *Efficient token mixer with gating mechanism for feature interaction*
- `PatchFeatureExtractor` (line 146) `class PatchFeatureExtractor` - *Efficient patch-based feature extractor with optional token mixing*
- `TaxonomicMLP` (line 207) `class TaxonomicMLP` - *Sparse MLP with taxonomic heads for hierarchical learning*
- `SpectralMonitor` (line 281) `class SpectralMonitor` - *Monitor spectral properties of weight matrices for evolutionary guidance*
- `TopologyController` (line 302) `class TopologyController` - *Advanced controller for targeted spectral surgery*
- `EvolutionaryEngine` (line 410) `class EvolutionaryEngine` - *Engine for evolving neural networks through spectral refinement*
- `CoarseCIFAR100` (line 480) `class CoarseCIFAR100` - *Wrapper that converts CIFAR100 into a pure 20-class classification problem (Superclasses).
Used to validate learned inductive bias.*
- `EvolutionaryTrainer` (line 544) `class EvolutionaryTrainer` - *Framework for evolutionary training with statistical validation*

**Functions:**
- `set_seed` (line 43) `def set_seed(seed)` - *Ensure full reproducibility across runs*
- `compute_spectral_loss` (line 265) `def compute_spectral_loss(W)` - *Optimization Objective for Spectral Control (L_opt)*
- `run_hierarchy_benchmark` (line 490) `def run_hierarchy_benchmark(model_apex, model_blind, feature_extractor_apex, feature_extractor_blind, device)` - *Run hierarchy stress test to validate inductive bias transfer*
- `parse_args` (line 1170) `def parse_args()`
- `main` (line 1181) `def main()` - *Main execution function*
- `__init__` (line 91) `def __init__(self, num_patches, embed_dim)`
- `_init_weights` (line 114) `def _init_weights(self)` - *Initialize weights for stable training*
- `forward` (line 128) `def forward(self, x)` - *Input:  [B, num_patches, embed_dim]
Output: [B, num_patches, embed_dim]*
- `__init__` (line 148) `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)`
- `freeze` (line 177) `def freeze(self)` - *Freeze all parameters for transfer learning*
- `unfreeze_mixer_only` (line 183) `def unfreeze_mixer_only(self)` - *Unfreeze only the mixer parameters for fine-tuning*
- `forward` (line 190) `def forward(self, x)` - *Input:  [B, C, H, W]
Output: [B, embed_dim]*
- `__init__` (line 209) `def __init__(self, input_dim, hidden_dim, num_classes, num_superclasses)`
- `apply_masks` (line 232) `def apply_masks(self)` - *Apply sparsity masks to weights*
- `get_sparsity` (line 239) `def get_sparsity(self)` - *Calculate overall sparsity percentage*
- `forward` (line 245) `def forward(self, x)` - *Input:  [B, input_dim]
Output: ([B, num_classes], [B, num_superclasses])*
- `__init__` (line 283) `def __init__(self, epsilon)`
- `compute_metrics` (line 286) `def compute_metrics(self, weight)` - *Compute spectral coherence metrics*
- `__init__` (line 304) `def __init__(self, target_coarse_v, stagnation_limit, mixer_noise_scale, dominant_energy_threshold)`
- `detect_phase_state` (line 314) `def detect_phase_state(self, ratio_history, phase_window, phase_std_dev_limit)` - *Detect phase state based on topology ratio history*
- `check_intervention` (line 332) `def check_intervention(self, phase_state, coarse_acc, extractor, current_topo_r, geo_window)` - *Check if intervention is needed based on geometric mismatch detection*
- `perturb_mixer_targeted` (line 377) `def perturb_mixer_targeted(self, extractor)` - *Targeted Spectral Surgery: Inject noise in the nullspace of dominant subspace*
- `__init__` (line 412) `def __init__(self, device, target_L)`
- `apply_rank_capping` (line 417) `def apply_rank_capping(self, model, layer_name, keep_ratio)` - *Apply rank capping shock to prevent over-specialization*
- `create_offspring` (line 434) `def create_offspring(self, parent_state, data_loader, feature_extractor, lambda_taxonomic, learning_rate)` - *Create refined offspring through gradient-based inheritance*
- `__getitem__` (line 485) `def __getitem__(self, index)`
- `evaluate` (line 503) `def evaluate(model, extractor, name)`
- `__init__` (line 546) `def __init__(self, device, output_dir)`
- `load_data` (line 572) `def load_data(self, cycle, batch_size)` - *Load curriculum dataset based on evolutionary cycle*
- `train_model` (line 614) `def train_model(self, model, cycle, chain_type, feature_extractor)` - *Train model with evolutionary pressure and hierarchical learning*
- `compute_topology_ratio` (line 837) `def compute_topology_ratio(self, model, extractor, chain_type)` - *Calculates Topo_R = L_opt / L_mon.*
- `run_evolution` (line 855) `def run_evolution(self, num_iterations, num_seeds, early_stop_patience)` - *Run full evolutionary experiment with statistical validation*
- `_save_results` (line 1027) `def _save_results(self, all_results, best_overall, hierarchy_delta)` - *Save results to files*
- `_plot_results` (line 1081) `def _plot_results(self, all_results)` - *Create publication-quality plots*

#### `apex27.py`
**Path:** `apex27.py`

**Classes:**
- `GatedTokenMixer` (line 83) `class GatedTokenMixer` - *Efficient token mixer with gating mechanism*
- `PatchFeatureExtractor` (line 124) `class PatchFeatureExtractor` - *Efficient patch-based feature extractor*
- `TaxonomicMLP` (line 170) `class TaxonomicMLP` - *Sparse MLP with taxonomic heads*
- `DynamicThresholdController` (line 233) `class DynamicThresholdController` - *v18 Improvement: Replaces fixed TARGET_COARSE_V with adaptive logic.
Triggers intervention if current performance stagnates relative to its own history.*
- `SpectralMonitor` (line 255) `class SpectralMonitor` - *Monitor spectral properties*
- `TopologyController` (line 275) `class TopologyController` - *v18 Improvement: Advanced controller with Adaptive Thresholding.
Implements Targeted Spectral Surgery with Nullspace Injection.*
- `IterativeRefinementEngine` (line 369) `class IterativeRefinementEngine` - *v18: Engine for iterative refinement (formerly Evolutionary)*
- `IterativeTrainer` (line 416) `class IterativeTrainer` - *Framework for Iterative Refinement with v18 Adaptive Control*

**Functions:**
- `set_seed` (line 38) `def set_seed(seed)` - *Ensure full reproducibility across runs*
- `compute_spectral_loss` (line 217) `def compute_spectral_loss(W)` - *Optimization Objective for Spectral Control (L_opt)*
- `parse_args` (line 900) `def parse_args()`
- `main` (line 914) `def main()`
- `__init__` (line 85) `def __init__(self, num_patches, embed_dim)`
- `_init_weights` (line 104) `def _init_weights(self)`
- `forward` (line 116) `def forward(self, x)`
- `__init__` (line 126) `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)`
- `freeze` (line 151) `def freeze(self)`
- `unfreeze_mixer_only` (line 156) `def unfreeze_mixer_only(self)`
- `forward` (line 162) `def forward(self, x)`
- `__init__` (line 172) `def __init__(self, input_dim, hidden_dim, num_classes, num_superclasses)`
- `apply_masks` (line 192) `def apply_masks(self)`
- `get_sparsity` (line 198) `def get_sparsity(self)`
- `forward` (line 203) `def forward(self, x)`
- `__init__` (line 238) `def __init__(self, window_size, percentile_trigger)`
- `update` (line 243) `def update(self, value)`
- `is_stagnant` (line 246) `def is_stagnant(self, current_val)`
- `__init__` (line 257) `def __init__(self, epsilon)`
- `compute_metrics` (line 260) `def compute_metrics(self, weight)`
- `__init__` (line 280) `def __init__(self, dynamic_threshold, stagnation_limit, mixer_noise_scale, dominant_energy_threshold, enable_surgery)`
- `check_intervention` (line 291) `def check_intervention(self, coarse_acc, extractor, current_topo_r, geo_window, alpha)`
- `perturb_mixer_targeted` (line 336) `def perturb_mixer_targeted(self, extractor)` - *Targeted Spectral Surgery: Inject noise in the nullspace of dominant subspace*
- `__init__` (line 371) `def __init__(self, device)`
- `create_offspring` (line 374) `def create_offspring(self, parent_state, data_loader, feature_extractor, lambda_taxonomic, learning_rate)` - *Create refined offspring through gradient-based inheritance*
- `__init__` (line 418) `def __init__(self, device, output_dir, enable_surgery, enable_taxonomy)`
- `load_data` (line 441) `def load_data(self, cycle, batch_size)`
- `train_model` (line 479) `def train_model(self, model, cycle, chain_type, feature_extractor)`
- `compute_topology_ratio` (line 683) `def compute_topology_ratio(self, model, extractor, chain_type)`
- `run_refinement` (line 696) `def run_refinement(self, num_iterations, num_seeds, early_stop_patience)`
- `_save_results` (line 809) `def _save_results(self, all_results)`
- `_plot_results_v18` (line 815) `def _plot_results_v18(self, all_results)`

#### `apex28.py`
**Path:** `apex28.py`

**Classes:**
- `GatedTokenMixer` (line 93) `class GatedTokenMixer` - *Efficient token mixer with gating mechanism for feature interaction*
- `PatchFeatureExtractor` (line 150) `class PatchFeatureExtractor` - *Efficient patch-based feature extractor with optional token mixing*
- `TaxonomicMLP` (line 211) `class TaxonomicMLP` - *Sparse MLP with taxonomic heads for hierarchical learning*
- `SpectralMonitor` (line 285) `class SpectralMonitor` - *Monitor spectral properties with adaptive analysis*
- `AdaptiveTopologyController` (line 316) `class AdaptiveTopologyController` - *Self-referential controller using semantic-plasticity ratio*
- `IterativeRefinementEngine` (line 415) `class IterativeRefinementEngine` - *Engine for iterative refinement through spectral control*
- `CoarseCIFAR100` (line 484) `class CoarseCIFAR100` - *Wrapper that converts CIFAR100 into a pure 20-class classification problem (Superclasses).
Used to validate learned inductive bias.*
- `IterativeRefinementTrainer` (line 611) `class IterativeRefinementTrainer` - *Framework for iterative refinement with statistical validation*

**Functions:**
- `set_seed` (line 47) `def set_seed(seed)` - *Ensure full reproducibility across runs*
- `compute_spectral_loss` (line 269) `def compute_spectral_loss(W)` - *Optimization Objective for Spectral Control (L_opt)*
- `run_hierarchy_benchmark` (line 494) `def run_hierarchy_benchmark(model_apex, model_blind, feature_extractor_apex, feature_extractor_blind, device)` - *Run hierarchy stress test to validate inductive bias transfer*
- `visualize_singular_values` (line 548) `def visualize_singular_values(model_apex, model_blind, feature_extractor_apex, feature_extractor_blind, device, output_dir)` - *Generate publication-quality singular value visualizations*
- `parse_args` (line 1270) `def parse_args()`
- `main` (line 1282) `def main()` - *Main execution function*
- `__init__` (line 95) `def __init__(self, num_patches, embed_dim)`
- `_init_weights` (line 118) `def _init_weights(self)` - *Initialize weights for stable training*
- `forward` (line 132) `def forward(self, x)` - *Input:  [B, num_patches, embed_dim]
Output: [B, num_patches, embed_dim]*
- `__init__` (line 152) `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)`
- `freeze` (line 181) `def freeze(self)` - *Freeze all parameters for transfer learning*
- `unfreeze_mixer_only` (line 187) `def unfreeze_mixer_only(self)` - *Unfreeze only the mixer parameters for fine-tuning*
- `forward` (line 194) `def forward(self, x)` - *Input:  [B, C, H, W]
Output: [B, embed_dim]*
- `__init__` (line 213) `def __init__(self, input_dim, hidden_dim, num_classes, num_superclasses)`
- `apply_masks` (line 236) `def apply_masks(self)` - *Apply sparsity masks to weights*
- `get_sparsity` (line 243) `def get_sparsity(self)` - *Calculate overall sparsity percentage*
- `forward` (line 249) `def forward(self, x)` - *Input:  [B, input_dim]
Output: ([B, num_classes], [B, num_superclasses])*
- `__init__` (line 287) `def __init__(self, epsilon)`
- `compute_metrics` (line 290) `def compute_metrics(self, weight)` - *Compute spectral coherence metrics*
- `get_singular_values` (line 306) `def get_singular_values(self, weight)` - *Get singular values for visualization*
- `__init__` (line 318) `def __init__(self, semantic_plasticity_threshold, stagnation_limit, mixer_noise_scale, dominant_energy_threshold, geo_window)`
- `compute_semantic_plasticity_ratio` (line 330) `def compute_semantic_plasticity_ratio(self)` - *Compute ratio of semantic gain to structural change*
- `detect_intervention_need` (line 343) `def detect_intervention_need(self, phase_state, extractor)` - *Determine if intervention is needed using adaptive criteria*
- `update_history` (line 365) `def update_history(self, topo_ratio, coarse_acc)` - *Update history for adaptive control*
- `perturb_mixer_targeted` (line 382) `def perturb_mixer_targeted(self, extractor)` - *Targeted Spectral Surgery: Inject noise in the nullspace of dominant subspace*
- `__init__` (line 417) `def __init__(self, device)`
- `apply_rank_capping` (line 421) `def apply_rank_capping(self, model, layer_name, keep_ratio)` - *Apply rank capping shock to prevent over-specialization*
- `create_refined_model` (line 438) `def create_refined_model(self, parent_state, data_loader, feature_extractor, lambda_taxonomic, learning_rate)` - *Create refined model through gradient-based inheritance*
- `__getitem__` (line 489) `def __getitem__(self, index)`
- `evaluate` (line 507) `def evaluate(model, extractor, name)`
- `get_singular_values` (line 556) `def get_singular_values(model, extractor, name)` - *Get singular values from model weights*
- `__init__` (line 613) `def __init__(self, device, output_dir)`
- `load_data` (line 639) `def load_data(self, cycle, batch_size)` - *Load curriculum dataset based on refinement cycle*
- `train_model` (line 681) `def train_model(self, model, cycle, chain_type, feature_extractor)` - *Train model with iterative refinement and hierarchical learning*
- `detect_phase_state` (line 908) `def detect_phase_state(self, ratio_history, phase_window, phase_std_dev_limit)` - *Detect phase state based on topology ratio history*
- `compute_topology_ratio` (line 926) `def compute_topology_ratio(self, model, extractor, chain_type)` - *Calculates Topo_R = L_opt / L_mon.*
- `run_refinement` (line 944) `def run_refinement(self, num_iterations, num_seeds, early_stop_patience)` - *Run full iterative refinement experiment with statistical validation*
- `_save_results` (line 1126) `def _save_results(self, all_results, best_overall, hierarchy_delta)` - *Save results to files*
- `_plot_results` (line 1181) `def _plot_results(self, all_results)` - *Create publication-quality plots*

#### `apex29.py`
**Path:** `apex29.py`

**Classes:**
- `GatedTokenMixer` (line 86) `class GatedTokenMixer` - *Efficient token mixer with gating mechanism for feature interaction*
- `PatchFeatureExtractor` (line 143) `class PatchFeatureExtractor` - *Efficient patch-based feature extractor with optional token mixing*
- `TaxonomicMLP` (line 204) `class TaxonomicMLP` - *Sparse MLP with taxonomic heads for hierarchical learning*
- `SpectralMonitor` (line 278) `class SpectralMonitor` - *Monitor spectral properties with adaptive analysis*
- `AdaptiveTopologyController` (line 309) `class AdaptiveTopologyController` - *Self-referential controller using semantic-plasticity ratio*
- `IterativeRefinementEngine` (line 408) `class IterativeRefinementEngine` - *Engine for iterative refinement through spectral control*
- `CoarseCIFAR100` (line 476) `class CoarseCIFAR100` - *Wrapper that converts CIFAR100 into a pure 20-class classification problem (Superclasses).
Used to validate learned inductive bias.*
- `IterativeRefinementTrainer` (line 606) `class IterativeRefinementTrainer` - *Framework for iterative refinement with statistical validation*

**Functions:**
- `set_seed` (line 43) `def set_seed(seed)` - *Ensure full reproducibility across runs*
- `compute_spectral_loss` (line 262) `def compute_spectral_loss(W)` - *Optimization Objective for Spectral Control (L_opt)*
- `run_hierarchy_benchmark` (line 486) `def run_hierarchy_benchmark(model_apex, model_blind, feature_extractor_apex, feature_extractor_blind, device)` - *Run hierarchy stress test to validate inductive bias transfer*
- `visualize_singular_values` (line 543) `def visualize_singular_values(model_apex, model_blind, feature_extractor_apex, feature_extractor_blind, device, output_dir)` - *Generate publication-quality singular value visualizations*
- `parse_args` (line 1244) `def parse_args()`
- `main` (line 1255) `def main()` - *Main execution function*
- `__init__` (line 88) `def __init__(self, num_patches, embed_dim)`
- `_init_weights` (line 111) `def _init_weights(self)` - *Initialize weights for stable training*
- `forward` (line 125) `def forward(self, x)` - *Input:  [B, num_patches, embed_dim]
Output: [B, num_patches, embed_dim]*
- `__init__` (line 145) `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)`
- `freeze` (line 174) `def freeze(self)` - *Freeze all parameters for transfer learning*
- `unfreeze_mixer_only` (line 180) `def unfreeze_mixer_only(self)` - *Unfreeze only the mixer parameters for fine-tuning*
- `forward` (line 187) `def forward(self, x)` - *Input:  [B, C, H, W]
Output: [B, embed_dim]*
- `__init__` (line 206) `def __init__(self, input_dim, hidden_dim, num_classes, num_superclasses)`
- `apply_masks` (line 229) `def apply_masks(self)` - *Apply sparsity masks to weights*
- `get_sparsity` (line 236) `def get_sparsity(self)` - *Calculate overall sparsity percentage*
- `forward` (line 242) `def forward(self, x)` - *Input:  [B, input_dim]
Output: ([B, num_classes], [B, num_superclasses])*
- `__init__` (line 280) `def __init__(self, epsilon)`
- `compute_metrics` (line 283) `def compute_metrics(self, weight)` - *Compute spectral coherence metrics*
- `get_singular_values` (line 299) `def get_singular_values(self, weight)` - *Get singular values for visualization*
- `__init__` (line 311) `def __init__(self, semantic_plasticity_threshold, stagnation_limit, mixer_noise_scale, dominant_energy_threshold, geo_window)`
- `compute_semantic_plasticity_ratio` (line 323) `def compute_semantic_plasticity_ratio(self)` - *Compute ratio of semantic gain to structural change*
- `detect_intervention_need` (line 336) `def detect_intervention_need(self, phase_state, extractor)` - *Determine if intervention is needed using adaptive criteria*
- `update_history` (line 358) `def update_history(self, topo_ratio, coarse_acc)` - *Update history for adaptive control*
- `perturb_mixer_targeted` (line 375) `def perturb_mixer_targeted(self, extractor)` - *Targeted Spectral Surgery: Inject noise in the nullspace of dominant subspace*
- `__init__` (line 410) `def __init__(self, device)`
- `apply_rank_capping` (line 414) `def apply_rank_capping(self, model, layer_name, keep_ratio)` - *Apply rank capping shock to prevent over-specialization*
- `create_refined_model` (line 430) `def create_refined_model(self, parent_state, data_loader, feature_extractor, lambda_taxonomic, learning_rate)` - *Create refined model through gradient-based inheritance*
- `__getitem__` (line 481) `def __getitem__(self, index)`
- `evaluate` (line 499) `def evaluate(model, extractor, name)`
- `get_singular_values` (line 551) `def get_singular_values(model, extractor, name)` - *Get singular values from model weights*
- `__init__` (line 608) `def __init__(self, device, output_dir)`
- `load_data` (line 634) `def load_data(self, cycle, batch_size)` - *Load curriculum dataset based on refinement cycle*
- `train_model` (line 674) `def train_model(self, model, cycle, chain_type, feature_extractor)` - *Train model with iterative refinement and hierarchical learning*
- `detect_phase_state` (line 894) `def detect_phase_state(self, ratio_history, phase_window, phase_std_dev_limit)` - *Detect phase state based on topology ratio history*
- `compute_topology_ratio` (line 912) `def compute_topology_ratio(self, model, extractor, chain_type)` - *Calculates Topo_R = L_opt / L_mon.*
- `run_refinement` (line 931) `def run_refinement(self, num_iterations, num_seeds, early_stop_patience)` - *Run full iterative refinement experiment with statistical validation*
- `_save_results` (line 1109) `def _save_results(self, all_results, best_overall, hierarchy_delta)` - *Save results to files*
- `_plot_results` (line 1164) `def _plot_results(self, all_results)` - *Create publication-quality plots*

#### `apex30.py`
**Path:** `apex30.py`

**Classes:**
- `GatedTokenMixer` (line 87) `class GatedTokenMixer` - *Efficient token mixer with gating mechanism for feature interaction*
- `PatchFeatureExtractor` (line 142) `class PatchFeatureExtractor`
- `TaxonomicMLP` (line 185) `class TaxonomicMLP` - *Sparse MLP with taxonomic heads for hierarchical learning*
- `SpectralMonitor` (line 248) `class SpectralMonitor` - *Monitor spectral properties with adaptive analysis*
- `AdaptiveTopologyController` (line 277) `class AdaptiveTopologyController` - *Self-referential controller using semantic-plasticity ratio*
- `IterativeRefinementEngine` (line 397) `class IterativeRefinementEngine` - *Engine for iterative refinement through spectral control*
- `CoarseCIFAR100` (line 459) `class CoarseCIFAR100`
- `IterativeRefinementTrainer` (line 570) `class IterativeRefinementTrainer`

**Functions:**
- `set_seed` (line 43) `def set_seed(seed)` - *Ensure full reproducibility across runs*
- `compute_spectral_loss` (line 232) `def compute_spectral_loss(W)` - *Optimization Objective for Spectral Control (L_opt)*
- `run_hierarchy_benchmark` (line 464) `def run_hierarchy_benchmark(model_apex, model_blind, feature_extractor_apex, feature_extractor_blind, device)`
- `visualize_singular_values` (line 510) `def visualize_singular_values(model_apex, model_blind, feature_extractor_apex, feature_extractor_blind, device, output_dir, monitor)` - *v18.1 FIX: Now accepts 'monitor' explicitly.
Generate publication-quality singular value visualizations.*
- `parse_args` (line 1062) `def parse_args()`
- `main` (line 1071) `def main()`
- `__init__` (line 89) `def __init__(self, num_patches, embed_dim)`
- `_init_weights` (line 111) `def _init_weights(self)`
- `forward` (line 134) `def forward(self, x)`
- `__init__` (line 143) `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)`
- `freeze` (line 164) `def freeze(self)`
- `unfreeze_mixer_only` (line 169) `def unfreeze_mixer_only(self)`
- `forward` (line 175) `def forward(self, x)`
- `__init__` (line 187) `def __init__(self, input_dim, hidden_dim, num_classes, num_superclasses)`
- `apply_masks` (line 207) `def apply_masks(self)`
- `get_sparsity` (line 213) `def get_sparsity(self)`
- `forward` (line 218) `def forward(self, x)`
- `__init__` (line 250) `def __init__(self, epsilon)`
- `compute_metrics` (line 253) `def compute_metrics(self, weight)`
- `get_singular_values` (line 268) `def get_singular_values(self, weight)`
- `__init__` (line 279) `def __init__(self, semantic_plasticity_threshold, stagnation_limit, mixer_noise_scale, dominant_energy_threshold, geo_window)`
- `compute_semantic_plasticity_ratio` (line 295) `def compute_semantic_plasticity_ratio(self)` - *Compute ratio of semantic gain to structural change*
- `detect_intervention_need` (line 308) `def detect_intervention_need(self, phase_state, extractor)` - *Determine if intervention is needed using adaptive criteria.
Implements hysteresis to avoid intervention during active SHIFTING phases.*
- `update_history` (line 348) `def update_history(self, topo_ratio, coarse_acc)` - *Update history for adaptive control*
- `perturb_mixer_targeted` (line 363) `def perturb_mixer_targeted(self, extractor)` - *Targeted Spectral Surgery: Inject noise in the nullspace of dominant subspace*
- `__init__` (line 399) `def __init__(self, device)`
- `apply_rank_capping` (line 403) `def apply_rank_capping(self, model, layer_name, keep_ratio)`
- `create_refined_model` (line 419) `def create_refined_model(self, parent_state, data_loader, feature_extractor, lambda_taxonomic, learning_rate)`
- `__getitem__` (line 460) `def __getitem__(self, index)`
- `evaluate` (line 471) `def evaluate(model, extractor)`
- `get_singular_values` (line 521) `def get_singular_values(model, extractor)`
- `__init__` (line 571) `def __init__(self, device, output_dir)`
- `load_data` (line 593) `def load_data(self, cycle, batch_size)`
- `train_model` (line 617) `def train_model(self, model, cycle, chain_type, feature_extractor)`
- `detect_phase_state` (line 797) `def detect_phase_state(self, ratio_history, phase_window, phase_std_dev_limit)`
- `compute_topology_ratio` (line 813) `def compute_topology_ratio(self, model, extractor, chain_type)`
- `run_refinement` (line 828) `def run_refinement(self, num_iterations, num_seeds, early_stop_patience)`
- `_save_results` (line 956) `def _save_results(self, all_results, best_overall, hierarchy_delta)`
- `_plot_results` (line 983) `def _plot_results(self, all_results, best_overall)` - *v18.1 FIX: Explicitly accepts best_overall to fix scope bug.
Create publication-quality plots.*

#### `apex31.py`
**Path:** `apex31.py`

**Classes:**
- `GatedTokenMixer` (line 89) `class GatedTokenMixer` - *Efficient token mixer with gating mechanism for feature interaction*
- `PatchFeatureExtractor` (line 144) `class PatchFeatureExtractor`
- `TaxonomicMLP` (line 187) `class TaxonomicMLP` - *Sparse MLP with taxonomic heads for hierarchical learning*
- `SpectralMonitor` (line 250) `class SpectralMonitor` - *Monitor spectral properties with adaptive analysis*
- `AdaptiveTopologyController` (line 279) `class AdaptiveTopologyController` - *Self-referential controller using semantic-plasticity ratio*
- `IterativeRefinementEngine` (line 404) `class IterativeRefinementEngine` - *Engine for iterative refinement through spectral control*
- `CoarseCIFAR100` (line 470) `class CoarseCIFAR100`
- `IterativeRefinementTrainer` (line 581) `class IterativeRefinementTrainer`

**Functions:**
- `set_seed` (line 45) `def set_seed(seed)` - *Ensure full reproducibility across runs*
- `compute_spectral_loss` (line 234) `def compute_spectral_loss(W)` - *Optimization Objective for Spectral Control (L_opt)*
- `run_hierarchy_benchmark` (line 475) `def run_hierarchy_benchmark(model_apex, model_blind, feature_extractor_apex, feature_extractor_blind, device)`
- `visualize_singular_values` (line 521) `def visualize_singular_values(model_apex, model_blind, feature_extractor_apex, feature_extractor_blind, device, output_dir, monitor)` - *v18.1 FIX: Now accepts 'monitor' explicitly.
Generate publication-quality singular value visualizations.*
- `parse_args` (line 1073) `def parse_args()`
- `main` (line 1082) `def main()`
- `__init__` (line 91) `def __init__(self, num_patches, embed_dim)`
- `_init_weights` (line 113) `def _init_weights(self)`
- `forward` (line 136) `def forward(self, x)`
- `__init__` (line 145) `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)`
- `freeze` (line 166) `def freeze(self)`
- `unfreeze_mixer_only` (line 171) `def unfreeze_mixer_only(self)`
- `forward` (line 177) `def forward(self, x)`
- `__init__` (line 189) `def __init__(self, input_dim, hidden_dim, num_classes, num_superclasses)`
- `apply_masks` (line 209) `def apply_masks(self)`
- `get_sparsity` (line 215) `def get_sparsity(self)`
- `forward` (line 220) `def forward(self, x)`
- `__init__` (line 252) `def __init__(self, epsilon)`
- `compute_metrics` (line 255) `def compute_metrics(self, weight)`
- `get_singular_values` (line 270) `def get_singular_values(self, weight)`
- `__init__` (line 281) `def __init__(self, semantic_plasticity_threshold, stagnation_limit, mixer_noise_scale, dominant_energy_threshold, geo_window)`
- `compute_semantic_plasticity_ratio` (line 297) `def compute_semantic_plasticity_ratio(self)` - *Compute ratio of semantic gain to structural change*
- `detect_intervention_need` (line 310) `def detect_intervention_need(self, phase_state, extractor)` - *Determine if intervention is needed using adaptive criteria.
Implements hysteresis to avoid intervention during active SHIFTING phases.*
- `update_history` (line 354) `def update_history(self, topo_ratio, coarse_acc)` - *Update history for adaptive control*
- `perturb_mixer_targeted` (line 369) `def perturb_mixer_targeted(self, extractor)` - *Targeted Spectral Surgery: Inject noise in the nullspace of dominant subspace*
- `__init__` (line 406) `def __init__(self, device)`
- `apply_rank_capping` (line 410) `def apply_rank_capping(self, model, layer_name, keep_ratio)`
- `create_refined_model` (line 430) `def create_refined_model(self, parent_state, data_loader, feature_extractor, lambda_taxonomic, learning_rate)`
- `__getitem__` (line 471) `def __getitem__(self, index)`
- `evaluate` (line 482) `def evaluate(model, extractor)`
- `get_singular_values` (line 532) `def get_singular_values(model, extractor)`
- `__init__` (line 582) `def __init__(self, device, output_dir)`
- `load_data` (line 604) `def load_data(self, cycle, batch_size)`
- `train_model` (line 628) `def train_model(self, model, cycle, chain_type, feature_extractor)`
- `detect_phase_state` (line 808) `def detect_phase_state(self, ratio_history, phase_window, phase_std_dev_limit)`
- `compute_topology_ratio` (line 824) `def compute_topology_ratio(self, model, extractor, chain_type)`
- `run_refinement` (line 839) `def run_refinement(self, num_iterations, num_seeds, early_stop_patience)`
- `_save_results` (line 967) `def _save_results(self, all_results, best_overall, hierarchy_delta)`
- `_plot_results` (line 994) `def _plot_results(self, all_results, best_overall)` - *v18.1 FIX: Explicitly accepts best_overall to fix scope bug.
Create publication-quality plots.*

#### `apex32.py`
**Path:** `apex32.py`

**Classes:**
- `GatedTokenMixer` (line 92) `class GatedTokenMixer`
- `PatchFeatureExtractor` (line 109) `class PatchFeatureExtractor`
- `TaxonomicMLP` (line 148) `class TaxonomicMLP`
- `SpectralMonitor` (line 187) `class SpectralMonitor`
- `TaxonomicTrainer` (line 226) `class TaxonomicTrainer`
- `CoarseCIFAR100` (line 404) `class CoarseCIFAR100`

**Functions:**
- `compute_spectral_loss` (line 76) `def compute_spectral_loss(W)` - *L_opt: Optimization Objective for Structural Control.*
- `run_hierarchy_benchmark` (line 409) `def run_hierarchy_benchmark(model_apex, model_blind, device)`
- `main` (line 455) `def main()`
- `__init__` (line 93) `def __init__(self, num_patches, embed_dim)`
- `forward` (line 102) `def forward(self, x)`
- `__init__` (line 110) `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)`
- `freeze` (line 126) `def freeze(self)`
- `unfreeze_mixer_only` (line 130) `def unfreeze_mixer_only(self)`
- `forward` (line 136) `def forward(self, x)`
- `__init__` (line 149) `def __init__(self, input_dim, hidden_dim, num_classes)`
- `apply_masks` (line 164) `def apply_masks(self)` - *Zero out weights based on masks. Runs on device (CUDA).*
- `get_sparsity` (line 171) `def get_sparsity(self)`
- `forward` (line 176) `def forward(self, x)`
- `compute_metrics` (line 188) `def compute_metrics(self, weight)`
- `detect_phase_state` (line 200) `def detect_phase_state(self, ratio_history)`
- `compute_topology_ratio` (line 211) `def compute_topology_ratio(self, model, extractor, chain_type)`
- `__init__` (line 227) `def __init__(self, device, extractor_apex, extractor_blind)`
- `_preprocess_batch` (line 234) `def _preprocess_batch(self, x, extractor)`
- `get_curriculum_dataset` (line 238) `def get_curriculum_dataset(self, cycle)`
- `train_single_chain` (line 244) `def train_single_chain(self, model, cycle, chain_type)`
- `__getitem__` (line 405) `def __getitem__(self, index)`
- `evaluate` (line 422) `def evaluate(model, extractor, name)`

#### `apex33.py`
**Path:** `apex33.py`

**Classes:**
- `GatedTokenMixer` (line 72) `class GatedTokenMixer` - *Efficient token mixer with high-std initialization for exploration*
- `PatchFeatureExtractor` (line 118) `class PatchFeatureExtractor`
- `TaxonomicMLP` (line 150) `class TaxonomicMLP` - *Sparse MLP with taxonomic heads*
- `SpectralMonitor` (line 201) `class SpectralMonitor`
- `AdaptiveTopologyController` (line 227) `class AdaptiveTopologyController` - *Self-referential controller with Hysteresis*
- `IterativeRefinementEngine` (line 310) `class IterativeRefinementEngine`
- `IterativeRefinementTrainer` (line 343) `class IterativeRefinementTrainer`
- `CoarseCIFAR100` (line 802) `class CoarseCIFAR100`

**Functions:**
- `set_seed` (line 41) `def set_seed(seed)` - *Ensure full reproducibility across runs*
- `compute_spectral_loss` (line 190) `def compute_spectral_loss(W)`
- `run_hierarchy_benchmark` (line 807) `def run_hierarchy_benchmark(model_apex, model_blind, feature_extractor_apex, feature_extractor_blind, device)`
- `visualize_singular_values` (line 845) `def visualize_singular_values(model_apex, model_blind, feature_extractor_apex, feature_extractor_blind, device, output_dir, monitor)`
- `parse_args` (line 894) `def parse_args()`
- `main` (line 903) `def main()`
- `__init__` (line 74) `def __init__(self, num_patches, embed_dim)`
- `_init_weights` (line 92) `def _init_weights(self)`
- `forward` (line 110) `def forward(self, x)`
- `__init__` (line 119) `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)`
- `freeze` (line 135) `def freeze(self)`
- `unfreeze_mixer_only` (line 138) `def unfreeze_mixer_only(self)`
- `forward` (line 143) `def forward(self, x)`
- `__init__` (line 152) `def __init__(self, input_dim, hidden_dim, num_classes, num_superclasses)`
- `apply_masks` (line 167) `def apply_masks(self)`
- `get_sparsity` (line 173) `def get_sparsity(self)`
- `forward` (line 178) `def forward(self, x)`
- `__init__` (line 202) `def __init__(self, epsilon)`
- `compute_metrics` (line 205) `def compute_metrics(self, weight)`
- `get_singular_values` (line 219) `def get_singular_values(self, weight)`
- `__init__` (line 229) `def __init__(self, semantic_plasticity_threshold, stagnation_limit, mixer_noise_scale, dominant_energy_threshold, geo_window)`
- `compute_semantic_plasticity_ratio` (line 242) `def compute_semantic_plasticity_ratio(self)`
- `detect_intervention_need` (line 250) `def detect_intervention_need(self, phase_state, extractor)`
- `update_history` (line 273) `def update_history(self, topo_ratio, coarse_acc)`
- `perturb_mixer_targeted` (line 284) `def perturb_mixer_targeted(self, extractor)`
- `__init__` (line 311) `def __init__(self, device)`
- `create_refined_model` (line 315) `def create_refined_model(self, parent_state, data_loader, feature_extractor, lambda_taxonomic, learning_rate)`
- `__init__` (line 344) `def __init__(self, device, output_dir)`
- `load_data` (line 367) `def load_data(self, cycle, batch_size)`
- `train_model` (line 385) `def train_model(self, model, cycle, chain_type, feature_extractor)`
- `detect_phase_state` (line 570) `def detect_phase_state(self, ratio_history, phase_window, phase_std_dev_limit)`
- `compute_topology_ratio` (line 580) `def compute_topology_ratio(self, model, extractor, chain_type)`
- `run_refinement` (line 592) `def run_refinement(self, num_iterations, num_seeds, early_stop_patience)`
- `_save_results` (line 712) `def _save_results(self, all_results, best_overall, hierarchy_delta)`
- `_plot_results` (line 738) `def _plot_results(self, all_results, best_overall)`
- `__getitem__` (line 803) `def __getitem__(self, index)`
- `evaluate` (line 814) `def evaluate(model, extractor)`
- `get_singular_values` (line 851) `def get_singular_values(model, extractor)`

#### `apex34.py`
**Path:** `apex34.py`

**Classes:**
- `GatedTokenMixer` (line 72) `class GatedTokenMixer` - *Efficient token mixer with high-std initialization for exploration*
- `PatchFeatureExtractor` (line 118) `class PatchFeatureExtractor`
- `TaxonomicMLP` (line 150) `class TaxonomicMLP` - *Sparse MLP with taxonomic heads*
- `SpectralMonitor` (line 201) `class SpectralMonitor`
- `AdaptiveTopologyController` (line 227) `class AdaptiveTopologyController` - *Self-referential controller with Hysteresis*
- `IterativeRefinementEngine` (line 310) `class IterativeRefinementEngine`
- `IterativeRefinementTrainer` (line 343) `class IterativeRefinementTrainer`
- `CoarseCIFAR100` (line 802) `class CoarseCIFAR100`

**Functions:**
- `set_seed` (line 41) `def set_seed(seed)` - *Ensure full reproducibility across runs*
- `compute_spectral_loss` (line 190) `def compute_spectral_loss(W)`
- `run_hierarchy_benchmark` (line 807) `def run_hierarchy_benchmark(model_apex, model_blind, feature_extractor_apex, feature_extractor_blind, device)`
- `visualize_singular_values` (line 845) `def visualize_singular_values(model_apex, model_blind, feature_extractor_apex, feature_extractor_blind, device, output_dir, monitor)`
- `main` (line 894) `def main()`
- `__init__` (line 74) `def __init__(self, num_patches, embed_dim)`
- `_init_weights` (line 92) `def _init_weights(self)`
- `forward` (line 110) `def forward(self, x)`
- `__init__` (line 119) `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)`
- `freeze` (line 135) `def freeze(self)`
- `unfreeze_mixer_only` (line 138) `def unfreeze_mixer_only(self)`
- `forward` (line 143) `def forward(self, x)`
- `__init__` (line 152) `def __init__(self, input_dim, hidden_dim, num_classes, num_superclasses)`
- `apply_masks` (line 167) `def apply_masks(self)`
- `get_sparsity` (line 173) `def get_sparsity(self)`
- `forward` (line 178) `def forward(self, x)`
- `__init__` (line 202) `def __init__(self, epsilon)`
- `compute_metrics` (line 205) `def compute_metrics(self, weight)`
- `get_singular_values` (line 219) `def get_singular_values(self, weight)`
- `__init__` (line 229) `def __init__(self, semantic_plasticity_threshold, stagnation_limit, mixer_noise_scale, dominant_energy_threshold, geo_window)`
- `compute_semantic_plasticity_ratio` (line 242) `def compute_semantic_plasticity_ratio(self)`
- `detect_intervention_need` (line 250) `def detect_intervention_need(self, phase_state, extractor)`
- `update_history` (line 273) `def update_history(self, topo_ratio, coarse_acc)`
- `perturb_mixer_targeted` (line 284) `def perturb_mixer_targeted(self, extractor)`
- `__init__` (line 311) `def __init__(self, device)`
- `create_refined_model` (line 315) `def create_refined_model(self, parent_state, data_loader, feature_extractor, lambda_taxonomic, learning_rate)`
- `__init__` (line 344) `def __init__(self, device, output_dir)`
- `load_data` (line 367) `def load_data(self, cycle, batch_size)`
- `train_model` (line 385) `def train_model(self, model, cycle, chain_type, feature_extractor)`
- `detect_phase_state` (line 570) `def detect_phase_state(self, ratio_history, phase_window, phase_std_dev_limit)`
- `compute_topology_ratio` (line 580) `def compute_topology_ratio(self, model, extractor, chain_type)`
- `run_refinement` (line 592) `def run_refinement(self, num_iterations, num_seeds, early_stop_patience)`
- `_save_results` (line 712) `def _save_results(self, all_results, best_overall, hierarchy_delta)`
- `_plot_results` (line 738) `def _plot_results(self, all_results, best_overall)`
- `__getitem__` (line 803) `def __getitem__(self, index)`
- `evaluate` (line 814) `def evaluate(model, extractor)`
- `get_singular_values` (line 851) `def get_singular_values(model, extractor)`

#### `apex35.py`
**Path:** `apex35.py`

**Classes:**
- `GatedTokenMixer` (line 66) `class GatedTokenMixer` - *Chaotic Mixer for Emergent Regime*
- `E8FusionLayer` (line 112) `class E8FusionLayer` - *🕸️ E8 Lattice Fusion (Synergy Component E)
Optimized version from Suite v4.0.
Fuses geometric structure (Orthogonal Proj) with attention.*
- `PatchFeatureExtractor` (line 156) `class PatchFeatureExtractor`
- `TaxonomicMLP` (line 188) `class TaxonomicMLP`
- `BlackMirrorMonitor` (line 227) `class BlackMirrorMonitor` - *Passive Ontological Monitor*
- `IterativeRefinementTrainer` (line 253) `class IterativeRefinementTrainer`
- `CoarseCIFAR100` (line 439) `class CoarseCIFAR100`

**Functions:**
- `set_seed` (line 36) `def set_seed(seed)`
- `main` (line 444) `def main()`
- `__init__` (line 68) `def __init__(self, num_patches, embed_dim)`
- `_init_weights` (line 86) `def _init_weights(self)`
- `forward` (line 104) `def forward(self, x)`
- `__init__` (line 118) `def __init__(self, embed_dim, num_heads)`
- `forward` (line 136) `def forward(self, x)`
- `__init__` (line 157) `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)`
- `freeze` (line 173) `def freeze(self)`
- `unfreeze_mixer_only` (line 176) `def unfreeze_mixer_only(self)`
- `forward` (line 181) `def forward(self, x)`
- `__init__` (line 189) `def __init__(self, input_dim, hidden_dim, num_classes, num_superclasses)`
- `apply_masks` (line 204) `def apply_masks(self)`
- `get_sparsity` (line 210) `def get_sparsity(self)`
- `forward` (line 215) `def forward(self, x)`
- `__init__` (line 229) `def __init__(self, epsilon)`
- `inspect` (line 232) `def inspect(self, weight)`
- `__init__` (line 254) `def __init__(self, device, output_dir)`
- `load_data` (line 274) `def load_data(self, cycle, batch_size)`
- `train_model` (line 292) `def train_model(self, model, cycle, chain_type, feature_extractor)`
- `__getitem__` (line 440) `def __getitem__(self, index)`
- `evaluate_safe` (line 483) `def evaluate_safe(model, extractor)`

#### `app.py`
**Path:** `app.py`

**Classes:**
- `SpectralMonitor` (line 51) `class SpectralMonitor`
- `PersistentPruner` (line 76) `class PersistentPruner`
- `SpectralMLP` (line 98) `class SpectralMLP`
- `EvolutionaryResonanceEngine` (line 121) `class EvolutionaryResonanceEngine`

**Functions:**
- `main` (line 561) `def main()`
- `__init__` (line 52) `def __init__(self, epsilon_c)`
- `compute_L` (line 55) `def compute_L(self, weight)`
- `__init__` (line 77) `def __init__(self, sparsity_target)`
- `apply_to_model` (line 81) `def apply_to_model(self, model)`
- `enforce_during_training` (line 91) `def enforce_during_training(self, model)`
- `__init__` (line 99) `def __init__(self, input_dim, hidden_dim, num_classes)`
- `reduce_input` (line 107) `def reduce_input(self, x)`
- `forward` (line 113) `def forward(self, x)`
- `__init__` (line 122) `def __init__(self, device, base_target_acc)`
- `load_best_legacy_model` (line 137) `def load_best_legacy_model(self, cycle)` - *Load the best model from previous cycle, with fallback to initial seed*
- `train_base_model_to_target` (line 175) `def train_base_model_to_target(self, hidden_dim, target_acc, test_loader)` - *Train a base model to target accuracy*
- `extract_seed_from_checkpoint` (line 228) `def extract_seed_from_checkpoint(self, checkpoint, expected_hidden_dim)` - *Extract seed weights from checkpoint, handling different formats*
- `extract_seed_weights` (line 254) `def extract_seed_weights(self, model)`
- `inoculate_seed_adaptive` (line 260) `def inoculate_seed_adaptive(self, large_model, seed_weights)` - *Adaptive inoculation that handles dimension mismatches*
- `measure_functional_alignment` (line 288) `def measure_functional_alignment(self, model1, model2, test_loader)` - *Measure functional alignment via logit cosine similarity*
- `progressive_pruning_with_target` (line 309) `def progressive_pruning_with_target(self, model, target_acc, test_loader, max_density)` - *Prune while maintaining target accuracy, with density constraint*
- `execute_resonance_cycle` (line 351) `def execute_resonance_cycle(self, cycle, test_loader, expansion_factor)`
- `run_evolutionary_experiment` (line 484) `def run_evolutionary_experiment(self, num_cycles)`

#### `plank.py`
**Path:** `plank.py`

**Classes:**
- `BlackMirrorMonitor` (line 28) `class BlackMirrorMonitor` - *Calcula el Lagrangiano de Verdad L usando entropía de von Neumann y rango efectivo.
Umbrales calibrados empíricamente para detectar mentiras estructurales (10% ruido).*
- `SovereignNeuron` (line 62) `class SovereignNeuron`
- `NeuroSovereign` (line 108) `class NeuroSovereign`
- `SovereignTrainer` (line 132) `class SovereignTrainer`

**Functions:**
- `main` (line 178) `def main()`
- `__init__` (line 33) `def __init__(self, epsilon_c)`
- `inspect` (line 36) `def inspect(self, weights)`
- `__init__` (line 63) `def __init__(self, in_features, out_features, sparsity_target)`
- `forward` (line 70) `def forward(self, x, inject_lies)`
- `apply_black_swan_refraction` (line 87) `def apply_black_swan_refraction(self)` - *Purificación extrema: sparsity 0.0004%*
- `__init__` (line 109) `def __init__(self, sparsity_target)`
- `forward` (line 117) `def forward(self, x, inject_lies)`
- `__init__` (line 133) `def __init__(self, model, device)`
- `train_epoch` (line 139) `def train_epoch(self, dataloader, epoch)`

#### `plank10.py`
**Path:** `plank10.py`

**Classes:**
- `PatchFeatureExtractor` (line 34) `class PatchFeatureExtractor` - *Extrae características mediante Patch Embedding y añade una capa de mezcla (Mixer).
Esto permite al modelo aprender relaciones espaciales entre parches antes de la clasificación.*
- `LotteryMLP` (line 91) `class LotteryMLP`
- `StandardBaseline` (line 121) `class StandardBaseline` - *Baseline moderno (Patch + Mixer + MLP simple) sin evolución.*
- `SpectralMonitor` (line 133) `class SpectralMonitor`
- `ApexEvolutionEngine` (line 148) `class ApexEvolutionEngine`
- `ApexTrainer` (line 247) `class ApexTrainer`

**Functions:**
- `main` (line 354) `def main()`
- `__init__` (line 39) `def __init__(self, img_size, patch_size, in_chans, embed_dim)`
- `freeze` (line 67) `def freeze(self)`
- `unfreeze` (line 71) `def unfreeze(self)`
- `forward` (line 75) `def forward(self, x)`
- `__init__` (line 92) `def __init__(self, input_dim, hidden_dim, num_classes)`
- `apply_masks` (line 104) `def apply_masks(self)`
- `get_sparsity` (line 109) `def get_sparsity(self)`
- `forward` (line 114) `def forward(self, x)`
- `__init__` (line 123) `def __init__(self, input_dim, hidden_dim)`
- `forward` (line 127) `def forward(self, x)`
- `compute_L` (line 134) `def compute_L(self, weight)`
- `__init__` (line 149) `def __init__(self, device)`
- `_gradient_nudge_inheritance` (line 154) `def _gradient_nudge_inheritance(self, child_model, elk_state, data_loader, feature_extractor, nudge_lr)`
- `_apply_dynamic_spectral_shock` (line 188) `def _apply_dynamic_spectral_shock(self, model, layer_name)`
- `create_apex_offspring` (line 211) `def create_apex_offspring(self, elk_state, new_hidden_dim, cycle, data_loader, feature_extractor, parent_gap)`
- `__init__` (line 248) `def __init__(self, device, feature_extractor)`
- `_preprocess_batch` (line 255) `def _preprocess_batch(self, x)`
- `get_curriculum_dataset` (line 259) `def get_curriculum_dataset(self, cycle)`
- `train_model` (line 265) `def train_model(self, model, cycle, is_baseline)`

#### `plank11.py`
**Path:** `plank11.py`

**Classes:**
- `TokenMixer` (line 36) `class TokenMixer` - *Mezcla tokens entre sí.
Input: (B, T, D) -> Transpose -> (B, D, T) -> Linear -> (B, D, T) -> Transpose*
- `PatchFeatureExtractor` (line 57) `class PatchFeatureExtractor` - *ViT-Lite + True Token Mixing.*
- `LotteryMLP` (line 97) `class LotteryMLP`
- `StandardBaseline` (line 126) `class StandardBaseline`
- `SpectralMonitor` (line 137) `class SpectralMonitor`
- `OrthogonalEvolutionEngine` (line 152) `class OrthogonalEvolutionEngine`
- `OrthogonalTrainer` (line 260) `class OrthogonalTrainer`

**Functions:**
- `main` (line 379) `def main()`
- `__init__` (line 41) `def __init__(self, num_tokens)`
- `forward` (line 50) `def forward(self, x)`
- `__init__` (line 61) `def __init__(self, img_size, patch_size, in_chans, embed_dim)`
- `freeze` (line 75) `def freeze(self)`
- `unfreeze` (line 79) `def unfreeze(self)`
- `forward` (line 83) `def forward(self, x)`
- `__init__` (line 98) `def __init__(self, input_dim, hidden_dim, num_classes)`
- `apply_masks` (line 109) `def apply_masks(self)`
- `get_sparsity` (line 114) `def get_sparsity(self)`
- `forward` (line 119) `def forward(self, x)`
- `__init__` (line 127) `def __init__(self, input_dim, hidden_dim)`
- `forward` (line 131) `def forward(self, x)`
- `compute_L` (line 138) `def compute_L(self, weight)`
- `__init__` (line 153) `def __init__(self, device)`
- `_gradient_nudge_inheritance` (line 158) `def _gradient_nudge_inheritance(self, child_model, elk_state, data_loader, feature_extractor, nudge_lr)`
- `_apply_minimalistic_shock` (line 190) `def _apply_minimalistic_shock(self, model, layer_name, target_rank_ratio)` - *Rank Capping: Cortamos singular values débiles y NO renormalizamos.
Esto fuerza la minimización (Energy Decay).*
- `create_orthogonal_offspring` (line 223) `def create_orthogonal_offspring(self, elk_state, new_hidden_dim, cycle, data_loader, feature_extractor, parent_gap)`
- `__init__` (line 261) `def __init__(self, device, feature_extractor)`
- `_preprocess_batch` (line 268) `def _preprocess_batch(self, x)`
- `get_curriculum_dataset` (line 272) `def get_curriculum_dataset(self, cycle)`
- `train_model` (line 278) `def train_model(self, model, cycle, is_baseline)`

#### `plank12.py`
**Path:** `plank12.py`

**Classes:**
- `TokenMixer` (line 35) `class TokenMixer` - *Mezcla tokens entre sí (eje T).*
- `PatchFeatureExtractor` (line 51) `class PatchFeatureExtractor`
- `LotteryMLP` (line 87) `class LotteryMLP`
- `StandardBaseline` (line 116) `class StandardBaseline`
- `SpectralMonitor` (line 127) `class SpectralMonitor`
- `OrthogonalEvolutionEngine` (line 154) `class OrthogonalEvolutionEngine`
- `OrthogonalTrainer` (line 248) `class OrthogonalTrainer`

**Functions:**
- `main` (line 363) `def main()`
- `__init__` (line 37) `def __init__(self, num_tokens)`
- `forward` (line 44) `def forward(self, x)`
- `__init__` (line 52) `def __init__(self, img_size, patch_size, in_chans, embed_dim)`
- `freeze` (line 66) `def freeze(self)`
- `unfreeze` (line 70) `def unfreeze(self)`
- `forward` (line 74) `def forward(self, x)`
- `__init__` (line 88) `def __init__(self, input_dim, hidden_dim, num_classes)`
- `apply_masks` (line 99) `def apply_masks(self)`
- `get_sparsity` (line 104) `def get_sparsity(self)`
- `forward` (line 109) `def forward(self, x)`
- `__init__` (line 117) `def __init__(self, input_dim, hidden_dim)`
- `forward` (line 121) `def forward(self, x)`
- `compute_metrics` (line 128) `def compute_metrics(self, weight)` - *Returns: (L, Rank_Efficient, S_vN)
Used for logging and decision making (NOT for backprop).*
- `__init__` (line 155) `def __init__(self, device)`
- `_gradient_nudge_inheritance` (line 160) `def _gradient_nudge_inheritance(self, child_model, elk_state, data_loader, feature_extractor, nudge_lr)`
- `_apply_rank_capping_shock` (line 192) `def _apply_rank_capping_shock(self, model, layer_name, keep_ratio)` - *Minimalistic Shock: Zero out weak singular values without renormalizing.
Force energy decay and minimality.*
- `create_refined_offspring` (line 224) `def create_refined_offspring(self, elk_state, data_loader, feature_extractor)` - *Crea un hijo de las MISMAS dimensiones (Fixed Width).
Evoluciona mediante Nudge (Aprendizaje) + Shock (Poda).*
- `__init__` (line 249) `def __init__(self, device, feature_extractor)`
- `_preprocess_batch` (line 256) `def _preprocess_batch(self, x)`
- `get_curriculum_dataset` (line 260) `def get_curriculum_dataset(self, cycle)`
- `train_model` (line 266) `def train_model(self, model, cycle, is_baseline)`

#### `plank13.py`
**Path:** `plank13.py`

**Classes:**
- `TokenMixer` (line 32) `class TokenMixer` - *Mezcla tokens entre sí (Solo para Apex).*
- `PatchFeatureExtractor` (line 47) `class PatchFeatureExtractor` - *Extractor configurable.
use_mixer=True -> Apex (Syntactic)
use_mixer=False -> Blind Structural Baseline*
- `LotteryMLP` (line 92) `class LotteryMLP`
- `SpectralMonitor` (line 123) `class SpectralMonitor`
- `OrthogonalEvolutionEngine` (line 149) `class OrthogonalEvolutionEngine`
- `DualTrainer` (line 224) `class DualTrainer`

**Functions:**
- `main` (line 329) `def main()`
- `__init__` (line 34) `def __init__(self, num_tokens)`
- `forward` (line 41) `def forward(self, x)`
- `__init__` (line 53) `def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)`
- `freeze` (line 70) `def freeze(self)`
- `unfreeze` (line 74) `def unfreeze(self)`
- `forward` (line 78) `def forward(self, x)`
- `__init__` (line 93) `def __init__(self, input_dim, hidden_dim, num_classes)`
- `apply_masks` (line 104) `def apply_masks(self)`
- `get_sparsity` (line 109) `def get_sparsity(self)`
- `forward` (line 114) `def forward(self, x)`
- `compute_metrics` (line 124) `def compute_metrics(self, weight)` - *Returns: (L, Rank_Efficient, S_vN)*
- `__init__` (line 150) `def __init__(self, device)`
- `_gradient_nudge_inheritance` (line 155) `def _gradient_nudge_inheritance(self, child_model, elk_state, data_loader, feature_extractor, nudge_lr)`
- `_apply_rank_capping_shock` (line 187) `def _apply_rank_capping_shock(self, model, layer_name, keep_ratio)`
- `create_refined_offspring` (line 207) `def create_refined_offspring(self, elk_state, data_loader, feature_extractor)`
- `__init__` (line 225) `def __init__(self, device, extractor_apex, extractor_blind)`
- `_preprocess_batch` (line 233) `def _preprocess_batch(self, x, extractor)`
- `get_curriculum_dataset` (line 237) `def get_curriculum_dataset(self, cycle)`
- `train_single_chain` (line 243) `def train_single_chain(self, model, cycle, chain_type)` - *Entrena una cadena específica (Apex o Blind).*

#### `plank2.py`
**Path:** `plank2.py`

**Classes:**
- `SpectralMonitor` (line 41) `class SpectralMonitor` - *Computes L = 1 / (|S_vN - log(rank_eff + 1)| + ε)
Used purely as a diagnostic—never to modify training.*
- `PersistentPruner` (line 82) `class PersistentPruner` - *Applies and ENFORCES magnitude-based pruning across training.
Unlike transient pruning, this modifies the parameter mask permanently.*
- `SpectralMLP` (line 117) `class SpectralMLP` - *Small MLP (1504 params) for clean spectral analysis.*

**Functions:**
- `train_condition` (line 173) `def train_condition(condition_name, config, device, seed)` - *Train one condition and return full log as DataFrame.*
- `main` (line 281) `def main()`
- `__init__` (line 46) `def __init__(self, epsilon_c)`
- `compute_L` (line 49) `def compute_L(self, weight)` - *Returns: (L, S_vN, rank_eff, regime)*
- `__init__` (line 87) `def __init__(self, sparsity_target)`
- `apply_to_model` (line 91) `def apply_to_model(self, model)` - *Apply pruning mask and register backward hook to zero gradients.*
- `enforce_during_training` (line 105) `def enforce_during_training(self, model)` - *Call this after every optimizer.step()*
- `__init__` (line 119) `def __init__(self, input_dim, hidden_dim, num_classes)`
- `reduce_input` (line 128) `def reduce_input(self, x)` - *Reduce CIFAR-10 (32x32x3) to 32D for focus*
- `forward` (line 135) `def forward(self, x)`

#### `plank3.py`
**Path:** `plank3.py`

**Classes:**
- `SpectralMonitor` (line 34) `class SpectralMonitor`
- `PersistentPruner` (line 62) `class PersistentPruner`
- `SpectralMLP` (line 87) `class SpectralMLP`

**Functions:**
- `train_dense_to_target` (line 110) `def train_dense_to_target(device, target_acc)` - *Train dense model until it reaches target accuracy.*
- `progressive_pruning_search` (line 188) `def progressive_pruning_search(model, device, target_acc)` - *Progressively prune model and find critical density threshold.*
- `find_critical_threshold` (line 264) `def find_critical_threshold(pruning_df, target_acc)` - *Find the minimum density where accuracy >= target_acc.*
- `main` (line 294) `def main()`
- `__init__` (line 35) `def __init__(self, epsilon_c)`
- `compute_L` (line 38) `def compute_L(self, weight)`
- `__init__` (line 63) `def __init__(self, sparsity_target)`
- `apply_to_model` (line 67) `def apply_to_model(self, model)`
- `enforce_during_training` (line 77) `def enforce_during_training(self, model)`
- `__init__` (line 88) `def __init__(self, input_dim, hidden_dim, num_classes)`
- `reduce_input` (line 96) `def reduce_input(self, x)`
- `forward` (line 102) `def forward(self, x)`

#### `plank4.py`
**Path:** `plank4.py`

**Classes:**
- `SpectralMonitor` (line 41) `class SpectralMonitor`
- `PersistentPruner` (line 66) `class PersistentPruner`
- `SpectralMLP` (line 88) `class SpectralMLP`
- `FractalSovereigntyEngine` (line 111) `class FractalSovereigntyEngine`

**Functions:**
- `main` (line 329) `def main()`
- `__init__` (line 42) `def __init__(self, epsilon_c)`
- `compute_L` (line 45) `def compute_L(self, weight)`
- `__init__` (line 67) `def __init__(self, sparsity_target)`
- `apply_to_model` (line 71) `def apply_to_model(self, model)`
- `enforce_during_training` (line 81) `def enforce_during_training(self, model)`
- `__init__` (line 89) `def __init__(self, input_dim, hidden_dim, num_classes)`
- `reduce_input` (line 97) `def reduce_input(self, x)`
- `forward` (line 103) `def forward(self, x)`
- `__init__` (line 112) `def __init__(self, device, base_target_acc)`
- `train_dense_model` (line 123) `def train_dense_model(self, hidden_dim, target_acc)`
- `extract_seed_weights` (line 168) `def extract_seed_weights(self, model)`
- `inoculate_seed` (line 174) `def inoculate_seed(self, large_model, seed_weights)` - *Embed seed into larger architecture*
- `progressive_pruning` (line 192) `def progressive_pruning(self, model, target_acc)`
- `execute_cycle` (line 236) `def execute_cycle(self, cycle, base_hidden_dim, expansion_factor)`
- `run_experiment` (line 281) `def run_experiment(self, num_cycles)`

#### `plank5.py`
**Path:** `plank5.py`

**Classes:**
- `SpectralMonitor` (line 30) `class SpectralMonitor`
- `PersistentPruner` (line 50) `class PersistentPruner`
- `SpectralMLP` (line 65) `class SpectralMLP`
- `GrokkingDetector` (line 88) `class GrokkingDetector`
- `SyntheticBlackSwanGenerator` (line 121) `class SyntheticBlackSwanGenerator`
- `EvolutionCycle` (line 211) `class EvolutionCycle`
- `EvolutionaryBlackSwanChain` (line 392) `class EvolutionaryBlackSwanChain`

**Functions:**
- `main` (line 556) `def main()`
- `__init__` (line 31) `def __init__(self, epsilon_c)`
- `compute_L` (line 34) `def compute_L(self, weight)`
- `__init__` (line 51) `def __init__(self, sparsity_target)`
- `apply_to_model` (line 55) `def apply_to_model(self, model)`
- `__init__` (line 66) `def __init__(self, input_dim, hidden_dim, num_classes)`
- `reduce_input` (line 74) `def reduce_input(self, x)`
- `forward` (line 80) `def forward(self, x)`
- `__init__` (line 89) `def __init__(self, patience, gap_threshold)`
- `update` (line 94) `def update(self, train_acc, test_acc, epoch)`
- `detect_grokking` (line 103) `def detect_grokking(self)`
- `__init__` (line 122) `def __init__(self, device, target_acc, min_L)`
- `generate` (line 128) `def generate(self, hidden_dim)` - *Genera cisne negro sintético si no existe legacy*
- `__init__` (line 212) `def __init__(self, device, base_acc)`
- `inoculate_dna` (line 218) `def inoculate_dna(self, large_model, seed_weights, noise_scale)` - *Inocula ADN del cisne anterior con mutación controlada*
- `train_with_grokking` (line 243) `def train_with_grokking(self, model, seed_model, target_acc)` - *Entrena modelo induciendo grokking y monitoreando transición de fase*
- `distill_sparse_model` (line 337) `def distill_sparse_model(self, model, target_acc)` - *Pruning progresivo para extraer nuevo cisne negro*
- `__init__` (line 393) `def __init__(self, device, num_cycles, base_acc)`
- `load_legacy_or_generate_seed` (line 402) `def load_legacy_or_generate_seed(self)` - *Carga legacy seed o genera uno sintético*
- `run_evolutionary_chain` (line 419) `def run_evolutionary_chain(self)` - *Ejecuta la cadena evolutiva completa*
- `save_chain_results` (line 501) `def save_chain_results(self)` - *Guarda resultados completos de la cadena evolutiva*
- `print_evolution_summary` (line 519) `def print_evolution_summary(self)` - *Imprime resumen ejecutivo de la cadena evolutiva*

#### `plank6.py`
**Path:** `plank6.py`

**Classes:**
- `SpectralMonitor` (line 29) `class SpectralMonitor` - *Calcula L (Coherencia Espectral) y Rank Efectivo*
- `SpectralMLP` (line 51) `class SpectralMLP` - *Red Neuronal Base para el experimento*
- `GrokkingDetector` (line 78) `class GrokkingDetector`
- `GuidedElkHuntingEngine` (line 101) `class GuidedElkHuntingEngine` - *Motor que toma el mejor modelo anterior (Elk), 
muta sus pesos guiadamente y expande la arquitectura.*
- `TrainingCycle` (line 199) `class TrainingCycle`

**Functions:**
- `main` (line 293) `def main()`
- `__init__` (line 31) `def __init__(self, epsilon_c)`
- `compute_L` (line 34) `def compute_L(self, weight)`
- `__init__` (line 53) `def __init__(self, input_dim, hidden_dim, num_classes)`
- `reduce_input` (line 63) `def reduce_input(self, x)`
- `forward` (line 70) `def forward(self, x)`
- `__init__` (line 79) `def __init__(self, patience, gap_threshold)`
- `update` (line 84) `def update(self, train_acc, test_acc, epoch)`
- `detect_grokking` (line 87) `def detect_grokking(self)`
- `__init__` (line 106) `def __init__(self, device)`
- `_guided_elk_mutation` (line 110) `def _guided_elk_mutation(self, old_weight, target_shape, noise_scale, refinement_steps)` - *Evoluciona los pesos del Elk a una dimensión mayor manteniendo coherencia.*
- `_apply_spectral_refinement` (line 146) `def _apply_spectral_refinement(self, W)` - *Filtra componentes de baja energía y reconstruye*
- `create_offspring_from_elk` (line 158) `def create_offspring_from_elk(self, elk_state, new_hidden_dim, generation)` - *Crea un nuevo modelo (Cisne Negro) basado en el Elk (mejor modelo previo).*
- `__init__` (line 200) `def __init__(self, device)`
- `train_phase` (line 204) `def train_phase(self, model, cycle_id)`

#### `plank7.py`
**Path:** `plank7.py`

**Classes:**
- `SpectralMonitor` (line 29) `class SpectralMonitor`
- `SpectralMLP` (line 45) `class SpectralMLP`
- `AdvancedEvolutionEngine` (line 68) `class AdvancedEvolutionEngine` - *Implementa Gradient Nudging y Espectral Shock.*
- `CurriculumTrainingCycle` (line 177) `class CurriculumTrainingCycle`

**Functions:**
- `main` (line 283) `def main()`
- `__init__` (line 30) `def __init__(self, epsilon_c)`
- `compute_L` (line 33) `def compute_L(self, weight)`
- `__init__` (line 46) `def __init__(self, input_dim, hidden_dim, num_classes)`
- `reduce_input` (line 54) `def reduce_input(self, x)`
- `forward` (line 60) `def forward(self, x)`
- `__init__` (line 72) `def __init__(self, device)`
- `_gradient_nudge_inheritance` (line 76) `def _gradient_nudge_inheritance(self, child_model, elk_state, data_loader, nudge_lr)` - *Antes de entrenar, hacemos 1 paso de gradiente del Elk sobre los nuevos datos.
Esto 'pre-ajusta' el ADN al contexto actual.*
- `_apply_spectral_shock` (line 107) `def _apply_spectral_shock(self, W, shock_intensity)` - *Aplica una perturbación no-lineal a los valores singulares.
Esto rompe mínimos locales planos sin destruir la estructura global.*
- `create_advanced_offspring` (line 124) `def create_advanced_offspring(self, elk_state, new_hidden_dim, cycle, data_loader)` - *Crea un hijo combinando:
1. Herencia de pesos
2. Gradient Nudge (context awareness)
3. Spectral Shock (ruptura de estancamiento)*
- `__init__` (line 178) `def __init__(self, device)`
- `get_curriculum_dataset` (line 186) `def get_curriculum_dataset(self, cycle)` - *Estrategia de Curriculum:
Ciclos 1-3: Subset pequeño (Foco en estructura).
Ciclos 4+: Expansión progresiva (Foco en generalización).*
- `train_phase` (line 204) `def train_phase(self, model, cycle)`

#### `plank8.py`
**Path:** `plank8.py`

**Classes:**
- `PatchFeatureExtractor` (line 34) `class PatchFeatureExtractor` - *Extrae características mediante Patch Embedding.
Convierte imagen (B, 3, 32, 32) en secuencia de parches proyectados.*
- `LotteryMLP` (line 78) `class LotteryMLP`
- `StandardBaseline` (line 109) `class StandardBaseline` - *Baseline moderno (Patch + MLP simple) sin evolución.*
- `SpectralMonitor` (line 121) `class SpectralMonitor`
- `ApexEvolutionEngine` (line 136) `class ApexEvolutionEngine`
- `ApexTrainer` (line 235) `class ApexTrainer`

**Functions:**
- `main` (line 342) `def main()`
- `__init__` (line 39) `def __init__(self, img_size, patch_size, in_chans, embed_dim)`
- `freeze` (line 57) `def freeze(self)`
- `unfreeze` (line 61) `def unfreeze(self)`
- `forward` (line 65) `def forward(self, x)`
- `__init__` (line 79) `def __init__(self, input_dim, hidden_dim, num_classes)`
- `apply_masks` (line 92) `def apply_masks(self)`
- `get_sparsity` (line 97) `def get_sparsity(self)`
- `forward` (line 102) `def forward(self, x)`
- `__init__` (line 111) `def __init__(self, input_dim, hidden_dim)`
- `forward` (line 115) `def forward(self, x)`
- `compute_L` (line 122) `def compute_L(self, weight)`
- `__init__` (line 137) `def __init__(self, device)`
- `_gradient_nudge_inheritance` (line 142) `def _gradient_nudge_inheritance(self, child_model, elk_state, data_loader, feature_extractor, nudge_lr)`
- `_apply_dynamic_spectral_shock` (line 176) `def _apply_dynamic_spectral_shock(self, model, layer_name)`
- `create_apex_offspring` (line 199) `def create_apex_offspring(self, elk_state, new_hidden_dim, cycle, data_loader, feature_extractor, parent_gap)`
- `__init__` (line 236) `def __init__(self, device, feature_extractor)`
- `_preprocess_batch` (line 243) `def _preprocess_batch(self, x)`
- `get_curriculum_dataset` (line 247) `def get_curriculum_dataset(self, cycle)`
- `train_model` (line 253) `def train_model(self, model, cycle, is_baseline)`

#### `resmav2_1.py`
**Path:** `resmav2_1.py`

**Classes:**
- `OptimizedE8Layer` (line 23) `class OptimizedE8Layer` - *Optimized E8 with caching and efficiency improvements*
- `RESMAv2Fast` (line 48) `class RESMAv2Fast` - *Fast version: E8 + GAT fusion with minimal overhead*
- `RESMAv2Standard` (line 86) `class RESMAv2Standard` - *Standard version: 2 layers of E8 + GAT fusion*
- `RESMAv2Deep` (line 134) `class RESMAv2Deep` - *Deeper version with 3 layers*
- `GAT_Baseline` (line 182) `class GAT_Baseline` - *Optimized GAT baseline*

**Functions:**
- `load_elliptic_data` (line 207) `def load_elliptic_data()`
- `train_and_evaluate` (line 264) `def train_and_evaluate(model, X, y, edge_index, train_idx, val_idx, epochs, lr, name, fold)`
- `cross_validate_model` (line 321) `def cross_validate_model(model_class, X, y, edge_index, num_nodes, n_splits, seed, name)`
- `__init__` (line 25) `def __init__(self, in_features, out_features, edge_index, num_nodes)`
- `forward` (line 40) `def forward(self, x)`
- `__init__` (line 50) `def __init__(self, input_dim, hidden_dim, edge_index, num_nodes, dropout)`
- `forward` (line 73) `def forward(self, x, edge_index)`
- `__init__` (line 88) `def __init__(self, input_dim, hidden_dim, edge_index, num_nodes, dropout)`
- `forward` (line 113) `def forward(self, x, edge_index)`
- `__init__` (line 136) `def __init__(self, input_dim, hidden_dim, edge_index, num_nodes, dropout)`
- `forward` (line 168) `def forward(self, x, edge_index)`
- `__init__` (line 184) `def __init__(self, input_dim, hidden_dim, dropout)`
- `forward` (line 194) `def forward(self, x, edge_index)`

### SH (1 files)

#### `install.sh`
**Path:** `install.sh`

*No symbols extracted*
