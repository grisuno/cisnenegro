# Polyglot Codebase Knowledge Graph

> Generated offline by **readmenator**. 37 files, 1059 symbols, 418 imports. Supports C, C++, Python, Go, Rust, JS/TS, Java, C#, Shell, PHP, Dart, GDScript, Nim, ASM, Ruby, Swift, Kotlin, Scala, Lua, Elixir.
> No LLMs. No tokens. Pure static analysis. See more [here](https://github.com/grisuno/ReadMenator)

**Start here:** Statistics Dashboard for scope, God Nodes for blast radius, Architecture Reference for per-file API. Agents: prefer `readmenator-agent/INDEX.md` + `SYMBOLS.md`.

**Wiki:** prefer `readmenator-wiki/index.md` for progressive disclosure: one synthesis page per community, `connections.json` with EXTRACTED vs INFERRED confidence, `queries.md` log, `REPORT.md` audit.

**Confidence:** EXTRACTED = parsed from source, INFERRED = heuristic bridge, AMBIGUOUS = reported, never hidden. See `readmenator-wiki/REPORT.md`.

**Total Files Parsed:** 37 | **Total Symbols Extracted:** 1059 | **Total Imports:** 418

<!-- ranking_model: v1.0 | weights: {ppr:0.45,auth:0.2,test:0.15,doc:0.1,fresh:0.1} | alpha:0.85 | commit:05a4468 | date:2026-07-18 -->


## Table of Contents

1. [Statistics Dashboard](#statistics-dashboard)
2. [Architectural Layers](#architectural-layers)
3. [Ranked Context](#ranked-context)
4. [God Nodes](#god-nodes)
5. [Suggested Questions](#suggested-questions)
6. [Hotspot Analysis](#hotspot-analysis)
7. [Change Impact Analysis](#change-impact-analysis)
8. [Suggested Linting Rules](#suggested-linting-rules)
9. [Orphans](#orphans)
10. [Query Recipes](#query-recipes)
11. [Structural Knowledge Map](#structural-knowledge-map)
12. [UML Class Diagram](#uml-class-diagram)
13. [Code Property Graph](#code-property-graph)
14. [Architecture Reference](#architecture-reference)
    - [PY (36 files)](#py-36-files)
    - [SH (1 files)](#sh-1-files)

---

## Statistics Dashboard

| Metric | Value |
|--------|-------|
| Total Files | 37 |
| Total Symbols | 1059 |
| Total Imports | 418 |
| Call Edges | 11494 |
| Inheritance Edges | 116 |
| Languages | 2 |
| Avg Symbols/File | 28.6 |
| Avg Imports/File | 11.3 |

### Top Files by Import Count (Fan-Out)

| File | Imports | Symbols | Language |
|------|---------|---------|----------|
| `apex27.py` | 16 | 40 | py |
| `resmav2_1.py` | 16 | 18 | py |
| `apex26.py` | 15 | 42 | py |
| `apex28.py` | 15 | 47 | py |
| `apex29.py` | 15 | 47 | py |
| `apex30.py` | 15 | 47 | py |
| `apex31.py` | 15 | 47 | py |
| `apex33.py` | 14 | 46 | py |
| `apex34.py` | 14 | 45 | py |
| `apex35.py` | 14 | 29 | py |

---

## Architectural Layers

Auto-detected from path patterns, naming conventions, and imported frameworks.

| Layer | Files |
|-------|-------|
| utility | 37 |

### utility

- `apex14.py` (py, 26 symbols)
- `apex15.py` (py, 22 symbols)
- `apex16.py` (py, 22 symbols)
- `apex17.py` (py, 26 symbols)
- `apex18.py` (py, 26 symbols)
- `apex19.py` (py, 27 symbols)
- `apex20.py` (py, 28 symbols)
- `apex21.py` (py, 32 symbols)
- `apex22.py` (py, 31 symbols)
- `apex23.py` (py, 32 symbols)
- `apex24.py` (py, 32 symbols)
- `apex25.py` (py, 32 symbols)
- `apex26.py` (py, 42 symbols)
- `apex27.py` (py, 40 symbols)
- `apex28.py` (py, 47 symbols)
- *... and 22 more*

---

## Ranked Context

Files ranked by composite score for the current query context. The ranking combines Personalized PageRank (query relevance), global authority, test coverage, documentation coverage, and code freshness. Model: v1.0.

| Rank | File | Composite | PPR | Authority | Test | Doc |
|------|------|-----------|-----|-----------|------|-----|
| 1 | `apex28.py` | 0.0809 | 0.0000 | 0.0000 | 0.00 | 0.81 |
| 2 | `apex29.py` | 0.0809 | 0.0000 | 0.0000 | 0.00 | 0.81 |
| 3 | `apex26.py` | 0.0786 | 0.0000 | 0.0000 | 0.00 | 0.79 |
| 4 | `plank2.py` | 0.0692 | 0.0000 | 0.0000 | 0.00 | 0.69 |
| 5 | `plank7.py` | 0.0353 | 0.0000 | 0.0000 | 0.00 | 0.35 |
| 6 | `plank6.py` | 0.0350 | 0.0000 | 0.0000 | 0.00 | 0.35 |
| 7 | `apex27.py` | 0.0325 | 0.0000 | 0.0000 | 0.00 | 0.33 |
| 8 | `plank5.py` | 0.0310 | 0.0000 | 0.0000 | 0.00 | 0.31 |
| 9 | `app.py` | 0.0304 | 0.0000 | 0.0000 | 0.00 | 0.30 |
| 10 | `apex30.py` | 0.0298 | 0.0000 | 0.0000 | 0.00 | 0.30 |

---

## God Nodes

Most architecturally central files ranked by combined import/export degree and symbol richness.

| File | Score | Connections | PageRank |
|------|-------|-------------|----------|
| `apex28.py` | 4.7 | | 0.0000 |
| `apex29.py` | 4.7 | | 0.0000 |
| `apex30.py` | 4.7 | | 0.0000 |
| `apex31.py` | 4.7 | | 0.0000 |
| `apex33.py` | 4.6 | | 0.0000 |
| `apex34.py` | 4.5 | | 0.0000 |
| `apex26.py` | 4.2 | | 0.0000 |
| `apex27.py` | 4.0 | | 0.0000 |
| `apex21.py` | 3.2 | | 0.0000 |
| `apex23.py` | 3.2 | | 0.0000 |

---

## Suggested Questions

Auto-generated exploration prompts based on graph structure:

- What does apex28.py depend on, and what depends on it? (0 connections)
- What does apex29.py depend on, and what depends on it? (0 connections)
- What does apex30.py depend on, and what depends on it? (0 connections)
- What is GatedTokenMixer in apex14.py and how is it used?
- What is GatedTokenMixer in apex15.py and how is it used?

---

## Hotspot Analysis

Files ranked by combined complexity (symbol count) and centrality (connection count). High-scoring files are architecturally critical and may need refactoring attention.

| File | Complexity | Centrality | Combined | Symbols | Connections |
|------|-----------|------------|----------|---------|-------------|
| `apex28.py` | 1.000 | 0.938 | 0.963 | 47 | 15 |
| `apex29.py` | 1.000 | 0.938 | 0.963 | 47 | 15 |
| `apex26.py` | 0.894 | 0.938 | 0.920 | 42 | 15 |
| `plank2.py` | 0.277 | 0.688 | 0.523 | 13 | 11 |
| `plank7.py` | 0.362 | 0.688 | 0.557 | 17 | 11 |
| `plank6.py` | 0.425 | 0.688 | 0.583 | 20 | 11 |
| `apex27.py` | 0.851 | 1.000 | 0.940 | 40 | 16 |
| `plank5.py` | 0.617 | 0.812 | 0.734 | 29 | 13 |
| `app.py` | 0.489 | 0.812 | 0.683 | 23 | 13 |
| `apex30.py` | 1.000 | 0.938 | 0.963 | 47 | 15 |
| `apex31.py` | 1.000 | 0.938 | 0.963 | 47 | 15 |
| `apex33.py` | 0.979 | 0.875 | 0.916 | 46 | 14 |
| `apex34.py` | 0.957 | 0.875 | 0.908 | 45 | 14 |
| `apex35.py` | 0.617 | 0.875 | 0.772 | 29 | 14 |
| `resmav2_1.py` | 0.383 | 1.000 | 0.753 | 18 | 16 |

---

## Change Impact Analysis

Files sorted by how many other files would be affected if they changed. High-impact files should be changed with caution.

| File | Direct Dependents | Transitive Dependents | Total Impact |
|------|------------------|----------------------|--------------|
| `apex14.py` | 0 | 0 | 0 |
| `apex15.py` | 0 | 0 | 0 |
| `apex16.py` | 0 | 0 | 0 |
| `apex17.py` | 0 | 0 | 0 |
| `apex18.py` | 0 | 0 | 0 |
| `apex19.py` | 0 | 0 | 0 |
| `apex20.py` | 0 | 0 | 0 |
| `apex21.py` | 0 | 0 | 0 |
| `apex22.py` | 0 | 0 | 0 |
| `apex23.py` | 0 | 0 | 0 |
| `apex24.py` | 0 | 0 | 0 |
| `apex25.py` | 0 | 0 | 0 |
| `apex26.py` | 0 | 0 | 0 |
| `apex27.py` | 0 | 0 | 0 |
| `apex28.py` | 0 | 0 | 0 |

---

## Suggested Linting Rules

Automatically suggested linting and security rules based on patterns detected in the codebase. These can be exported as Semgrep rules using the `--export-rules` flag.

| Rule ID | Severity | Description | Language | Matches |
|---------|----------|-------------|----------|---------|
| `RM002` | warning | Bare except clause catches all exceptions including SystemExit | python | 54 |
| `RM001` | info | Large number of functions in py: 836 total | py | 836 |
| `RM003` | info | Print statement found (consider logging instead) | python | 1522 |

---

## Orphans

Files with no documentation or low connectivity. These are candidates for documentation investment or cleanup.

- `install.sh` (0 symbols, no doc)

---

## Query Recipes

Example queries you can run against this knowledge base using the ranking engine:

```
# Find files most relevant to a concept
readmenator query "Where is the import resolver implemented?"

# Rank files by relevance to a topic
readmenator query "How does documentation generation work?"

# Explain why a file ranks highly
readmenator query "explain readmenator/_documentation.py"

# Trace dependency paths with ranked context
readmenator query "path from CLI to exporter"
```

The ranking model uses the following signals:

- **Personalized PageRank** (45% weight): query-specific relevance via seed propagation
- **Global Authority** (20% weight): structural importance via standard PageRank
- **Test Coverage** (15% weight): fraction of symbols referenced in test files
- **Doc Coverage** (10% weight): presence of docstrings and file-level docs
- **Freshness** (10% weight): recent modification activity

Results include score decomposition and justification paths for each ranked item.

---

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
    apex28_py["apex28.py (py)"]
    class apex28_py mod;
    apex29_py["apex29.py (py)"]
    class apex29_py mod;
    apex30_py["apex30.py (py)"]
    class apex30_py mod;
    apex31_py["apex31.py (py)"]
    class apex31_py mod;
    apex26_py["apex26.py (py)"]
    class apex26_py mod;
    apex33_py["apex33.py (py)"]
    class apex33_py mod;
    apex34_py["apex34.py (py)"]
    class apex34_py mod;
    apex35_py["apex35.py (py)"]
    class apex35_py mod;
    plank5_py["plank5.py (py)"]
    class plank5_py mod;
    app_py["app.py (py)"]
    class app_py mod;
    plank4_py["plank4.py (py)"]
    class plank4_py mod;
    plank6_py["plank6.py (py)"]
    class plank6_py mod;
    plank7_py["plank7.py (py)"]
    class plank7_py mod;
    plank3_py["plank3.py (py)"]
    class plank3_py mod;
    plank2_py["plank2.py (py)"]
    class plank2_py mod;
    apex21_py["apex21.py (py)"]
    class apex21_py mod;
    apex23_py["apex23.py (py)"]
    class apex23_py mod;
    apex24_py["apex24.py (py)"]
    class apex24_py mod;
    apex25_py["apex25.py (py)"]
    class apex25_py mod;
    apex22_py["apex22.py (py)"]
    class apex22_py mod;
    plank11_py["plank11.py (py)"]
    class plank11_py mod;
    plank12_py["plank12.py (py)"]
    class plank12_py mod;
    apex20_py["apex20.py (py)"]
    class apex20_py mod;
    apex32_py["apex32.py (py)"]
    class apex32_py mod;
    apex19_py["apex19.py (py)"]
    class apex19_py mod;
    apex14_py["apex14.py (py)"]
    class apex14_py mod;
    apex17_py["apex17.py (py)"]
    class apex17_py mod;
    apex18_py["apex18.py (py)"]
    class apex18_py mod;
    plank10_py["plank10.py (py)"]
    class plank10_py mod;
    plank13_py["plank13.py (py)"]
    class plank13_py mod;
    plank8_py["plank8.py (py)"]
    class plank8_py mod;
    apex15_py["apex15.py (py)"]
    class apex15_py mod;
    apex16_py["apex16.py (py)"]
    class apex16_py mod;
    plank_py["plank.py (py)"]
    class plank_py mod;
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

## UML Class Diagram

Auto-generated Mermaid class diagram from parsed class-level symbols. Shows classes, structs, interfaces, traits, and their methods with inheritance and dependency relationships.

```mermaid
classDiagram
  class apex14_py_GatedTokenMixer {
    <<class>>
    +main()
    +__init__(self, num_tokens, embed_dim)
    +forward(self, x)
    +__init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)
    +freeze(self)
    +unfreeze(self)
    +forward(self, x)
    +__init__(self, input_dim, hidden_dim, num_classes)
    +apply_masks(self)
    +get_sparsity(self)
  }
  class apex14_py_PatchFeatureExtractor {
    <<class>>
    +main()
    +__init__(self, num_tokens, embed_dim)
    +forward(self, x)
    +__init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)
    +freeze(self)
    +unfreeze(self)
    +forward(self, x)
    +__init__(self, input_dim, hidden_dim, num_classes)
    +apply_masks(self)
    +get_sparsity(self)
  }
  class apex14_py_LotteryMLP {
    <<class>>
    +main()
    +__init__(self, num_tokens, embed_dim)
    +forward(self, x)
    +__init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)
    +freeze(self)
    +unfreeze(self)
    +forward(self, x)
    +__init__(self, input_dim, hidden_dim, num_classes)
    +apply_masks(self)
    +get_sparsity(self)
  }
  class apex14_py_SpectralMonitor {
    <<class>>
    +main()
    +__init__(self, num_tokens, embed_dim)
    +forward(self, x)
    +__init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)
    +freeze(self)
    +unfreeze(self)
    +forward(self, x)
    +__init__(self, input_dim, hidden_dim, num_classes)
    +apply_masks(self)
    +get_sparsity(self)
  }
  class apex14_py_OrthogonalEvolutionEngine {
    <<class>>
    +main()
    +__init__(self, num_tokens, embed_dim)
    +forward(self, x)
    +__init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)
    +freeze(self)
    +unfreeze(self)
    +forward(self, x)
    +__init__(self, input_dim, hidden_dim, num_classes)
    +apply_masks(self)
    +get_sparsity(self)
  }
  class apex14_py_HierarchicalTrainer {
    <<class>>
    +main()
    +__init__(self, num_tokens, embed_dim)
    +forward(self, x)
    +__init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)
    +freeze(self)
    +unfreeze(self)
    +forward(self, x)
    +__init__(self, input_dim, hidden_dim, num_classes)
    +apply_masks(self)
    +get_sparsity(self)
  }
  class apex15_py_GatedTokenMixer {
    <<class>>
    +compute_spectral_loss(W, target_rank_factor)
    +main()
    +__init__(self, num_patches, embed_dim)
    +forward(self, x)
    +__init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)
    +freeze(self)
    +unfreeze_mixer_only(self)
    +forward(self, x)
    +__init__(self, input_dim, hidden_dim, num_classes)
    +apply_masks(self)
  }
  class apex15_py_PatchFeatureExtractor {
    <<class>>
    +compute_spectral_loss(W, target_rank_factor)
    +main()
    +__init__(self, num_patches, embed_dim)
    +forward(self, x)
    +__init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)
    +freeze(self)
    +unfreeze_mixer_only(self)
    +forward(self, x)
    +__init__(self, input_dim, hidden_dim, num_classes)
    +apply_masks(self)
  }
  class apex15_py_TaxonomicMLP {
    <<class>>
    +compute_spectral_loss(W, target_rank_factor)
    +main()
    +__init__(self, num_patches, embed_dim)
    +forward(self, x)
    +__init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)
    +freeze(self)
    +unfreeze_mixer_only(self)
    +forward(self, x)
    +__init__(self, input_dim, hidden_dim, num_classes)
    +apply_masks(self)
  }
  class apex15_py_SpectralMonitor {
    <<class>>
    +compute_spectral_loss(W, target_rank_factor)
    +main()
    +__init__(self, num_patches, embed_dim)
    +forward(self, x)
    +__init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)
    +freeze(self)
    +unfreeze_mixer_only(self)
    +forward(self, x)
    +__init__(self, input_dim, hidden_dim, num_classes)
    +apply_masks(self)
  }
  class apex15_py_TaxonomicTrainer {
    <<class>>
    +compute_spectral_loss(W, target_rank_factor)
    +main()
    +__init__(self, num_patches, embed_dim)
    +forward(self, x)
    +__init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)
    +freeze(self)
    +unfreeze_mixer_only(self)
    +forward(self, x)
    +__init__(self, input_dim, hidden_dim, num_classes)
    +apply_masks(self)
  }
  class apex16_py_GatedTokenMixer {
    <<class>>
    +compute_spectral_loss(W, target_rank_factor)
    +main()
    +__init__(self, num_patches, embed_dim)
    +forward(self, x)
    +__init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)
    +freeze(self)
    +unfreeze_mixer_only(self)
    +forward(self, x)
    +__init__(self, input_dim, hidden_dim, num_classes)
    +apply_masks(self)
  }
  class apex16_py_PatchFeatureExtractor {
    <<class>>
    +compute_spectral_loss(W, target_rank_factor)
    +main()
    +__init__(self, num_patches, embed_dim)
    +forward(self, x)
    +__init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)
    +freeze(self)
    +unfreeze_mixer_only(self)
    +forward(self, x)
    +__init__(self, input_dim, hidden_dim, num_classes)
    +apply_masks(self)
  }
  class apex16_py_TaxonomicMLP {
    <<class>>
    +compute_spectral_loss(W, target_rank_factor)
    +main()
    +__init__(self, num_patches, embed_dim)
    +forward(self, x)
    +__init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)
    +freeze(self)
    +unfreeze_mixer_only(self)
    +forward(self, x)
    +__init__(self, input_dim, hidden_dim, num_classes)
    +apply_masks(self)
  }
  class apex16_py_SpectralMonitor {
    <<class>>
    +compute_spectral_loss(W, target_rank_factor)
    +main()
    +__init__(self, num_patches, embed_dim)
    +forward(self, x)
    +__init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)
    +freeze(self)
    +unfreeze_mixer_only(self)
    +forward(self, x)
    +__init__(self, input_dim, hidden_dim, num_classes)
    +apply_masks(self)
  }
  class apex16_py_TaxonomicTrainer {
    <<class>>
    +compute_spectral_loss(W, target_rank_factor)
    +main()
    +__init__(self, num_patches, embed_dim)
    +forward(self, x)
    +__init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)
    +freeze(self)
    +unfreeze_mixer_only(self)
    +forward(self, x)
    +__init__(self, input_dim, hidden_dim, num_classes)
    +apply_masks(self)
  }
  class apex17_py_GatedTokenMixer {
    <<class>>
    +compute_spectral_loss(W)
    +run_hierarchy_benchmark(model_apex, model_blind, device)
    +main()
    +__init__(self, num_patches, embed_dim)
    +forward(self, x)
    +__init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)
    +freeze(self)
    +unfreeze_mixer_only(self)
    +forward(self, x)
    +__init__(self, input_dim, hidden_dim, num_classes)
  }
  class apex17_py_PatchFeatureExtractor {
    <<class>>
    +compute_spectral_loss(W)
    +run_hierarchy_benchmark(model_apex, model_blind, device)
    +main()
    +__init__(self, num_patches, embed_dim)
    +forward(self, x)
    +__init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)
    +freeze(self)
    +unfreeze_mixer_only(self)
    +forward(self, x)
    +__init__(self, input_dim, hidden_dim, num_classes)
  }
  class apex17_py_TaxonomicMLP {
    <<class>>
    +compute_spectral_loss(W)
    +run_hierarchy_benchmark(model_apex, model_blind, device)
    +main()
    +__init__(self, num_patches, embed_dim)
    +forward(self, x)
    +__init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)
    +freeze(self)
    +unfreeze_mixer_only(self)
    +forward(self, x)
    +__init__(self, input_dim, hidden_dim, num_classes)
  }
  class apex17_py_SpectralMonitor {
    <<class>>
    +compute_spectral_loss(W)
    +run_hierarchy_benchmark(model_apex, model_blind, device)
    +main()
    +__init__(self, num_patches, embed_dim)
    +forward(self, x)
    +__init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)
    +freeze(self)
    +unfreeze_mixer_only(self)
    +forward(self, x)
    +__init__(self, input_dim, hidden_dim, num_classes)
  }
  class apex17_py_TaxonomicTrainer {
    <<class>>
    +compute_spectral_loss(W)
    +run_hierarchy_benchmark(model_apex, model_blind, device)
    +main()
    +__init__(self, num_patches, embed_dim)
    +forward(self, x)
    +__init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)
    +freeze(self)
    +unfreeze_mixer_only(self)
    +forward(self, x)
    +__init__(self, input_dim, hidden_dim, num_classes)
  }
  class apex17_py_CoarseCIFAR100 {
    <<class>>
    +compute_spectral_loss(W)
    +run_hierarchy_benchmark(model_apex, model_blind, device)
    +main()
    +__init__(self, num_patches, embed_dim)
    +forward(self, x)
    +__init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)
    +freeze(self)
    +unfreeze_mixer_only(self)
    +forward(self, x)
    +__init__(self, input_dim, hidden_dim, num_classes)
  }
  class apex18_py_GatedTokenMixer {
    <<class>>
    +compute_spectral_loss(W)
    +run_hierarchy_benchmark(model_apex, model_blind, device)
    +main()
    +__init__(self, num_patches, embed_dim)
    +forward(self, x)
    +__init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)
    +freeze(self)
    +unfreeze_mixer_only(self)
    +forward(self, x)
    +__init__(self, input_dim, hidden_dim, num_classes)
  }
  class apex18_py_PatchFeatureExtractor {
    <<class>>
    +compute_spectral_loss(W)
    +run_hierarchy_benchmark(model_apex, model_blind, device)
    +main()
    +__init__(self, num_patches, embed_dim)
    +forward(self, x)
    +__init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)
    +freeze(self)
    +unfreeze_mixer_only(self)
    +forward(self, x)
    +__init__(self, input_dim, hidden_dim, num_classes)
  }
  class apex18_py_TaxonomicMLP {
    <<class>>
    +compute_spectral_loss(W)
    +run_hierarchy_benchmark(model_apex, model_blind, device)
    +main()
    +__init__(self, num_patches, embed_dim)
    +forward(self, x)
    +__init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)
    +freeze(self)
    +unfreeze_mixer_only(self)
    +forward(self, x)
    +__init__(self, input_dim, hidden_dim, num_classes)
  }
  class apex18_py_SpectralMonitor {
    <<class>>
    +compute_spectral_loss(W)
    +run_hierarchy_benchmark(model_apex, model_blind, device)
    +main()
    +__init__(self, num_patches, embed_dim)
    +forward(self, x)
    +__init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)
    +freeze(self)
    +unfreeze_mixer_only(self)
    +forward(self, x)
    +__init__(self, input_dim, hidden_dim, num_classes)
  }
  class apex18_py_TaxonomicTrainer {
    <<class>>
    +compute_spectral_loss(W)
    +run_hierarchy_benchmark(model_apex, model_blind, device)
    +main()
    +__init__(self, num_patches, embed_dim)
    +forward(self, x)
    +__init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)
    +freeze(self)
    +unfreeze_mixer_only(self)
    +forward(self, x)
    +__init__(self, input_dim, hidden_dim, num_classes)
  }
  class apex18_py_CoarseCIFAR100 {
    <<class>>
    +compute_spectral_loss(W)
    +run_hierarchy_benchmark(model_apex, model_blind, device)
    +main()
    +__init__(self, num_patches, embed_dim)
    +forward(self, x)
    +__init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)
    +freeze(self)
    +unfreeze_mixer_only(self)
    +forward(self, x)
    +__init__(self, input_dim, hidden_dim, num_classes)
  }
  class apex19_py_GatedTokenMixer {
    <<class>>
    +compute_spectral_loss(W)
    +run_hierarchy_benchmark(model_apex, model_blind, device)
    +main()
    +__init__(self, num_patches, embed_dim)
    +forward(self, x)
    +__init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)
    +freeze(self)
    +unfreeze_mixer_only(self)
    +forward(self, x)
    +__init__(self, input_dim, hidden_dim, num_classes)
  }
  class apex19_py_PatchFeatureExtractor {
    <<class>>
    +compute_spectral_loss(W)
    +run_hierarchy_benchmark(model_apex, model_blind, device)
    +main()
    +__init__(self, num_patches, embed_dim)
    +forward(self, x)
    +__init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)
    +freeze(self)
    +unfreeze_mixer_only(self)
    +forward(self, x)
    +__init__(self, input_dim, hidden_dim, num_classes)
  }
  class apex19_py_TaxonomicMLP {
    <<class>>
    +compute_spectral_loss(W)
    +run_hierarchy_benchmark(model_apex, model_blind, device)
    +main()
    +__init__(self, num_patches, embed_dim)
    +forward(self, x)
    +__init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)
    +freeze(self)
    +unfreeze_mixer_only(self)
    +forward(self, x)
    +__init__(self, input_dim, hidden_dim, num_classes)
  }
  class apex19_py_SpectralMonitor {
    <<class>>
    +compute_spectral_loss(W)
    +run_hierarchy_benchmark(model_apex, model_blind, device)
    +main()
    +__init__(self, num_patches, embed_dim)
    +forward(self, x)
    +__init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)
    +freeze(self)
    +unfreeze_mixer_only(self)
    +forward(self, x)
    +__init__(self, input_dim, hidden_dim, num_classes)
  }
  class apex19_py_TaxonomicTrainer {
    <<class>>
    +compute_spectral_loss(W)
    +run_hierarchy_benchmark(model_apex, model_blind, device)
    +main()
    +__init__(self, num_patches, embed_dim)
    +forward(self, x)
    +__init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)
    +freeze(self)
    +unfreeze_mixer_only(self)
    +forward(self, x)
    +__init__(self, input_dim, hidden_dim, num_classes)
  }
  class apex19_py_CoarseCIFAR100 {
    <<class>>
    +compute_spectral_loss(W)
    +run_hierarchy_benchmark(model_apex, model_blind, device)
    +main()
    +__init__(self, num_patches, embed_dim)
    +forward(self, x)
    +__init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)
    +freeze(self)
    +unfreeze_mixer_only(self)
    +forward(self, x)
    +__init__(self, input_dim, hidden_dim, num_classes)
  }
  class apex20_py_GatedTokenMixer {
    <<class>>
    +compute_spectral_loss(W)
    +run_hierarchy_benchmark(model_apex, model_blind, device)
    +main()
    +__init__(self, num_patches, embed_dim)
    +forward(self, x)
    +__init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)
    +freeze(self)
    +unfreeze_mixer_only(self)
    +forward(self, x)
    +__init__(self, input_dim, hidden_dim, num_classes)
  }
  class apex20_py_PatchFeatureExtractor {
    <<class>>
    +compute_spectral_loss(W)
    +run_hierarchy_benchmark(model_apex, model_blind, device)
    +main()
    +__init__(self, num_patches, embed_dim)
    +forward(self, x)
    +__init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)
    +freeze(self)
    +unfreeze_mixer_only(self)
    +forward(self, x)
    +__init__(self, input_dim, hidden_dim, num_classes)
  }
  class apex20_py_TaxonomicMLP {
    <<class>>
    +compute_spectral_loss(W)
    +run_hierarchy_benchmark(model_apex, model_blind, device)
    +main()
    +__init__(self, num_patches, embed_dim)
    +forward(self, x)
    +__init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)
    +freeze(self)
    +unfreeze_mixer_only(self)
    +forward(self, x)
    +__init__(self, input_dim, hidden_dim, num_classes)
  }
  class apex20_py_SpectralMonitor {
    <<class>>
    +compute_spectral_loss(W)
    +run_hierarchy_benchmark(model_apex, model_blind, device)
    +main()
    +__init__(self, num_patches, embed_dim)
    +forward(self, x)
    +__init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)
    +freeze(self)
    +unfreeze_mixer_only(self)
    +forward(self, x)
    +__init__(self, input_dim, hidden_dim, num_classes)
  }
  class apex20_py_TaxonomicTrainer {
    <<class>>
    +compute_spectral_loss(W)
    +run_hierarchy_benchmark(model_apex, model_blind, device)
    +main()
    +__init__(self, num_patches, embed_dim)
    +forward(self, x)
    +__init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)
    +freeze(self)
    +unfreeze_mixer_only(self)
    +forward(self, x)
    +__init__(self, input_dim, hidden_dim, num_classes)
  }
  class apex20_py_CoarseCIFAR100 {
    <<class>>
    +compute_spectral_loss(W)
    +run_hierarchy_benchmark(model_apex, model_blind, device)
    +main()
    +__init__(self, num_patches, embed_dim)
    +forward(self, x)
    +__init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)
    +freeze(self)
    +unfreeze_mixer_only(self)
    +forward(self, x)
    +__init__(self, input_dim, hidden_dim, num_classes)
  }
  class apex21_py_GatedTokenMixer {
    <<class>>
    +compute_spectral_loss(W)
    +run_hierarchy_benchmark(model_apex, model_blind, device)
    +main()
    +__init__(self, num_patches, embed_dim)
    +forward(self, x)
    +__init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)
    +freeze(self)
    +unfreeze_mixer_only(self)
    +forward(self, x)
    +__init__(self, input_dim, hidden_dim, num_classes)
  }
  class apex21_py_PatchFeatureExtractor {
    <<class>>
    +compute_spectral_loss(W)
    +run_hierarchy_benchmark(model_apex, model_blind, device)
    +main()
    +__init__(self, num_patches, embed_dim)
    +forward(self, x)
    +__init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)
    +freeze(self)
    +unfreeze_mixer_only(self)
    +forward(self, x)
    +__init__(self, input_dim, hidden_dim, num_classes)
  }
  class apex21_py_TaxonomicMLP {
    <<class>>
    +compute_spectral_loss(W)
    +run_hierarchy_benchmark(model_apex, model_blind, device)
    +main()
    +__init__(self, num_patches, embed_dim)
    +forward(self, x)
    +__init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)
    +freeze(self)
    +unfreeze_mixer_only(self)
    +forward(self, x)
    +__init__(self, input_dim, hidden_dim, num_classes)
  }
  class apex21_py_SpectralMonitor {
    <<class>>
    +compute_spectral_loss(W)
    +run_hierarchy_benchmark(model_apex, model_blind, device)
    +main()
    +__init__(self, num_patches, embed_dim)
    +forward(self, x)
    +__init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)
    +freeze(self)
    +unfreeze_mixer_only(self)
    +forward(self, x)
    +__init__(self, input_dim, hidden_dim, num_classes)
  }
  class apex21_py_TopologyController {
    <<class>>
    +compute_spectral_loss(W)
    +run_hierarchy_benchmark(model_apex, model_blind, device)
    +main()
    +__init__(self, num_patches, embed_dim)
    +forward(self, x)
    +__init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)
    +freeze(self)
    +unfreeze_mixer_only(self)
    +forward(self, x)
    +__init__(self, input_dim, hidden_dim, num_classes)
  }
  class apex21_py_TaxonomicTrainer {
    <<class>>
    +compute_spectral_loss(W)
    +run_hierarchy_benchmark(model_apex, model_blind, device)
    +main()
    +__init__(self, num_patches, embed_dim)
    +forward(self, x)
    +__init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)
    +freeze(self)
    +unfreeze_mixer_only(self)
    +forward(self, x)
    +__init__(self, input_dim, hidden_dim, num_classes)
  }
  class apex21_py_CoarseCIFAR100 {
    <<class>>
    +compute_spectral_loss(W)
    +run_hierarchy_benchmark(model_apex, model_blind, device)
    +main()
    +__init__(self, num_patches, embed_dim)
    +forward(self, x)
    +__init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)
    +freeze(self)
    +unfreeze_mixer_only(self)
    +forward(self, x)
    +__init__(self, input_dim, hidden_dim, num_classes)
  }
  class apex22_py_GatedTokenMixer {
    <<class>>
    +compute_spectral_loss(W)
    +run_hierarchy_benchmark(model_apex, model_blind, device)
    +main()
    +__init__(self, num_patches, embed_dim)
    +forward(self, x)
    +__init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)
    +freeze(self)
    +unfreeze_mixer_only(self)
    +forward(self, x)
    +__init__(self, input_dim, hidden_dim, num_classes)
  }
  class apex22_py_PatchFeatureExtractor {
    <<class>>
    +compute_spectral_loss(W)
    +run_hierarchy_benchmark(model_apex, model_blind, device)
    +main()
    +__init__(self, num_patches, embed_dim)
    +forward(self, x)
    +__init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)
    +freeze(self)
    +unfreeze_mixer_only(self)
    +forward(self, x)
    +__init__(self, input_dim, hidden_dim, num_classes)
  }
  class apex22_py_TaxonomicMLP {
    <<class>>
    +compute_spectral_loss(W)
    +run_hierarchy_benchmark(model_apex, model_blind, device)
    +main()
    +__init__(self, num_patches, embed_dim)
    +forward(self, x)
    +__init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)
    +freeze(self)
    +unfreeze_mixer_only(self)
    +forward(self, x)
    +__init__(self, input_dim, hidden_dim, num_classes)
  }
```

---

## Code Property Graph

Machine-readable Code Property Graph (CPG) in JSON-LD format. This block allows AI agents to parse the full structural graph without additional file reads. Compatible with GraphRAG pipelines.

```json
{"@context": "https://schema.org", "analysis": {"communities": [], "god_nodes": [{"node_id": "apex28.py", "score": 4.7}, {"node_id": "apex29.py", "score": 4.7}, {"node_id": "apex30.py", "score": 4.7}, {"node_id": "apex31.py", "score": 4.7}, {"node_id": "apex33.py", "score": 4.6}, {"node_id": "apex34.py", "score": 4.5}, {"node_id": "apex26.py", "score": 4.2}, {"node_id": "apex27.py", "score": 4.0}, {"node_id": "apex21.py", "score": 3.2}, {"node_id": "apex23.py", "score": 3.2}], "surprising_connections": []}, "edges": [{"confidence": "EXTRACTED", "relation": "imports", "source": "apex14.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex14.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex14.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex14.py", "target": "torchvision"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex14.py", "target": "torchvision.transforms"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex14.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex14.py", "target": "pandas"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex14.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex14.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex14.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex15.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex15.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex15.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex15.py", "target": "torchvision"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex15.py", "target": "torchvision.transforms"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex15.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex15.py", "target": "pandas"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex15.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex15.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex15.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex16.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex16.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex16.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex16.py", "target": "torchvision"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex16.py", "target": "torchvision.transforms"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex16.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex16.py", "target": "pandas"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex16.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex16.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex16.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex17.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex17.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex17.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex17.py", "target": "torchvision"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex17.py", "target": "torchvision.transforms"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex17.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex17.py", "target": "pandas"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex17.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex17.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex17.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex18.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex18.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex18.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex18.py", "target": "torchvision"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex18.py", "target": "torchvision.transforms"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex18.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex18.py", "target": "pandas"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex18.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex18.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex18.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex19.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex19.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex19.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex19.py", "target": "torchvision"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex19.py", "target": "torchvision.transforms"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex19.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex19.py", "target": "pandas"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex19.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex19.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex19.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex20.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex20.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex20.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex20.py", "target": "torchvision"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex20.py", "target": "torchvision.transforms"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex20.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex20.py", "target": "pandas"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex20.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex20.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex20.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex21.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex21.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex21.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex21.py", "target": "torchvision"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex21.py", "target": "torchvision.transforms"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex21.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex21.py", "target": "pandas"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex21.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex21.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex21.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex22.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex22.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex22.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex22.py", "target": "torchvision"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex22.py", "target": "torchvision.transforms"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex22.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex22.py", "target": "pandas"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex22.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex22.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex22.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex23.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex23.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex23.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex23.py", "target": "torchvision"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex23.py", "target": "torchvision.transforms"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex23.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex23.py", "target": "pandas"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex23.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex23.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex23.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex24.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex24.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex24.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex24.py", "target": "torchvision"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex24.py", "target": "torchvision.transforms"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex24.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex24.py", "target": "pandas"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex24.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex24.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex24.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex25.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex25.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex25.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex25.py", "target": "torchvision"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex25.py", "target": "torchvision.transforms"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex25.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex25.py", "target": "pandas"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex25.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex25.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex25.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex26.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex26.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex26.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex26.py", "target": "torchvision"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex26.py", "target": "torchvision.transforms"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex26.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex26.py", "target": "pandas"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex26.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex26.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex26.py", "target": "time"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex26.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex26.py", "target": "random"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex26.py", "target": "argparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex26.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex26.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex27.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex27.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex27.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex27.py", "target": "torchvision"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex27.py", "target": "torchvision.transforms"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex27.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex27.py", "target": "pandas"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex27.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex27.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex27.py", "target": "time"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex27.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex27.py", "target": "random"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex27.py", "target": "argparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex27.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex27.py", "target": "collections"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex27.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex28.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex28.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex28.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex28.py", "target": "torchvision"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex28.py", "target": "torchvision.transforms"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex28.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex28.py", "target": "pandas"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex28.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex28.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex28.py", "target": "time"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex28.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex28.py", "target": "random"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex28.py", "target": "argparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex28.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex28.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex29.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex29.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex29.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex29.py", "target": "torchvision"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex29.py", "target": "torchvision.transforms"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex29.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex29.py", "target": "pandas"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex29.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex29.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex29.py", "target": "time"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex29.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex29.py", "target": "random"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex29.py", "target": "argparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex29.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex29.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex30.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex30.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex30.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex30.py", "target": "torchvision"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex30.py", "target": "torchvision.transforms"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex30.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex30.py", "target": "pandas"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex30.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex30.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex30.py", "target": "time"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex30.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex30.py", "target": "random"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex30.py", "target": "argparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex30.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex30.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex31.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex31.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex31.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex31.py", "target": "torchvision"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex31.py", "target": "torchvision.transforms"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex31.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex31.py", "target": "pandas"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex31.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex31.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex31.py", "target": "time"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex31.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex31.py", "target": "random"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex31.py", "target": "argparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex31.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex31.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex32.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex32.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex32.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex32.py", "target": "torchvision"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex32.py", "target": "torchvision.transforms"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex32.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex32.py", "target": "pandas"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex32.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex32.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex32.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex33.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex33.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex33.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex33.py", "target": "torchvision"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex33.py", "target": "torchvision.transforms"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex33.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex33.py", "target": "pandas"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex33.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex33.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex33.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex33.py", "target": "random"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex33.py", "target": "argparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex33.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex33.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex34.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex34.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex34.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex34.py", "target": "torchvision"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex34.py", "target": "torchvision.transforms"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex34.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex34.py", "target": "pandas"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex34.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex34.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex34.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex34.py", "target": "random"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex34.py", "target": "argparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex34.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex34.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex35.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex35.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex35.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex35.py", "target": "torchvision"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex35.py", "target": "torchvision.transforms"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex35.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex35.py", "target": "pandas"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex35.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex35.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex35.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex35.py", "target": "random"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex35.py", "target": "argparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex35.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "apex35.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app.py", "target": "torchvision"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app.py", "target": "torchvision.transforms"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app.py", "target": "pandas"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app.py", "target": "time"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app.py", "target": "sklearn.metrics.pairwise"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank.py", "target": "torchvision"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank.py", "target": "torchvision.transforms"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank10.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank10.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank10.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank10.py", "target": "torchvision"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank10.py", "target": "torchvision.transforms"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank10.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank10.py", "target": "pandas"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank10.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank10.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank10.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank11.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank11.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank11.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank11.py", "target": "torchvision"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank11.py", "target": "torchvision.transforms"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank11.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank11.py", "target": "pandas"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank11.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank11.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank11.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank12.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank12.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank12.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank12.py", "target": "torchvision"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank12.py", "target": "torchvision.transforms"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank12.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank12.py", "target": "pandas"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank12.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank12.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank12.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank13.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank13.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank13.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank13.py", "target": "torchvision"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank13.py", "target": "torchvision.transforms"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank13.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank13.py", "target": "pandas"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank13.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank13.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank13.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank2.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank2.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank2.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank2.py", "target": "torchvision"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank2.py", "target": "torchvision.transforms"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank2.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank2.py", "target": "pandas"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank2.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank2.py", "target": "time"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank2.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank2.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank3.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank3.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank3.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank3.py", "target": "torchvision"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank3.py", "target": "torchvision.transforms"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank3.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank3.py", "target": "pandas"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank3.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank3.py", "target": "time"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank3.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank3.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank4.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank4.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank4.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank4.py", "target": "torchvision"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank4.py", "target": "torchvision.transforms"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank4.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank4.py", "target": "pandas"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank4.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank4.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank4.py", "target": "time"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank4.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank4.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank5.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank5.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank5.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank5.py", "target": "torchvision"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank5.py", "target": "torchvision.transforms"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank5.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank5.py", "target": "pandas"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank5.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank5.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank5.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank5.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank5.py", "target": "sklearn.metrics.pairwise"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank5.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank6.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank6.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank6.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank6.py", "target": "torchvision"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank6.py", "target": "torchvision.transforms"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank6.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank6.py", "target": "pandas"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank6.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank6.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank6.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank6.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank7.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank7.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank7.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank7.py", "target": "torchvision"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank7.py", "target": "torchvision.transforms"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank7.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank7.py", "target": "pandas"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank7.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank7.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank7.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank7.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank8.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank8.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank8.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank8.py", "target": "torchvision"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank8.py", "target": "torchvision.transforms"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank8.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank8.py", "target": "pandas"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank8.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank8.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank8.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resmav2_1.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resmav2_1.py", "target": "glob"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resmav2_1.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resmav2_1.py", "target": "zipfile"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resmav2_1.py", "target": "kagglehub"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resmav2_1.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resmav2_1.py", "target": "pandas"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resmav2_1.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resmav2_1.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resmav2_1.py", "target": "torch_geometric.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resmav2_1.py", "target": "torch_geometric.utils"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resmav2_1.py", "target": "sklearn.preprocessing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resmav2_1.py", "target": "sklearn.model_selection"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resmav2_1.py", "target": "sklearn.metrics"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resmav2_1.py", "target": "time"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "resmav2_1.py", "target": "warnings"}], "generator": "readmenator", "metadata": {"edge_count": 12028, "file_count": 37, "language_count": 2, "symbol_count": 1059}, "nodes": [{"doc": "NeuroSovereign v14.0: Hierarchical Apex (FIXED) Objective: Prove Structural Necessity on CIFAR-100 (Hierarchical Task). Features: 1. Dataset Upgrade: CIFAR-10 -> CIFAR-100 (Requires feature composition). 2. Gated Token Mixer: Content-aware mixing (not just static linear). 3. Gate Usage Metric: Implicit ablation to prove mixer activity.", "id": "apex14.py", "kind": "module", "label": "apex14.py", "language": "py", "sha256": "9ded4c39036618cd", "symbol_count": 26, "symbols": [{"kind": "class", "line": 36, "name": "GatedTokenMixer", "signature": "class GatedTokenMixer(Module)"}, {"doc": "Extractor configurable para CIFAR-100.\nuse_mixer=True -> Apex (Syntactic)\nuse_mixer=False -> Blind Structural Baseline", "kind": "class", "line": 66, "name": "PatchFeatureExtractor", "signature": "class PatchFeatureExtractor(Module)"}, {"kind": "class", "line": 111, "name": "LotteryMLP", "signature": "class LotteryMLP(Module)"}, {"kind": "class", "line": 142, "name": "SpectralMonitor", "signature": "class SpectralMonitor"}, {"kind": "class", "line": 158, "name": "OrthogonalEvolutionEngine", "signature": "class OrthogonalEvolutionEngine"}, {"kind": "class", "line": 231, "name": "HierarchicalTrainer", "signature": "class HierarchicalTrainer"}, {"kind": "method", "line": 336, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 37, "name": "__init__", "signature": "def __init__(self, num_tokens, embed_dim)"}, {"kind": "method", "line": 54, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 72, "name": "__init__", "signature": "def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)"}, {"kind": "method", "line": 89, "name": "freeze", "signature": "def freeze(self)"}, {"kind": "method", "line": 93, "name": "unfreeze", "signature": "def unfreeze(self)"}, {"kind": "method", "line": 97, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 112, "name": "__init__", "signature": "def __init__(self, input_dim, hidden_dim, num_classes)"}, {"kind": "method", "line": 123, "name": "apply_masks", "signature": "def apply_masks(self)"}, {"kind": "method", "line": 128, "name": "get_sparsity", "signature": "def get_sparsity(self)"}, {"kind": "method", "line": 133, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 143, "name": "compute_metrics", "signature": "def compute_metrics(self, weight)"}, {"kind": "method", "line": 159, "name": "__init__", "signature": "def __init__(self, device)"}, {"kind": "method", "line": 164, "name": "_gradient_nudge_inheritance", "signature": "def _gradient_nudge_inheritance(self, child_model, elk_state, data_loader, feature_extractor, nudge_lr)"}, {"kind": "method", "line": 198, "name": "_apply_rank_capping_shock", "signature": "def _apply_rank_capping_shock(self, model, layer_name, keep_ratio)"}, {"kind": "method", "line": 217, "name": "create_refined_offspring", "signature": "def create_refined_offspring(self, elk_state, data_loader, feature_extractor)"}, {"kind": "method", "line": 232, "name": "__init__", "signature": "def __init__(self, device, extractor_apex, extractor_blind)"}, {"kind": "method", "line": 240, "name": "_preprocess_batch", "signature": "def _preprocess_batch(self, x, extractor)"}, {"kind": "method", "line": 244, "name": "get_curriculum_dataset", "signature": "def get_curriculum_dataset(self, cycle)"}, {"kind": "method", "line": 250, "name": "train_single_chain", "signature": "def train_single_chain(self, model, cycle, chain_type)"}]}, {"doc": "NeuroSovereign v14.4: SOTA Structural Control & Emergency Shock Objective: Stabilize Training via Active Spectral Regularization & Dynamic Shock. Changes from v14.3: 1. FIX: Frozen fc_super in BLIND to prevent NoneType gradient errors. 2. REMOVED: Genetic Nudge (Replaced by direct state inheritance for reproducibility). 3. ADDED: Spectral Entropy Loss (Active L-Metric) to prevent memory collapse. 4. ADDED: Taxonomic Shock Logic (Lambda boost if Gap > 5.0). 5. ADDED: Mixer LR Injection (5x learning rate for Mixer weights).", "id": "apex15.py", "kind": "module", "label": "apex15.py", "language": "py", "sha256": "58b3e7d774ee314f", "symbol_count": 22, "symbols": [{"doc": "Penaliza la desalineación entre Entropía Espectral y Rango Efectivo.\nEsta es la versión 'activa' de la métrica L.", "kind": "function", "line": 63, "name": "compute_spectral_loss", "signature": "def compute_spectral_loss(W, target_rank_factor)"}, {"kind": "class", "line": 88, "name": "GatedTokenMixer", "signature": "class GatedTokenMixer(Module)"}, {"kind": "class", "line": 105, "name": "PatchFeatureExtractor", "signature": "class PatchFeatureExtractor(Module)"}, {"kind": "class", "line": 143, "name": "TaxonomicMLP", "signature": "class TaxonomicMLP(Module)"}, {"kind": "class", "line": 179, "name": "SpectralMonitor", "signature": "class SpectralMonitor"}, {"kind": "class", "line": 195, "name": "TaxonomicTrainer", "signature": "class TaxonomicTrainer"}, {"kind": "method", "line": 367, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 89, "name": "__init__", "signature": "def __init__(self, num_patches, embed_dim)"}, {"kind": "method", "line": 98, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 106, "name": "__init__", "signature": "def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)"}, {"kind": "method", "line": 121, "name": "freeze", "signature": "def freeze(self)"}, {"kind": "method", "line": 125, "name": "unfreeze_mixer_only", "signature": "def unfreeze_mixer_only(self)"}, {"kind": "method", "line": 131, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 144, "name": "__init__", "signature": "def __init__(self, input_dim, hidden_dim, num_classes)"}, {"kind": "method", "line": 158, "name": "apply_masks", "signature": "def apply_masks(self)"}, {"kind": "method", "line": 164, "name": "get_sparsity", "signature": "def get_sparsity(self)"}, {"kind": "method", "line": 169, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 180, "name": "compute_metrics", "signature": "def compute_metrics(self, weight)"}, {"kind": "method", "line": 196, "name": "__init__", "signature": "def __init__(self, device, extractor_apex, extractor_blind)"}, {"kind": "method", "line": 203, "name": "_preprocess_batch", "signature": "def _preprocess_batch(self, x, extractor)"}, {"kind": "method", "line": 207, "name": "get_curriculum_dataset", "signature": "def get_curriculum_dataset(self, cycle)"}, {"kind": "method", "line": 213, "name": "train_single_chain", "signature": "def train_single_chain(self, model, cycle, chain_type)"}]}, {"doc": "NeuroSovereign v14.5: Adaptive Shock & Full Spectrum Control Objective: Fix v14.4 based on technical review. Changes from v14.4: 1. FIX: Adaptive Taxonomic Shock (Lambda reacts to REAL validation gap, not epochs). 2. ADDED: Spectral Loss applied to Mixer weights (Upstream structural control). 3. ADDED: Feedback Loop Logic (Prev epoch gap determines current epoch lambda). 4. TUNED: Gap Threshold set to 5.0 for Shock activation.", "id": "apex16.py", "kind": "module", "label": "apex16.py", "language": "py", "sha256": "a397bcac15537029", "symbol_count": 22, "symbols": [{"doc": "Penaliza la desalineación entre Entropía Espectral y Rango Efectivo.\nv14.5: Se aplicará a pesos del MLP y del Mixer.", "kind": "function", "line": 63, "name": "compute_spectral_loss", "signature": "def compute_spectral_loss(W, target_rank_factor)"}, {"kind": "class", "line": 82, "name": "GatedTokenMixer", "signature": "class GatedTokenMixer(Module)"}, {"kind": "class", "line": 99, "name": "PatchFeatureExtractor", "signature": "class PatchFeatureExtractor(Module)"}, {"kind": "class", "line": 137, "name": "TaxonomicMLP", "signature": "class TaxonomicMLP(Module)"}, {"kind": "class", "line": 173, "name": "SpectralMonitor", "signature": "class SpectralMonitor"}, {"kind": "class", "line": 189, "name": "TaxonomicTrainer", "signature": "class TaxonomicTrainer"}, {"kind": "method", "line": 377, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 83, "name": "__init__", "signature": "def __init__(self, num_patches, embed_dim)"}, {"kind": "method", "line": 92, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 100, "name": "__init__", "signature": "def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)"}, {"kind": "method", "line": 115, "name": "freeze", "signature": "def freeze(self)"}, {"kind": "method", "line": 119, "name": "unfreeze_mixer_only", "signature": "def unfreeze_mixer_only(self)"}, {"kind": "method", "line": 125, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 138, "name": "__init__", "signature": "def __init__(self, input_dim, hidden_dim, num_classes)"}, {"kind": "method", "line": 152, "name": "apply_masks", "signature": "def apply_masks(self)"}, {"kind": "method", "line": 158, "name": "get_sparsity", "signature": "def get_sparsity(self)"}, {"kind": "method", "line": 163, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 174, "name": "compute_metrics", "signature": "def compute_metrics(self, weight)"}, {"kind": "method", "line": 190, "name": "__init__", "signature": "def __init__(self, device, extractor_apex, extractor_blind)"}, {"kind": "method", "line": 197, "name": "_preprocess_batch", "signature": "def _preprocess_batch(self, x, extractor)"}, {"kind": "method", "line": 201, "name": "get_curriculum_dataset", "signature": "def get_curriculum_dataset(self, cycle)"}, {"kind": "method", "line": 207, "name": "train_single_chain", "signature": "def train_single_chain(self, model, cycle, chain_type)"}]}, {"doc": "NeuroSovereign v15.0: Paper Candidate Release Objective: Finalize architecture for rigorous review. Changes from v14.5: 1. CLARITY: Separated L_opt (Optimization Objective) vs L_mon (Reporting Metric) in logs. 2. BENCHMARK: Added CIFAR-20 Stress Test (Coarse-only validation). 3. LOGIC: Final validation based on Delta (APEX - BLIND) to prove structural advantage.", "id": "apex17.py", "kind": "module", "label": "apex17.py", "language": "py", "sha256": "e921efb7c2349c57", "symbol_count": 26, "symbols": [{"doc": "v15.0: Optimization Objective for Spectral Control.", "kind": "function", "line": 65, "name": "compute_spectral_loss", "signature": "def compute_spectral_loss(W)"}, {"kind": "class", "line": 81, "name": "GatedTokenMixer", "signature": "class GatedTokenMixer(Module)"}, {"kind": "class", "line": 98, "name": "PatchFeatureExtractor", "signature": "class PatchFeatureExtractor(Module)"}, {"kind": "class", "line": 136, "name": "TaxonomicMLP", "signature": "class TaxonomicMLP(Module)"}, {"kind": "class", "line": 172, "name": "SpectralMonitor", "signature": "class SpectralMonitor"}, {"kind": "class", "line": 190, "name": "TaxonomicTrainer", "signature": "class TaxonomicTrainer"}, {"doc": "Wrapper que convierte CIFAR100 en un problema de clasificación pura de 20 clases (Superclases).\nSe usa para validar el inductive bias aprendido.", "kind": "class", "line": 368, "name": "CoarseCIFAR100", "signature": "class CoarseCIFAR100(CIFAR100)"}, {"kind": "method", "line": 378, "name": "run_hierarchy_benchmark", "signature": "def run_hierarchy_benchmark(model_apex, model_blind, device)"}, {"kind": "method", "line": 428, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 82, "name": "__init__", "signature": "def __init__(self, num_patches, embed_dim)"}, {"kind": "method", "line": 91, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 99, "name": "__init__", "signature": "def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)"}, {"kind": "method", "line": 114, "name": "freeze", "signature": "def freeze(self)"}, {"kind": "method", "line": 118, "name": "unfreeze_mixer_only", "signature": "def unfreeze_mixer_only(self)"}, {"kind": "method", "line": 124, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 137, "name": "__init__", "signature": "def __init__(self, input_dim, hidden_dim, num_classes)"}, {"kind": "method", "line": 151, "name": "apply_masks", "signature": "def apply_masks(self)"}, {"kind": "method", "line": 157, "name": "get_sparsity", "signature": "def get_sparsity(self)"}, {"kind": "method", "line": 162, "name": "forward", "signature": "def forward(self, x)"}, {"doc": "L_mon: Used for plotting and historical reporting, not optimization.", "kind": "method", "line": 173, "name": "compute_metrics", "signature": "def compute_metrics(self, weight)"}, {"kind": "method", "line": 191, "name": "__init__", "signature": "def __init__(self, device, extractor_apex, extractor_blind)"}, {"kind": "method", "line": 198, "name": "_preprocess_batch", "signature": "def _preprocess_batch(self, x, extractor)"}, {"kind": "method", "line": 202, "name": "get_curriculum_dataset", "signature": "def get_curriculum_dataset(self, cycle)"}, {"kind": "method", "line": 208, "name": "train_single_chain", "signature": "def train_single_chain(self, model, cycle, chain_type)"}, {"kind": "method", "line": 373, "name": "__getitem__", "signature": "def __getitem__(self, index)"}, {"kind": "method", "line": 391, "name": "evaluate", "signature": "def evaluate(model, extractor, name)"}]}, {"doc": "NeuroSovereign v15.1: Symmetric Control (The Reviewer's Fix) Objective: Isolate the Hierarchy Signal by equating Regularization budgets. Changes from v15.0: 1. FIX: BLIND now includes Spectral Loss (L_opt) on FC1. 2. FIX: BLIND now includes Sparse Penalty (applies where applicable). 3. LOGIC: APEX vs BLIND comparison is now valid; only difference is Hierarchy Signal (CE_coarse).", "id": "apex18.py", "kind": "module", "label": "apex18.py", "language": "py", "sha256": "225b3fe1d7d2502f", "symbol_count": 26, "symbols": [{"doc": "v15.1: Optimization Objective for Spectral Control (Applied to both APEX and BLIND).", "kind": "function", "line": 65, "name": "compute_spectral_loss", "signature": "def compute_spectral_loss(W)"}, {"kind": "class", "line": 81, "name": "GatedTokenMixer", "signature": "class GatedTokenMixer(Module)"}, {"kind": "class", "line": 98, "name": "PatchFeatureExtractor", "signature": "class PatchFeatureExtractor(Module)"}, {"kind": "class", "line": 136, "name": "TaxonomicMLP", "signature": "class TaxonomicMLP(Module)"}, {"kind": "class", "line": 172, "name": "SpectralMonitor", "signature": "class SpectralMonitor"}, {"kind": "class", "line": 188, "name": "TaxonomicTrainer", "signature": "class TaxonomicTrainer"}, {"kind": "class", "line": 369, "name": "CoarseCIFAR100", "signature": "class CoarseCIFAR100(CIFAR100)"}, {"kind": "method", "line": 374, "name": "run_hierarchy_benchmark", "signature": "def run_hierarchy_benchmark(model_apex, model_blind, device)"}, {"kind": "method", "line": 420, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 82, "name": "__init__", "signature": "def __init__(self, num_patches, embed_dim)"}, {"kind": "method", "line": 91, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 99, "name": "__init__", "signature": "def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)"}, {"kind": "method", "line": 114, "name": "freeze", "signature": "def freeze(self)"}, {"kind": "method", "line": 118, "name": "unfreeze_mixer_only", "signature": "def unfreeze_mixer_only(self)"}, {"kind": "method", "line": 124, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 137, "name": "__init__", "signature": "def __init__(self, input_dim, hidden_dim, num_classes)"}, {"kind": "method", "line": 151, "name": "apply_masks", "signature": "def apply_masks(self)"}, {"kind": "method", "line": 157, "name": "get_sparsity", "signature": "def get_sparsity(self)"}, {"kind": "method", "line": 162, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 173, "name": "compute_metrics", "signature": "def compute_metrics(self, weight)"}, {"kind": "method", "line": 189, "name": "__init__", "signature": "def __init__(self, device, extractor_apex, extractor_blind)"}, {"kind": "method", "line": 196, "name": "_preprocess_batch", "signature": "def _preprocess_batch(self, x, extractor)"}, {"kind": "method", "line": 200, "name": "get_curriculum_dataset", "signature": "def get_curriculum_dataset(self, cycle)"}, {"kind": "method", "line": 206, "name": "train_single_chain", "signature": "def train_single_chain(self, model, cycle, chain_type)"}, {"kind": "method", "line": 370, "name": "__getitem__", "signature": "def __getitem__(self, index)"}, {"kind": "method", "line": 386, "name": "evaluate", "signature": "def evaluate(model, extractor, name)"}]}, {"doc": "NeuroSovereign v15.2: Topological Phase Detection Objective: Detect Grokking moments via Topological Ratio (L_opt / L_mon). Changes from v15.1: 1. ADDED: TopologicalMonitor to track Phase Shift (Ratio R = L_opt / L_mon). 2. LOGIC: Identification of \"Stagnation Phase\" vs \"Plasticity Phase\". 3. FOCUS: Run Cycle 0 (Seeding) to prove Architectural Necessity vs Regularization sufficiency.", "id": "apex19.py", "kind": "module", "label": "apex19.py", "language": "py", "sha256": "0a5ccada2c30c42e", "symbol_count": 27, "symbols": [{"doc": "L_opt: Optimization Objective for Structural Control.", "kind": "function", "line": 68, "name": "compute_spectral_loss", "signature": "def compute_spectral_loss(W)"}, {"kind": "class", "line": 84, "name": "GatedTokenMixer", "signature": "class GatedTokenMixer(Module)"}, {"kind": "class", "line": 101, "name": "PatchFeatureExtractor", "signature": "class PatchFeatureExtractor(Module)"}, {"kind": "class", "line": 139, "name": "TaxonomicMLP", "signature": "class TaxonomicMLP(Module)"}, {"kind": "class", "line": 175, "name": "SpectralMonitor", "signature": "class SpectralMonitor"}, {"kind": "class", "line": 211, "name": "TaxonomicTrainer", "signature": "class TaxonomicTrainer"}, {"kind": "class", "line": 394, "name": "CoarseCIFAR100", "signature": "class CoarseCIFAR100(CIFAR100)"}, {"kind": "method", "line": 399, "name": "run_hierarchy_benchmark", "signature": "def run_hierarchy_benchmark(model_apex, model_blind, device)"}, {"kind": "method", "line": 445, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 85, "name": "__init__", "signature": "def __init__(self, num_patches, embed_dim)"}, {"kind": "method", "line": 94, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 102, "name": "__init__", "signature": "def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)"}, {"kind": "method", "line": 117, "name": "freeze", "signature": "def freeze(self)"}, {"kind": "method", "line": 121, "name": "unfreeze_mixer_only", "signature": "def unfreeze_mixer_only(self)"}, {"kind": "method", "line": 127, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 140, "name": "__init__", "signature": "def __init__(self, input_dim, hidden_dim, num_classes)"}, {"kind": "method", "line": 154, "name": "apply_masks", "signature": "def apply_masks(self)"}, {"kind": "method", "line": 160, "name": "get_sparsity", "signature": "def get_sparsity(self)"}, {"kind": "method", "line": 165, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 176, "name": "compute_metrics", "signature": "def compute_metrics(self, weight)"}, {"doc": "v15.2: Calcula el ratio R = L_opt / L_mon.\nValores bajos indican alineación estable.\nValores altos o erráticos indican transición de fase (Grokking/Collapse).", "kind": "method", "line": 188, "name": "compute_topology_ratio", "signature": "def compute_topology_ratio(self, model, extractor, chain_type)"}, {"kind": "method", "line": 212, "name": "__init__", "signature": "def __init__(self, device, extractor_apex, extractor_blind)"}, {"kind": "method", "line": 219, "name": "_preprocess_batch", "signature": "def _preprocess_batch(self, x, extractor)"}, {"kind": "method", "line": 223, "name": "get_curriculum_dataset", "signature": "def get_curriculum_dataset(self, cycle)"}, {"kind": "method", "line": 229, "name": "train_single_chain", "signature": "def train_single_chain(self, model, cycle, chain_type)"}, {"kind": "method", "line": 395, "name": "__getitem__", "signature": "def __getitem__(self, index)"}, {"kind": "method", "line": 411, "name": "evaluate", "signature": "def evaluate(model, extractor, name)"}]}, {"doc": "NeuroSovereign v15.3: Relative Phase & Cross-Scale Monitor Objective: Fix arbitrary thresholds and spatial leakage for Paper-Ready Rigor. Changes from v15.2: 1. FIX: Phase Detection is now Relative (based on deviation from history mean). 2. FIX: Hierarchy Benchmark uses correct forward pass flow (fixes feature space mixing). 3. LOGIC: Explicit separation of L_opt components (Upstream vs Downstream) in logs.", "id": "apex20.py", "kind": "module", "label": "apex20.py", "language": "py", "sha256": "34799a6bf5b28d2e", "symbol_count": 28, "symbols": [{"doc": "L_opt: Optimization Objective for Structural Control.", "kind": "function", "line": 69, "name": "compute_spectral_loss", "signature": "def compute_spectral_loss(W)"}, {"kind": "class", "line": 85, "name": "GatedTokenMixer", "signature": "class GatedTokenMixer(Module)"}, {"kind": "class", "line": 102, "name": "PatchFeatureExtractor", "signature": "class PatchFeatureExtractor(Module)"}, {"kind": "class", "line": 140, "name": "TaxonomicMLP", "signature": "class TaxonomicMLP(Module)"}, {"kind": "class", "line": 176, "name": "SpectralMonitor", "signature": "class SpectralMonitor"}, {"kind": "class", "line": 242, "name": "TaxonomicTrainer", "signature": "class TaxonomicTrainer"}, {"kind": "class", "line": 426, "name": "CoarseCIFAR100", "signature": "class CoarseCIFAR100(CIFAR100)"}, {"kind": "method", "line": 431, "name": "run_hierarchy_benchmark", "signature": "def run_hierarchy_benchmark(model_apex, model_blind, device)"}, {"kind": "method", "line": 490, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 86, "name": "__init__", "signature": "def __init__(self, num_patches, embed_dim)"}, {"kind": "method", "line": 95, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 103, "name": "__init__", "signature": "def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)"}, {"kind": "method", "line": 118, "name": "freeze", "signature": "def freeze(self)"}, {"kind": "method", "line": 122, "name": "unfreeze_mixer_only", "signature": "def unfreeze_mixer_only(self)"}, {"kind": "method", "line": 128, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 141, "name": "__init__", "signature": "def __init__(self, input_dim, hidden_dim, num_classes)"}, {"kind": "method", "line": 155, "name": "apply_masks", "signature": "def apply_masks(self)"}, {"kind": "method", "line": 161, "name": "get_sparsity", "signature": "def get_sparsity(self)"}, {"kind": "method", "line": 166, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 177, "name": "compute_metrics", "signature": "def compute_metrics(self, weight)"}, {"doc": "v15.3: Detects phase state based on relative deviation, not absolute value.\nReturns: 'STABLE', 'SHIFTING', or 'INIT'", "kind": "method", "line": 190, "name": "detect_phase_state", "signature": "def detect_phase_state(self, ratio_history)"}, {"doc": "v15.3: Returns L_opt components and Total Ratio.\nRatio = L_opt_Total / L_mon(Downstream)", "kind": "method", "line": 216, "name": "compute_topology_ratio", "signature": "def compute_topology_ratio(self, model, extractor, chain_type)"}, {"kind": "method", "line": 243, "name": "__init__", "signature": "def __init__(self, device, extractor_apex, extractor_blind)"}, {"kind": "method", "line": 250, "name": "_preprocess_batch", "signature": "def _preprocess_batch(self, x, extractor)"}, {"kind": "method", "line": 254, "name": "get_curriculum_dataset", "signature": "def get_curriculum_dataset(self, cycle)"}, {"kind": "method", "line": 260, "name": "train_single_chain", "signature": "def train_single_chain(self, model, cycle, chain_type)"}, {"kind": "method", "line": 427, "name": "__getitem__", "signature": "def __getitem__(self, index)"}, {"kind": "method", "line": 443, "name": "evaluate", "signature": "def evaluate(model, extractor, name)"}]}, {"doc": "NeuroSovereign v15.4: Active Phase Intervention (Breaking the Ceiling) Objective: Move from Observation to Causal Control. Changes from v15.3: 1. ADDED: TopologyController to manage Phase Interventions. 2. LOGIC: If Phase is STABLE + Low Coarse Acc -> Inject Topological Noise (Reversibility). 3. ADDED: Stagnation Counter to trigger \"Active Shocks\". 4. FIX: Implemented causal intervention to break local minima in hierarchy learning.", "id": "apex21.py", "kind": "module", "label": "apex21.py", "language": "py", "sha256": "beac31b726ac1062", "symbol_count": 32, "symbols": [{"kind": "function", "line": 73, "name": "compute_spectral_loss", "signature": "def compute_spectral_loss(W)"}, {"kind": "class", "line": 88, "name": "GatedTokenMixer", "signature": "class GatedTokenMixer(Module)"}, {"kind": "class", "line": 105, "name": "PatchFeatureExtractor", "signature": "class PatchFeatureExtractor(Module)"}, {"kind": "class", "line": 143, "name": "TaxonomicMLP", "signature": "class TaxonomicMLP(Module)"}, {"kind": "class", "line": 179, "name": "SpectralMonitor", "signature": "class SpectralMonitor"}, {"doc": "v15.4: Manages Active Interventions to break stagnation.", "kind": "class", "line": 228, "name": "TopologyController", "signature": "class TopologyController"}, {"kind": "class", "line": 270, "name": "TaxonomicTrainer", "signature": "class TaxonomicTrainer"}, {"kind": "class", "line": 459, "name": "CoarseCIFAR100", "signature": "class CoarseCIFAR100(CIFAR100)"}, {"kind": "method", "line": 464, "name": "run_hierarchy_benchmark", "signature": "def run_hierarchy_benchmark(model_apex, model_blind, device)"}, {"kind": "method", "line": 513, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 89, "name": "__init__", "signature": "def __init__(self, num_patches, embed_dim)"}, {"kind": "method", "line": 98, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 106, "name": "__init__", "signature": "def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)"}, {"kind": "method", "line": 121, "name": "freeze", "signature": "def freeze(self)"}, {"kind": "method", "line": 125, "name": "unfreeze_mixer_only", "signature": "def unfreeze_mixer_only(self)"}, {"kind": "method", "line": 131, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 144, "name": "__init__", "signature": "def __init__(self, input_dim, hidden_dim, num_classes)"}, {"kind": "method", "line": 158, "name": "apply_masks", "signature": "def apply_masks(self)"}, {"kind": "method", "line": 164, "name": "get_sparsity", "signature": "def get_sparsity(self)"}, {"kind": "method", "line": 169, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 180, "name": "compute_metrics", "signature": "def compute_metrics(self, weight)"}, {"kind": "method", "line": 192, "name": "detect_phase_state", "signature": "def detect_phase_state(self, ratio_history)"}, {"doc": "v15.3: Returns L_opt components and Total Ratio.\nRatio = L_opt_Total / L_mon(Downstream)", "kind": "method", "line": 204, "name": "compute_topology_ratio", "signature": "def compute_topology_ratio(self, model, extractor, chain_type)"}, {"kind": "method", "line": 230, "name": "__init__", "signature": "def __init__(self)"}, {"doc": "Decides whether to intervene.\nReturns 'INTERVENE' if action is taken, 'NONE' otherwise.", "kind": "method", "line": 233, "name": "check_intervention", "signature": "def check_intervention(self, phase_state, coarse_acc, extractor)"}, {"doc": "Causal Intervention: Inject topological noise to force phase shift.", "kind": "method", "line": 255, "name": "perturb_mixer", "signature": "def perturb_mixer(self, extractor)"}, {"kind": "method", "line": 271, "name": "__init__", "signature": "def __init__(self, device, extractor_apex, extractor_blind)"}, {"kind": "method", "line": 279, "name": "_preprocess_batch", "signature": "def _preprocess_batch(self, x, extractor)"}, {"kind": "method", "line": 283, "name": "get_curriculum_dataset", "signature": "def get_curriculum_dataset(self, cycle)"}, {"kind": "method", "line": 289, "name": "train_single_chain", "signature": "def train_single_chain(self, model, cycle, chain_type)"}, {"kind": "method", "line": 460, "name": "__getitem__", "signature": "def __getitem__(self, index)"}, {"kind": "method", "line": 476, "name": "evaluate", "signature": "def evaluate(model, extractor, name)"}]}, {"doc": "NeuroSovereign v15.5: Targeted Phase Surgery (Spectral Noise) Objective: Fix \"Blind Noise\" critique by using Orthogonal Projection. Changes from v15.4: 1. FIX: perturb_mixer() now uses Targeted Spectral Noise (Orthogonal to dominant subspace). 2. LOGIC: Intervention preserves dominant features while exciting latent modes. 3. MATH: Projection matrix P = I - V_dominant @ V_dominant.T ensures noise injection only in weak dimensions.", "id": "apex22.py", "kind": "module", "label": "apex22.py", "language": "py", "sha256": "89363ebbc5f96001", "symbol_count": 31, "symbols": [{"kind": "function", "line": 73, "name": "compute_spectral_loss", "signature": "def compute_spectral_loss(W)"}, {"kind": "class", "line": 88, "name": "GatedTokenMixer", "signature": "class GatedTokenMixer(Module)"}, {"kind": "class", "line": 105, "name": "PatchFeatureExtractor", "signature": "class PatchFeatureExtractor(Module)"}, {"kind": "class", "line": 143, "name": "TaxonomicMLP", "signature": "class TaxonomicMLP(Module)"}, {"kind": "class", "line": 179, "name": "SpectralMonitor", "signature": "class SpectralMonitor"}, {"doc": "v15.5: Manages Targeted Spectral Interventions (Surgery).", "kind": "class", "line": 204, "name": "TopologyController", "signature": "class TopologyController"}, {"kind": "class", "line": 273, "name": "TaxonomicTrainer", "signature": "class TaxonomicTrainer"}, {"kind": "class", "line": 459, "name": "CoarseCIFAR100", "signature": "class CoarseCIFAR100(CIFAR100)"}, {"kind": "method", "line": 464, "name": "run_hierarchy_benchmark", "signature": "def run_hierarchy_benchmark(model_apex, model_blind, device)"}, {"kind": "method", "line": 513, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 89, "name": "__init__", "signature": "def __init__(self, num_patches, embed_dim)"}, {"kind": "method", "line": 98, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 106, "name": "__init__", "signature": "def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)"}, {"kind": "method", "line": 121, "name": "freeze", "signature": "def freeze(self)"}, {"kind": "method", "line": 125, "name": "unfreeze_mixer_only", "signature": "def unfreeze_mixer_only(self)"}, {"kind": "method", "line": 131, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 144, "name": "__init__", "signature": "def __init__(self, input_dim, hidden_dim, num_classes)"}, {"kind": "method", "line": 158, "name": "apply_masks", "signature": "def apply_masks(self)"}, {"kind": "method", "line": 164, "name": "get_sparsity", "signature": "def get_sparsity(self)"}, {"kind": "method", "line": 169, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 180, "name": "compute_metrics", "signature": "def compute_metrics(self, weight)"}, {"kind": "method", "line": 192, "name": "detect_phase_state", "signature": "def detect_phase_state(self, ratio_history)"}, {"kind": "method", "line": 206, "name": "__init__", "signature": "def __init__(self)"}, {"kind": "method", "line": 209, "name": "check_intervention", "signature": "def check_intervention(self, phase_state, coarse_acc, extractor)"}, {"doc": "v15.5: Targeted Phase Surgery.\nInjects noise ONLY in the nullspace of the dominant spectral subspace.\nPreserves existing structure while forcing exploration of latent dimensions.", "kind": "method", "line": 226, "name": "perturb_mixer_targeted", "signature": "def perturb_mixer_targeted(self, extractor)"}, {"kind": "method", "line": 274, "name": "__init__", "signature": "def __init__(self, device, extractor_apex, extractor_blind)"}, {"kind": "method", "line": 282, "name": "_preprocess_batch", "signature": "def _preprocess_batch(self, x, extractor)"}, {"kind": "method", "line": 286, "name": "get_curriculum_dataset", "signature": "def get_curriculum_dataset(self, cycle)"}, {"kind": "method", "line": 292, "name": "train_single_chain", "signature": "def train_single_chain(self, model, cycle, chain_type)"}, {"kind": "method", "line": 460, "name": "__getitem__", "signature": "def __getitem__(self, index)"}, {"kind": "method", "line": 476, "name": "evaluate", "signature": "def evaluate(model, extractor, name)"}]}, {"doc": "NeuroSovereign v15.5: Targeted Phase Surgery (FINAL RELEASE) Objective: Break Hierarchy Ceiling via Orthogonal Spectral Projection. Fixes Applied: 1. Syntax correction in compute_spectral_loss. 2. Implementation of compute_topology_ratio for Phase Monitoring. 3. Geometric fix in Orthogonal Projection (Input Space Injection). 4. Robust State Management (Migration from v15.4).", "id": "apex23.py", "kind": "module", "label": "apex23.py", "language": "py", "sha256": "ebe99d0ba4aaf112", "symbol_count": 32, "symbols": [{"doc": "L_opt: Computes the discrepancy between spectral entropy and effective rank.", "kind": "function", "line": 74, "name": "compute_spectral_loss", "signature": "def compute_spectral_loss(W)"}, {"kind": "class", "line": 91, "name": "GatedTokenMixer", "signature": "class GatedTokenMixer(Module)"}, {"kind": "class", "line": 108, "name": "PatchFeatureExtractor", "signature": "class PatchFeatureExtractor(Module)"}, {"kind": "class", "line": 146, "name": "TaxonomicMLP", "signature": "class TaxonomicMLP(Module)"}, {"kind": "class", "line": 182, "name": "SpectralMonitor", "signature": "class SpectralMonitor"}, {"kind": "class", "line": 228, "name": "TopologyController", "signature": "class TopologyController"}, {"kind": "class", "line": 306, "name": "TaxonomicTrainer", "signature": "class TaxonomicTrainer"}, {"kind": "class", "line": 493, "name": "CoarseCIFAR100", "signature": "class CoarseCIFAR100(CIFAR100)"}, {"kind": "method", "line": 498, "name": "run_hierarchy_benchmark", "signature": "def run_hierarchy_benchmark(model_apex, model_blind, device)"}, {"kind": "method", "line": 544, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 92, "name": "__init__", "signature": "def __init__(self, num_patches, embed_dim)"}, {"kind": "method", "line": 101, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 109, "name": "__init__", "signature": "def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)"}, {"kind": "method", "line": 124, "name": "freeze", "signature": "def freeze(self)"}, {"kind": "method", "line": 128, "name": "unfreeze_mixer_only", "signature": "def unfreeze_mixer_only(self)"}, {"kind": "method", "line": 134, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 147, "name": "__init__", "signature": "def __init__(self, input_dim, hidden_dim, num_classes)"}, {"kind": "method", "line": 161, "name": "apply_masks", "signature": "def apply_masks(self)"}, {"kind": "method", "line": 167, "name": "get_sparsity", "signature": "def get_sparsity(self)"}, {"kind": "method", "line": 172, "name": "forward", "signature": "def forward(self, x)"}, {"doc": "L_mon: Legacy reporting metric.", "kind": "method", "line": 183, "name": "compute_metrics", "signature": "def compute_metrics(self, weight)"}, {"kind": "method", "line": 196, "name": "detect_phase_state", "signature": "def detect_phase_state(self, ratio_history)"}, {"doc": "Calculates Topo_R = L_opt / L_mon.\nL_opt is the active optimization energy (FC1 + Mixer).\nL_mon is the passive structural metric.", "kind": "method", "line": 208, "name": "compute_topology_ratio", "signature": "def compute_topology_ratio(self, model, extractor, chain_type)"}, {"kind": "method", "line": 229, "name": "__init__", "signature": "def __init__(self)"}, {"doc": "Decides if intervention is needed based on Phase and Performance.", "kind": "method", "line": 232, "name": "check_intervention", "signature": "def check_intervention(self, phase_state, coarse_acc, extractor)"}, {"doc": "v15.5: Targeted Spectral Surgery.\nInjects noise in the nullspace of the dominant spectral subspace to explore\nlatent modes without destroying learned hierarchy.", "kind": "method", "line": 250, "name": "perturb_mixer_targeted", "signature": "def perturb_mixer_targeted(self, extractor)"}, {"kind": "method", "line": 307, "name": "__init__", "signature": "def __init__(self, device, extractor_apex, extractor_blind)"}, {"kind": "method", "line": 315, "name": "_preprocess_batch", "signature": "def _preprocess_batch(self, x, extractor)"}, {"kind": "method", "line": 319, "name": "get_curriculum_dataset", "signature": "def get_curriculum_dataset(self, cycle)"}, {"kind": "method", "line": 325, "name": "train_single_chain", "signature": "def train_single_chain(self, model, cycle, chain_type)"}, {"kind": "method", "line": 494, "name": "__getitem__", "signature": "def __getitem__(self, index)"}, {"kind": "method", "line": 510, "name": "evaluate", "signature": "def evaluate(model, extractor, name)"}]}, {"doc": "NeuroSovereign v15.5: Targeted Phase Surgery (FINAL RELEASE) Objective: Break Hierarchy Ceiling via Orthogonal Spectral Projection. Fixes Applied: 1. Geometric Intervention Logic (Topo-R vs CV mismatch detection). 2. Implementation of compute_topology_ratio for Phase Monitoring. 3. Geometric fix in Orthogonal Projection (Input Space Injection). 4. Robust State Management (Migration from v15.4).", "id": "apex24.py", "kind": "module", "label": "apex24.py", "language": "py", "sha256": "28cc8f8e1932e048", "symbol_count": 32, "symbols": [{"doc": "L_opt: Computes the discrepancy between spectral entropy and effective rank.", "kind": "function", "line": 75, "name": "compute_spectral_loss", "signature": "def compute_spectral_loss(W)"}, {"kind": "class", "line": 92, "name": "GatedTokenMixer", "signature": "class GatedTokenMixer(Module)"}, {"kind": "class", "line": 109, "name": "PatchFeatureExtractor", "signature": "class PatchFeatureExtractor(Module)"}, {"kind": "class", "line": 147, "name": "TaxonomicMLP", "signature": "class TaxonomicMLP(Module)"}, {"kind": "class", "line": 183, "name": "SpectralMonitor", "signature": "class SpectralMonitor"}, {"kind": "class", "line": 229, "name": "TopologyController", "signature": "class TopologyController"}, {"kind": "class", "line": 339, "name": "TaxonomicTrainer", "signature": "class TaxonomicTrainer"}, {"kind": "class", "line": 527, "name": "CoarseCIFAR100", "signature": "class CoarseCIFAR100(CIFAR100)"}, {"kind": "method", "line": 532, "name": "run_hierarchy_benchmark", "signature": "def run_hierarchy_benchmark(model_apex, model_blind, device)"}, {"kind": "method", "line": 578, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 93, "name": "__init__", "signature": "def __init__(self, num_patches, embed_dim)"}, {"kind": "method", "line": 102, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 110, "name": "__init__", "signature": "def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)"}, {"kind": "method", "line": 125, "name": "freeze", "signature": "def freeze(self)"}, {"kind": "method", "line": 129, "name": "unfreeze_mixer_only", "signature": "def unfreeze_mixer_only(self)"}, {"kind": "method", "line": 135, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 148, "name": "__init__", "signature": "def __init__(self, input_dim, hidden_dim, num_classes)"}, {"kind": "method", "line": 162, "name": "apply_masks", "signature": "def apply_masks(self)"}, {"kind": "method", "line": 168, "name": "get_sparsity", "signature": "def get_sparsity(self)"}, {"kind": "method", "line": 173, "name": "forward", "signature": "def forward(self, x)"}, {"doc": "L_mon: Legacy reporting metric.", "kind": "method", "line": 184, "name": "compute_metrics", "signature": "def compute_metrics(self, weight)"}, {"kind": "method", "line": 197, "name": "detect_phase_state", "signature": "def detect_phase_state(self, ratio_history)"}, {"doc": "Calculates Topo_R = L_opt / L_mon.\nL_opt is the active optimization energy (FC1 + Mixer).\nL_mon is the passive structural metric.", "kind": "method", "line": 209, "name": "compute_topology_ratio", "signature": "def compute_topology_ratio(self, model, extractor, chain_type)"}, {"kind": "method", "line": 230, "name": "__init__", "signature": "def __init__(self)"}, {"doc": "v15.5 Final: Geometric Mismatch Detection.\nTriggers intervention if Topo_R (Structure) changes but Coarse Acc (Semantics) does not.", "kind": "method", "line": 235, "name": "check_intervention", "signature": "def check_intervention(self, phase_state, coarse_acc, extractor, current_topo_r)"}, {"doc": "v15.5: Targeted Spectral Surgery.\nInjects noise in the nullspace of the dominant spectral subspace to explore\nlatent modes without destroying learned hierarchy.", "kind": "method", "line": 284, "name": "perturb_mixer_targeted", "signature": "def perturb_mixer_targeted(self, extractor)"}, {"kind": "method", "line": 340, "name": "__init__", "signature": "def __init__(self, device, extractor_apex, extractor_blind)"}, {"kind": "method", "line": 348, "name": "_preprocess_batch", "signature": "def _preprocess_batch(self, x, extractor)"}, {"kind": "method", "line": 352, "name": "get_curriculum_dataset", "signature": "def get_curriculum_dataset(self, cycle)"}, {"kind": "method", "line": 358, "name": "train_single_chain", "signature": "def train_single_chain(self, model, cycle, chain_type)"}, {"kind": "method", "line": 528, "name": "__getitem__", "signature": "def __getitem__(self, index)"}, {"kind": "method", "line": 544, "name": "evaluate", "signature": "def evaluate(model, extractor, name)"}]}, {"doc": "NeuroSovereign v15.5: Targeted Phase Surgery (FINAL RELEASE) Objective: Break Hierarchy Ceiling via Orthogonal Spectral Projection. Status: Validated for High-Plasticity Init & Nullspace Surgery. Key Features: 1. Geometric Mismatch Detection (d(Topo_R) vs d(C.V)). 2. Orthogonal Noise Injection (Nullspace Surgery). 3. Topology Ratio (L_opt / L_mon) for Phase Monitoring.", "id": "apex25.py", "kind": "module", "label": "apex25.py", "language": "py", "sha256": "9a54292bc06626c8", "symbol_count": 32, "symbols": [{"doc": "L_opt: Computes the discrepancy between spectral entropy and effective rank.", "kind": "function", "line": 75, "name": "compute_spectral_loss", "signature": "def compute_spectral_loss(W)"}, {"kind": "class", "line": 89, "name": "GatedTokenMixer", "signature": "class GatedTokenMixer(Module)"}, {"kind": "class", "line": 106, "name": "PatchFeatureExtractor", "signature": "class PatchFeatureExtractor(Module)"}, {"kind": "class", "line": 144, "name": "TaxonomicMLP", "signature": "class TaxonomicMLP(Module)"}, {"kind": "class", "line": 180, "name": "SpectralMonitor", "signature": "class SpectralMonitor"}, {"kind": "class", "line": 224, "name": "TopologyController", "signature": "class TopologyController"}, {"kind": "class", "line": 316, "name": "TaxonomicTrainer", "signature": "class TaxonomicTrainer"}, {"kind": "class", "line": 504, "name": "CoarseCIFAR100", "signature": "class CoarseCIFAR100(CIFAR100)"}, {"kind": "method", "line": 509, "name": "run_hierarchy_benchmark", "signature": "def run_hierarchy_benchmark(model_apex, model_blind, device)"}, {"kind": "method", "line": 555, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 90, "name": "__init__", "signature": "def __init__(self, num_patches, embed_dim)"}, {"kind": "method", "line": 99, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 107, "name": "__init__", "signature": "def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)"}, {"kind": "method", "line": 122, "name": "freeze", "signature": "def freeze(self)"}, {"kind": "method", "line": 126, "name": "unfreeze_mixer_only", "signature": "def unfreeze_mixer_only(self)"}, {"kind": "method", "line": 132, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 145, "name": "__init__", "signature": "def __init__(self, input_dim, hidden_dim, num_classes)"}, {"kind": "method", "line": 159, "name": "apply_masks", "signature": "def apply_masks(self)"}, {"kind": "method", "line": 165, "name": "get_sparsity", "signature": "def get_sparsity(self)"}, {"kind": "method", "line": 170, "name": "forward", "signature": "def forward(self, x)"}, {"doc": "L_mon: Legacy reporting metric.", "kind": "method", "line": 181, "name": "compute_metrics", "signature": "def compute_metrics(self, weight)"}, {"kind": "method", "line": 194, "name": "detect_phase_state", "signature": "def detect_phase_state(self, ratio_history)"}, {"doc": "Calculates Topo_R = L_opt / L_mon.", "kind": "method", "line": 206, "name": "compute_topology_ratio", "signature": "def compute_topology_ratio(self, model, extractor, chain_type)"}, {"kind": "method", "line": 225, "name": "__init__", "signature": "def __init__(self)"}, {"doc": "v15.5 Final: Geometric Mismatch Detection.\nTriggers intervention if Topo_R (Structure) changes but Coarse Acc (Semantics) does not.", "kind": "method", "line": 230, "name": "check_intervention", "signature": "def check_intervention(self, phase_state, coarse_acc, extractor, current_topo_r)"}, {"doc": "v15.5: Targeted Spectral Surgery.\nInjects noise in the nullspace of the dominant spectral subspace.", "kind": "method", "line": 277, "name": "perturb_mixer_targeted", "signature": "def perturb_mixer_targeted(self, extractor)"}, {"kind": "method", "line": 317, "name": "__init__", "signature": "def __init__(self, device, extractor_apex, extractor_blind)"}, {"kind": "method", "line": 325, "name": "_preprocess_batch", "signature": "def _preprocess_batch(self, x, extractor)"}, {"kind": "method", "line": 329, "name": "get_curriculum_dataset", "signature": "def get_curriculum_dataset(self, cycle)"}, {"kind": "method", "line": 335, "name": "train_single_chain", "signature": "def train_single_chain(self, model, cycle, chain_type)"}, {"kind": "method", "line": 505, "name": "__getitem__", "signature": "def __getitem__(self, index)"}, {"kind": "method", "line": 521, "name": "evaluate", "signature": "def evaluate(model, extractor, name)"}]}, {"doc": "NeuroSovereign v17.0: Evolutionary Taxonomic Optimization Production-ready implementation with configurable iterations and statistical validation  Key improvements over v15.5: 1. Configurable evolutionary iterations (default: 20) 2. Statistical validation with 5 seeds per configuration 3. Early stopping with adaptive patience 4. Rigorous benchmarking against 3 baselines 5. Production-ready logging and checkpointing 6. Memory optimization for large-scale training 7. Complete reproducibility with fixed seeds 8. Targeted Spectral Surgery with Nullspace Injection  Outputs: - evolutionary_results.csv: Complete metrics across iterations - taxonomic_report.json: Statistical summary with confidence intervals - best_model_apex.pth / best_model_blind.pth: Final evolved models - evolution_curves.png: Publication-quality visualization", "id": "apex26.py", "kind": "module", "label": "apex26.py", "language": "py", "sha256": "8a1fcfc3b4d299d0", "symbol_count": 42, "symbols": [{"doc": "Ensure full reproducibility across runs", "kind": "function", "line": 43, "name": "set_seed", "signature": "def set_seed(seed)"}, {"doc": "Efficient token mixer with gating mechanism for feature interaction", "kind": "class", "line": 89, "name": "GatedTokenMixer", "signature": "class GatedTokenMixer(Module)"}, {"doc": "Efficient patch-based feature extractor with optional token mixing", "kind": "class", "line": 146, "name": "PatchFeatureExtractor", "signature": "class PatchFeatureExtractor(Module)"}, {"doc": "Sparse MLP with taxonomic heads for hierarchical learning", "kind": "class", "line": 207, "name": "TaxonomicMLP", "signature": "class TaxonomicMLP(Module)"}, {"doc": "Optimization Objective for Spectral Control (L_opt)", "kind": "method", "line": 265, "name": "compute_spectral_loss", "signature": "def compute_spectral_loss(W)"}, {"doc": "Monitor spectral properties of weight matrices for evolutionary guidance", "kind": "class", "line": 281, "name": "SpectralMonitor", "signature": "class SpectralMonitor"}, {"doc": "Advanced controller for targeted spectral surgery", "kind": "class", "line": 302, "name": "TopologyController", "signature": "class TopologyController"}, {"doc": "Engine for evolving neural networks through spectral refinement", "kind": "class", "line": 410, "name": "EvolutionaryEngine", "signature": "class EvolutionaryEngine"}, {"doc": "Wrapper that converts CIFAR100 into a pure 20-class classification problem (Superclasses).\nUsed to validate learned inductive bias.", "kind": "class", "line": 480, "name": "CoarseCIFAR100", "signature": "class CoarseCIFAR100(CIFAR100)"}, {"doc": "Run hierarchy stress test to validate inductive bias transfer", "kind": "method", "line": 490, "name": "run_hierarchy_benchmark", "signature": "def run_hierarchy_benchmark(model_apex, model_blind, feature_extractor_apex, feature_extractor_blind, device)"}, {"doc": "Framework for evolutionary training with statistical validation", "kind": "class", "line": 544, "name": "EvolutionaryTrainer", "signature": "class EvolutionaryTrainer"}, {"kind": "method", "line": 1170, "name": "parse_args", "signature": "def parse_args()"}, {"doc": "Main execution function", "kind": "method", "line": 1181, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 91, "name": "__init__", "signature": "def __init__(self, num_patches, embed_dim)"}, {"doc": "Initialize weights for stable training", "kind": "method", "line": 114, "name": "_init_weights", "signature": "def _init_weights(self)"}, {"doc": "Input:  [B, num_patches, embed_dim]\nOutput: [B, num_patches, embed_dim]", "kind": "method", "line": 128, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 148, "name": "__init__", "signature": "def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)"}, {"doc": "Freeze all parameters for transfer learning", "kind": "method", "line": 177, "name": "freeze", "signature": "def freeze(self)"}, {"doc": "Unfreeze only the mixer parameters for fine-tuning", "kind": "method", "line": 183, "name": "unfreeze_mixer_only", "signature": "def unfreeze_mixer_only(self)"}, {"doc": "Input:  [B, C, H, W]\nOutput: [B, embed_dim]", "kind": "method", "line": 190, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 209, "name": "__init__", "signature": "def __init__(self, input_dim, hidden_dim, num_classes, num_superclasses)"}, {"doc": "Apply sparsity masks to weights", "kind": "method", "line": 232, "name": "apply_masks", "signature": "def apply_masks(self)"}, {"doc": "Calculate overall sparsity percentage", "kind": "method", "line": 239, "name": "get_sparsity", "signature": "def get_sparsity(self)"}, {"doc": "Input:  [B, input_dim]\nOutput: ([B, num_classes], [B, num_superclasses])", "kind": "method", "line": 245, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 283, "name": "__init__", "signature": "def __init__(self, epsilon)"}, {"doc": "Compute spectral coherence metrics", "kind": "method", "line": 286, "name": "compute_metrics", "signature": "def compute_metrics(self, weight)"}, {"kind": "method", "line": 304, "name": "__init__", "signature": "def __init__(self, target_coarse_v, stagnation_limit, mixer_noise_scale, dominant_energy_threshold)"}, {"doc": "Detect phase state based on topology ratio history", "kind": "method", "line": 314, "name": "detect_phase_state", "signature": "def detect_phase_state(self, ratio_history, phase_window, phase_std_dev_limit)"}, {"doc": "Check if intervention is needed based on geometric mismatch detection", "kind": "method", "line": 332, "name": "check_intervention", "signature": "def check_intervention(self, phase_state, coarse_acc, extractor, current_topo_r, geo_window)"}, {"doc": "Targeted Spectral Surgery: Inject noise in the nullspace of dominant subspace", "kind": "method", "line": 377, "name": "perturb_mixer_targeted", "signature": "def perturb_mixer_targeted(self, extractor)"}, {"kind": "method", "line": 412, "name": "__init__", "signature": "def __init__(self, device, target_L)"}, {"doc": "Apply rank capping shock to prevent over-specialization", "kind": "method", "line": 417, "name": "apply_rank_capping", "signature": "def apply_rank_capping(self, model, layer_name, keep_ratio)"}, {"doc": "Create refined offspring through gradient-based inheritance", "kind": "method", "line": 434, "name": "create_offspring", "signature": "def create_offspring(self, parent_state, data_loader, feature_extractor, lambda_taxonomic, learning_rate)"}, {"kind": "method", "line": 485, "name": "__getitem__", "signature": "def __getitem__(self, index)"}, {"kind": "method", "line": 503, "name": "evaluate", "signature": "def evaluate(model, extractor, name)"}, {"kind": "method", "line": 546, "name": "__init__", "signature": "def __init__(self, device, output_dir)"}, {"doc": "Load curriculum dataset based on evolutionary cycle", "kind": "method", "line": 572, "name": "load_data", "signature": "def load_data(self, cycle, batch_size)"}, {"doc": "Train model with evolutionary pressure and hierarchical learning", "kind": "method", "line": 614, "name": "train_model", "signature": "def train_model(self, model, cycle, chain_type, feature_extractor)"}, {"doc": "Calculates Topo_R = L_opt / L_mon.", "kind": "method", "line": 837, "name": "compute_topology_ratio", "signature": "def compute_topology_ratio(self, model, extractor, chain_type)"}, {"doc": "Run full evolutionary experiment with statistical validation", "kind": "method", "line": 855, "name": "run_evolution", "signature": "def run_evolution(self, num_iterations, num_seeds, early_stop_patience)"}, {"doc": "Save results to files", "kind": "method", "line": 1027, "name": "_save_results", "signature": "def _save_results(self, all_results, best_overall, hierarchy_delta)"}, {"doc": "Create publication-quality plots", "kind": "method", "line": 1081, "name": "_plot_results", "signature": "def _plot_results(self, all_results)"}]}, {"doc": "NeuroSovereign v18.0: Adaptive Iterative Spectral Refinement Production-ready implementation addressing v17.0 technical review.  Key improvements over v17.0 (Review Implementation): 1. Replaced fixed threshold with DynamicThresholdController (Percentile-based). 2. Renamed \"Evolutionary\" to \"Iterative Refinement\" for scientific accuracy. 3. Added Singular Value tracking and visualization (Spectral Map). 4. Added Ablation flags via CLI for rigorous validation. 5. Improved Nullspace Surgery logging for transparency.  Scientific Goal: \"Demonstrate controlled induction of hierarchical structure beyond symmetric spectral regularization.\"", "id": "apex27.py", "kind": "module", "label": "apex27.py", "language": "py", "sha256": "a492b861ee25e869", "symbol_count": 40, "symbols": [{"doc": "Ensure full reproducibility across runs", "kind": "function", "line": 38, "name": "set_seed", "signature": "def set_seed(seed)"}, {"doc": "Efficient token mixer with gating mechanism", "kind": "class", "line": 83, "name": "GatedTokenMixer", "signature": "class GatedTokenMixer(Module)"}, {"doc": "Efficient patch-based feature extractor", "kind": "class", "line": 124, "name": "PatchFeatureExtractor", "signature": "class PatchFeatureExtractor(Module)"}, {"doc": "Sparse MLP with taxonomic heads", "kind": "class", "line": 170, "name": "TaxonomicMLP", "signature": "class TaxonomicMLP(Module)"}, {"doc": "Optimization Objective for Spectral Control (L_opt)", "kind": "method", "line": 217, "name": "compute_spectral_loss", "signature": "def compute_spectral_loss(W)"}, {"doc": "v18 Improvement: Replaces fixed TARGET_COARSE_V with adaptive logic.\nTriggers intervention if current performance stagnates relative to its own history.", "kind": "class", "line": 233, "name": "DynamicThresholdController", "signature": "class DynamicThresholdController"}, {"doc": "Monitor spectral properties", "kind": "class", "line": 255, "name": "SpectralMonitor", "signature": "class SpectralMonitor"}, {"doc": "v18 Improvement: Advanced controller with Adaptive Thresholding.\nImplements Targeted Spectral Surgery with Nullspace Injection.", "kind": "class", "line": 275, "name": "TopologyController", "signature": "class TopologyController"}, {"doc": "v18: Engine for iterative refinement (formerly Evolutionary)", "kind": "class", "line": 369, "name": "IterativeRefinementEngine", "signature": "class IterativeRefinementEngine"}, {"doc": "Framework for Iterative Refinement with v18 Adaptive Control", "kind": "class", "line": 416, "name": "IterativeTrainer", "signature": "class IterativeTrainer"}, {"kind": "method", "line": 900, "name": "parse_args", "signature": "def parse_args()"}, {"kind": "method", "line": 914, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 85, "name": "__init__", "signature": "def __init__(self, num_patches, embed_dim)"}, {"kind": "method", "line": 104, "name": "_init_weights", "signature": "def _init_weights(self)"}, {"kind": "method", "line": 116, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 126, "name": "__init__", "signature": "def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)"}, {"kind": "method", "line": 151, "name": "freeze", "signature": "def freeze(self)"}, {"kind": "method", "line": 156, "name": "unfreeze_mixer_only", "signature": "def unfreeze_mixer_only(self)"}, {"kind": "method", "line": 162, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 172, "name": "__init__", "signature": "def __init__(self, input_dim, hidden_dim, num_classes, num_superclasses)"}, {"kind": "method", "line": 192, "name": "apply_masks", "signature": "def apply_masks(self)"}, {"kind": "method", "line": 198, "name": "get_sparsity", "signature": "def get_sparsity(self)"}, {"kind": "method", "line": 203, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 238, "name": "__init__", "signature": "def __init__(self, window_size, percentile_trigger)"}, {"kind": "method", "line": 243, "name": "update", "signature": "def update(self, value)"}, {"kind": "method", "line": 246, "name": "is_stagnant", "signature": "def is_stagnant(self, current_val)"}, {"kind": "method", "line": 257, "name": "__init__", "signature": "def __init__(self, epsilon)"}, {"kind": "method", "line": 260, "name": "compute_metrics", "signature": "def compute_metrics(self, weight)"}, {"kind": "method", "line": 280, "name": "__init__", "signature": "def __init__(self, dynamic_threshold, stagnation_limit, mixer_noise_scale, dominant_energy_threshold, enable_surgery)"}, {"kind": "method", "line": 291, "name": "check_intervention", "signature": "def check_intervention(self, coarse_acc, extractor, current_topo_r, geo_window, alpha)"}, {"doc": "Targeted Spectral Surgery: Inject noise in the nullspace of dominant subspace", "kind": "method", "line": 336, "name": "perturb_mixer_targeted", "signature": "def perturb_mixer_targeted(self, extractor)"}, {"kind": "method", "line": 371, "name": "__init__", "signature": "def __init__(self, device)"}, {"doc": "Create refined offspring through gradient-based inheritance", "kind": "method", "line": 374, "name": "create_offspring", "signature": "def create_offspring(self, parent_state, data_loader, feature_extractor, lambda_taxonomic, learning_rate)"}, {"kind": "method", "line": 418, "name": "__init__", "signature": "def __init__(self, device, output_dir, enable_surgery, enable_taxonomy)"}, {"kind": "method", "line": 441, "name": "load_data", "signature": "def load_data(self, cycle, batch_size)"}, {"kind": "method", "line": 479, "name": "train_model", "signature": "def train_model(self, model, cycle, chain_type, feature_extractor)"}, {"kind": "method", "line": 683, "name": "compute_topology_ratio", "signature": "def compute_topology_ratio(self, model, extractor, chain_type)"}, {"kind": "method", "line": 696, "name": "run_refinement", "signature": "def run_refinement(self, num_iterations, num_seeds, early_stop_patience)"}, {"kind": "method", "line": 809, "name": "_save_results", "signature": "def _save_results(self, all_results)"}, {"kind": "method", "line": 815, "name": "_plot_results_v18", "signature": "def _plot_results_v18(self, all_results)"}]}, {"doc": "NeuroSovereign v18.0: Iterative Spectral Refinement with Adaptive Control Addressing all senior reviewer concerns from v17:  1. REPLACED fixed TARGET_COARSE_V with adaptive semantic-plasticity ratio 2. RENAMED \"evolutionary\" to \"iterative refinement\" throughout 3. REFACTORED topology controller for dataset-agnostic operation 4. ADDED singular value visualization for paper figures 5. IMPLEMENTED ablation-ready architecture variants 6. OPTIMIZED SVD operations for production scalability  This implementation delivers: - Statistically validated hierarchical advantage (1.0%+ over symmetric baseline) - Self-referential control without dataset-specific thresholds - Publication-ready visualizations of spectral dynamics - Production-grade reproducibility and checkpointing  Outputs: - refinement_results.csv: Complete metrics across iterations - spectral_analysis/ directory: Singular value visualizations - taxonomic_report.json: Statistical summary with confidence intervals - best_model_apex.pth / best_model_blind.pth: Final refined models - refinement_curves.png: Publication-quality visualization", "id": "apex28.py", "kind": "module", "label": "apex28.py", "language": "py", "sha256": "3c9c96d3e1f60171", "symbol_count": 47, "symbols": [{"doc": "Ensure full reproducibility across runs", "kind": "function", "line": 47, "name": "set_seed", "signature": "def set_seed(seed)"}, {"doc": "Efficient token mixer with gating mechanism for feature interaction", "kind": "class", "line": 93, "name": "GatedTokenMixer", "signature": "class GatedTokenMixer(Module)"}, {"doc": "Efficient patch-based feature extractor with optional token mixing", "kind": "class", "line": 150, "name": "PatchFeatureExtractor", "signature": "class PatchFeatureExtractor(Module)"}, {"doc": "Sparse MLP with taxonomic heads for hierarchical learning", "kind": "class", "line": 211, "name": "TaxonomicMLP", "signature": "class TaxonomicMLP(Module)"}, {"doc": "Optimization Objective for Spectral Control (L_opt)", "kind": "method", "line": 269, "name": "compute_spectral_loss", "signature": "def compute_spectral_loss(W)"}, {"doc": "Monitor spectral properties with adaptive analysis", "kind": "class", "line": 285, "name": "SpectralMonitor", "signature": "class SpectralMonitor"}, {"doc": "Self-referential controller using semantic-plasticity ratio", "kind": "class", "line": 316, "name": "AdaptiveTopologyController", "signature": "class AdaptiveTopologyController"}, {"doc": "Engine for iterative refinement through spectral control", "kind": "class", "line": 415, "name": "IterativeRefinementEngine", "signature": "class IterativeRefinementEngine"}, {"doc": "Wrapper that converts CIFAR100 into a pure 20-class classification problem (Superclasses).\nUsed to validate learned inductive bias.", "kind": "class", "line": 484, "name": "CoarseCIFAR100", "signature": "class CoarseCIFAR100(CIFAR100)"}, {"doc": "Run hierarchy stress test to validate inductive bias transfer", "kind": "method", "line": 494, "name": "run_hierarchy_benchmark", "signature": "def run_hierarchy_benchmark(model_apex, model_blind, feature_extractor_apex, feature_extractor_blind, device)"}, {"doc": "Generate publication-quality singular value visualizations", "kind": "method", "line": 548, "name": "visualize_singular_values", "signature": "def visualize_singular_values(model_apex, model_blind, feature_extractor_apex, feature_extractor_blind, device, output_dir)"}, {"doc": "Framework for iterative refinement with statistical validation", "kind": "class", "line": 611, "name": "IterativeRefinementTrainer", "signature": "class IterativeRefinementTrainer"}, {"kind": "method", "line": 1270, "name": "parse_args", "signature": "def parse_args()"}, {"doc": "Main execution function", "kind": "method", "line": 1282, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 95, "name": "__init__", "signature": "def __init__(self, num_patches, embed_dim)"}, {"doc": "Initialize weights for stable training", "kind": "method", "line": 118, "name": "_init_weights", "signature": "def _init_weights(self)"}, {"doc": "Input:  [B, num_patches, embed_dim]\nOutput: [B, num_patches, embed_dim]", "kind": "method", "line": 132, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 152, "name": "__init__", "signature": "def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)"}, {"doc": "Freeze all parameters for transfer learning", "kind": "method", "line": 181, "name": "freeze", "signature": "def freeze(self)"}, {"doc": "Unfreeze only the mixer parameters for fine-tuning", "kind": "method", "line": 187, "name": "unfreeze_mixer_only", "signature": "def unfreeze_mixer_only(self)"}, {"doc": "Input:  [B, C, H, W]\nOutput: [B, embed_dim]", "kind": "method", "line": 194, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 213, "name": "__init__", "signature": "def __init__(self, input_dim, hidden_dim, num_classes, num_superclasses)"}, {"doc": "Apply sparsity masks to weights", "kind": "method", "line": 236, "name": "apply_masks", "signature": "def apply_masks(self)"}, {"doc": "Calculate overall sparsity percentage", "kind": "method", "line": 243, "name": "get_sparsity", "signature": "def get_sparsity(self)"}, {"doc": "Input:  [B, input_dim]\nOutput: ([B, num_classes], [B, num_superclasses])", "kind": "method", "line": 249, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 287, "name": "__init__", "signature": "def __init__(self, epsilon)"}, {"doc": "Compute spectral coherence metrics", "kind": "method", "line": 290, "name": "compute_metrics", "signature": "def compute_metrics(self, weight)"}, {"doc": "Get singular values for visualization", "kind": "method", "line": 306, "name": "get_singular_values", "signature": "def get_singular_values(self, weight)"}, {"kind": "method", "line": 318, "name": "__init__", "signature": "def __init__(self, semantic_plasticity_threshold, stagnation_limit, mixer_noise_scale, dominant_energy_threshold, geo_window)"}, {"doc": "Compute ratio of semantic gain to structural change", "kind": "method", "line": 330, "name": "compute_semantic_plasticity_ratio", "signature": "def compute_semantic_plasticity_ratio(self)"}, {"doc": "Determine if intervention is needed using adaptive criteria", "kind": "method", "line": 343, "name": "detect_intervention_need", "signature": "def detect_intervention_need(self, phase_state, extractor)"}, {"doc": "Update history for adaptive control", "kind": "method", "line": 365, "name": "update_history", "signature": "def update_history(self, topo_ratio, coarse_acc)"}, {"doc": "Targeted Spectral Surgery: Inject noise in the nullspace of dominant subspace", "kind": "method", "line": 382, "name": "perturb_mixer_targeted", "signature": "def perturb_mixer_targeted(self, extractor)"}, {"kind": "method", "line": 417, "name": "__init__", "signature": "def __init__(self, device)"}, {"doc": "Apply rank capping shock to prevent over-specialization", "kind": "method", "line": 421, "name": "apply_rank_capping", "signature": "def apply_rank_capping(self, model, layer_name, keep_ratio)"}, {"doc": "Create refined model through gradient-based inheritance", "kind": "method", "line": 438, "name": "create_refined_model", "signature": "def create_refined_model(self, parent_state, data_loader, feature_extractor, lambda_taxonomic, learning_rate)"}, {"kind": "method", "line": 489, "name": "__getitem__", "signature": "def __getitem__(self, index)"}, {"kind": "method", "line": 507, "name": "evaluate", "signature": "def evaluate(model, extractor, name)"}, {"doc": "Get singular values from model weights", "kind": "method", "line": 556, "name": "get_singular_values", "signature": "def get_singular_values(model, extractor, name)"}, {"kind": "method", "line": 613, "name": "__init__", "signature": "def __init__(self, device, output_dir)"}, {"doc": "Load curriculum dataset based on refinement cycle", "kind": "method", "line": 639, "name": "load_data", "signature": "def load_data(self, cycle, batch_size)"}, {"doc": "Train model with iterative refinement and hierarchical learning", "kind": "method", "line": 681, "name": "train_model", "signature": "def train_model(self, model, cycle, chain_type, feature_extractor)"}, {"doc": "Detect phase state based on topology ratio history", "kind": "method", "line": 908, "name": "detect_phase_state", "signature": "def detect_phase_state(self, ratio_history, phase_window, phase_std_dev_limit)"}, {"doc": "Calculates Topo_R = L_opt / L_mon.", "kind": "method", "line": 926, "name": "compute_topology_ratio", "signature": "def compute_topology_ratio(self, model, extractor, chain_type)"}, {"doc": "Run full iterative refinement experiment with statistical validation", "kind": "method", "line": 944, "name": "run_refinement", "signature": "def run_refinement(self, num_iterations, num_seeds, early_stop_patience)"}, {"doc": "Save results to files", "kind": "method", "line": 1126, "name": "_save_results", "signature": "def _save_results(self, all_results, best_overall, hierarchy_delta)"}, {"doc": "Create publication-quality plots", "kind": "method", "line": 1181, "name": "_plot_results", "signature": "def _plot_results(self, all_results)"}]}, {"doc": "NeuroSovereign v18.0: Iterative Spectral Refinement with Adaptive Control Addressing all senior reviewer concerns from v17: 1. REPLACED fixed TARGET_COARSE_V with adaptive semantic-plasticity ratio 2. RENAMED \"evolutionary\" to \"iterative refinement\" throughout 3. REFACTORED topology controller for dataset-agnostic operation 4. ADDED singular value visualization for paper figures 5. IMPLEMENTED ablation-ready architecture variants 6. OPTIMIZED SVD operations for production scalability This implementation delivers: - Statistically validated hierarchical advantage (1.0%+ over symmetric baseline) - Self-referential control without dataset-specific thresholds - Publication-ready visualizations of spectral dynamics - Production-grade reproducibility and checkpointing Outputs: - refinement_results.csv: Complete metrics across iterations - spectral_analysis/ directory: Singular value visualizations - taxonomic_report.json: Statistical summary with confidence intervals - best_model_apex.pth / best_model_blind.pth: Final refined models - refinement_curves.png: Publication-quality visualization", "id": "apex29.py", "kind": "module", "label": "apex29.py", "language": "py", "sha256": "527de683706ade66", "symbol_count": 47, "symbols": [{"doc": "Ensure full reproducibility across runs", "kind": "function", "line": 43, "name": "set_seed", "signature": "def set_seed(seed)"}, {"doc": "Efficient token mixer with gating mechanism for feature interaction", "kind": "class", "line": 86, "name": "GatedTokenMixer", "signature": "class GatedTokenMixer(Module)"}, {"doc": "Efficient patch-based feature extractor with optional token mixing", "kind": "class", "line": 143, "name": "PatchFeatureExtractor", "signature": "class PatchFeatureExtractor(Module)"}, {"doc": "Sparse MLP with taxonomic heads for hierarchical learning", "kind": "class", "line": 204, "name": "TaxonomicMLP", "signature": "class TaxonomicMLP(Module)"}, {"doc": "Optimization Objective for Spectral Control (L_opt)", "kind": "method", "line": 262, "name": "compute_spectral_loss", "signature": "def compute_spectral_loss(W)"}, {"doc": "Monitor spectral properties with adaptive analysis", "kind": "class", "line": 278, "name": "SpectralMonitor", "signature": "class SpectralMonitor"}, {"doc": "Self-referential controller using semantic-plasticity ratio", "kind": "class", "line": 309, "name": "AdaptiveTopologyController", "signature": "class AdaptiveTopologyController"}, {"doc": "Engine for iterative refinement through spectral control", "kind": "class", "line": 408, "name": "IterativeRefinementEngine", "signature": "class IterativeRefinementEngine"}, {"doc": "Wrapper that converts CIFAR100 into a pure 20-class classification problem (Superclasses).\nUsed to validate learned inductive bias.", "kind": "class", "line": 476, "name": "CoarseCIFAR100", "signature": "class CoarseCIFAR100(CIFAR100)"}, {"doc": "Run hierarchy stress test to validate inductive bias transfer", "kind": "method", "line": 486, "name": "run_hierarchy_benchmark", "signature": "def run_hierarchy_benchmark(model_apex, model_blind, feature_extractor_apex, feature_extractor_blind, device)"}, {"doc": "Generate publication-quality singular value visualizations", "kind": "method", "line": 543, "name": "visualize_singular_values", "signature": "def visualize_singular_values(model_apex, model_blind, feature_extractor_apex, feature_extractor_blind, device, output_dir)"}, {"doc": "Framework for iterative refinement with statistical validation", "kind": "class", "line": 606, "name": "IterativeRefinementTrainer", "signature": "class IterativeRefinementTrainer"}, {"kind": "method", "line": 1244, "name": "parse_args", "signature": "def parse_args()"}, {"doc": "Main execution function", "kind": "method", "line": 1255, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 88, "name": "__init__", "signature": "def __init__(self, num_patches, embed_dim)"}, {"doc": "Initialize weights for stable training", "kind": "method", "line": 111, "name": "_init_weights", "signature": "def _init_weights(self)"}, {"doc": "Input:  [B, num_patches, embed_dim]\nOutput: [B, num_patches, embed_dim]", "kind": "method", "line": 125, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 145, "name": "__init__", "signature": "def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)"}, {"doc": "Freeze all parameters for transfer learning", "kind": "method", "line": 174, "name": "freeze", "signature": "def freeze(self)"}, {"doc": "Unfreeze only the mixer parameters for fine-tuning", "kind": "method", "line": 180, "name": "unfreeze_mixer_only", "signature": "def unfreeze_mixer_only(self)"}, {"doc": "Input:  [B, C, H, W]\nOutput: [B, embed_dim]", "kind": "method", "line": 187, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 206, "name": "__init__", "signature": "def __init__(self, input_dim, hidden_dim, num_classes, num_superclasses)"}, {"doc": "Apply sparsity masks to weights", "kind": "method", "line": 229, "name": "apply_masks", "signature": "def apply_masks(self)"}, {"doc": "Calculate overall sparsity percentage", "kind": "method", "line": 236, "name": "get_sparsity", "signature": "def get_sparsity(self)"}, {"doc": "Input:  [B, input_dim]\nOutput: ([B, num_classes], [B, num_superclasses])", "kind": "method", "line": 242, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 280, "name": "__init__", "signature": "def __init__(self, epsilon)"}, {"doc": "Compute spectral coherence metrics", "kind": "method", "line": 283, "name": "compute_metrics", "signature": "def compute_metrics(self, weight)"}, {"doc": "Get singular values for visualization", "kind": "method", "line": 299, "name": "get_singular_values", "signature": "def get_singular_values(self, weight)"}, {"kind": "method", "line": 311, "name": "__init__", "signature": "def __init__(self, semantic_plasticity_threshold, stagnation_limit, mixer_noise_scale, dominant_energy_threshold, geo_window)"}, {"doc": "Compute ratio of semantic gain to structural change", "kind": "method", "line": 323, "name": "compute_semantic_plasticity_ratio", "signature": "def compute_semantic_plasticity_ratio(self)"}, {"doc": "Determine if intervention is needed using adaptive criteria", "kind": "method", "line": 336, "name": "detect_intervention_need", "signature": "def detect_intervention_need(self, phase_state, extractor)"}, {"doc": "Update history for adaptive control", "kind": "method", "line": 358, "name": "update_history", "signature": "def update_history(self, topo_ratio, coarse_acc)"}, {"doc": "Targeted Spectral Surgery: Inject noise in the nullspace of dominant subspace", "kind": "method", "line": 375, "name": "perturb_mixer_targeted", "signature": "def perturb_mixer_targeted(self, extractor)"}, {"kind": "method", "line": 410, "name": "__init__", "signature": "def __init__(self, device)"}, {"doc": "Apply rank capping shock to prevent over-specialization", "kind": "method", "line": 414, "name": "apply_rank_capping", "signature": "def apply_rank_capping(self, model, layer_name, keep_ratio)"}, {"doc": "Create refined model through gradient-based inheritance", "kind": "method", "line": 430, "name": "create_refined_model", "signature": "def create_refined_model(self, parent_state, data_loader, feature_extractor, lambda_taxonomic, learning_rate)"}, {"kind": "method", "line": 481, "name": "__getitem__", "signature": "def __getitem__(self, index)"}, {"kind": "method", "line": 499, "name": "evaluate", "signature": "def evaluate(model, extractor, name)"}, {"doc": "Get singular values from model weights", "kind": "method", "line": 551, "name": "get_singular_values", "signature": "def get_singular_values(model, extractor, name)"}, {"kind": "method", "line": 608, "name": "__init__", "signature": "def __init__(self, device, output_dir)"}, {"doc": "Load curriculum dataset based on refinement cycle", "kind": "method", "line": 634, "name": "load_data", "signature": "def load_data(self, cycle, batch_size)"}, {"doc": "Train model with iterative refinement and hierarchical learning", "kind": "method", "line": 674, "name": "train_model", "signature": "def train_model(self, model, cycle, chain_type, feature_extractor)"}, {"doc": "Detect phase state based on topology ratio history", "kind": "method", "line": 894, "name": "detect_phase_state", "signature": "def detect_phase_state(self, ratio_history, phase_window, phase_std_dev_limit)"}, {"doc": "Calculates Topo_R = L_opt / L_mon.", "kind": "method", "line": 912, "name": "compute_topology_ratio", "signature": "def compute_topology_ratio(self, model, extractor, chain_type)"}, {"doc": "Run full iterative refinement experiment with statistical validation", "kind": "method", "line": 931, "name": "run_refinement", "signature": "def run_refinement(self, num_iterations, num_seeds, early_stop_patience)"}, {"doc": "Save results to files", "kind": "method", "line": 1109, "name": "_save_results", "signature": "def _save_results(self, all_results, best_overall, hierarchy_delta)"}, {"doc": "Create publication-quality plots", "kind": "method", "line": 1164, "name": "_plot_results", "signature": "def _plot_results(self, all_results)"}]}, {"doc": "NeuroSovereign v18.1: Production-Grade Iterative Spectral Refinement Hardened Implementation based on Senior Review Feedback (v18.0 -> v18.1)  Key Corrections in v18.1: 1. [FIXED] Critical Scope Error: visualize_singular_values now accepts monitor object. 2. [FIXED] Plotting Logic: _plot_results explicitly handles best_overall dict. 3. [REFACTOR] Removed Hardcoded Logic: 28.0 is now REFERENCE_BASELINE only (not control flow). 4. [OPTIMIZED] Control Flow: Clarified hysteresis in AdaptiveTopologyController.  Validated Claims: - Statistically validated hierarchical advantage (>1.0% over baseline) - Zero-shot CIFAR-20 transfer validation - Dataset-agnostic control (no fixed coarse accuracy targets) - Ablation-ready architecture", "id": "apex30.py", "kind": "module", "label": "apex30.py", "language": "py", "sha256": "1f6e07bf7c0e26df", "symbol_count": 47, "symbols": [{"doc": "Ensure full reproducibility across runs", "kind": "function", "line": 43, "name": "set_seed", "signature": "def set_seed(seed)"}, {"doc": "Efficient token mixer with gating mechanism for feature interaction", "kind": "class", "line": 87, "name": "GatedTokenMixer", "signature": "class GatedTokenMixer(Module)"}, {"kind": "class", "line": 142, "name": "PatchFeatureExtractor", "signature": "class PatchFeatureExtractor(Module)"}, {"doc": "Sparse MLP with taxonomic heads for hierarchical learning", "kind": "class", "line": 185, "name": "TaxonomicMLP", "signature": "class TaxonomicMLP(Module)"}, {"doc": "Optimization Objective for Spectral Control (L_opt)", "kind": "method", "line": 232, "name": "compute_spectral_loss", "signature": "def compute_spectral_loss(W)"}, {"doc": "Monitor spectral properties with adaptive analysis", "kind": "class", "line": 248, "name": "SpectralMonitor", "signature": "class SpectralMonitor"}, {"doc": "Self-referential controller using semantic-plasticity ratio", "kind": "class", "line": 277, "name": "AdaptiveTopologyController", "signature": "class AdaptiveTopologyController"}, {"doc": "Engine for iterative refinement through spectral control", "kind": "class", "line": 397, "name": "IterativeRefinementEngine", "signature": "class IterativeRefinementEngine"}, {"kind": "class", "line": 459, "name": "CoarseCIFAR100", "signature": "class CoarseCIFAR100(CIFAR100)"}, {"kind": "method", "line": 464, "name": "run_hierarchy_benchmark", "signature": "def run_hierarchy_benchmark(model_apex, model_blind, feature_extractor_apex, feature_extractor_blind, device)"}, {"doc": "v18.1 FIX: Now accepts 'monitor' explicitly.\nGenerate publication-quality singular value visualizations.", "kind": "method", "line": 510, "name": "visualize_singular_values", "signature": "def visualize_singular_values(model_apex, model_blind, feature_extractor_apex, feature_extractor_blind, device, output_dir, monitor)"}, {"kind": "class", "line": 570, "name": "IterativeRefinementTrainer", "signature": "class IterativeRefinementTrainer"}, {"kind": "method", "line": 1062, "name": "parse_args", "signature": "def parse_args()"}, {"kind": "method", "line": 1071, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 89, "name": "__init__", "signature": "def __init__(self, num_patches, embed_dim)"}, {"kind": "method", "line": 111, "name": "_init_weights", "signature": "def _init_weights(self)"}, {"kind": "method", "line": 134, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 143, "name": "__init__", "signature": "def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)"}, {"kind": "method", "line": 164, "name": "freeze", "signature": "def freeze(self)"}, {"kind": "method", "line": 169, "name": "unfreeze_mixer_only", "signature": "def unfreeze_mixer_only(self)"}, {"kind": "method", "line": 175, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 187, "name": "__init__", "signature": "def __init__(self, input_dim, hidden_dim, num_classes, num_superclasses)"}, {"kind": "method", "line": 207, "name": "apply_masks", "signature": "def apply_masks(self)"}, {"kind": "method", "line": 213, "name": "get_sparsity", "signature": "def get_sparsity(self)"}, {"kind": "method", "line": 218, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 250, "name": "__init__", "signature": "def __init__(self, epsilon)"}, {"kind": "method", "line": 253, "name": "compute_metrics", "signature": "def compute_metrics(self, weight)"}, {"kind": "method", "line": 268, "name": "get_singular_values", "signature": "def get_singular_values(self, weight)"}, {"kind": "method", "line": 279, "name": "__init__", "signature": "def __init__(self, semantic_plasticity_threshold, stagnation_limit, mixer_noise_scale, dominant_energy_threshold, geo_window)"}, {"doc": "Compute ratio of semantic gain to structural change", "kind": "method", "line": 295, "name": "compute_semantic_plasticity_ratio", "signature": "def compute_semantic_plasticity_ratio(self)"}, {"doc": "Determine if intervention is needed using adaptive criteria.\nImplements hysteresis to avoid intervention during active SHIFTING phases.", "kind": "method", "line": 308, "name": "detect_intervention_need", "signature": "def detect_intervention_need(self, phase_state, extractor)"}, {"doc": "Update history for adaptive control", "kind": "method", "line": 348, "name": "update_history", "signature": "def update_history(self, topo_ratio, coarse_acc)"}, {"doc": "Targeted Spectral Surgery: Inject noise in the nullspace of dominant subspace", "kind": "method", "line": 363, "name": "perturb_mixer_targeted", "signature": "def perturb_mixer_targeted(self, extractor)"}, {"kind": "method", "line": 399, "name": "__init__", "signature": "def __init__(self, device)"}, {"kind": "method", "line": 403, "name": "apply_rank_capping", "signature": "def apply_rank_capping(self, model, layer_name, keep_ratio)"}, {"kind": "method", "line": 419, "name": "create_refined_model", "signature": "def create_refined_model(self, parent_state, data_loader, feature_extractor, lambda_taxonomic, learning_rate)"}, {"kind": "method", "line": 460, "name": "__getitem__", "signature": "def __getitem__(self, index)"}, {"kind": "method", "line": 471, "name": "evaluate", "signature": "def evaluate(model, extractor)"}, {"kind": "method", "line": 521, "name": "get_singular_values", "signature": "def get_singular_values(model, extractor)"}, {"kind": "method", "line": 571, "name": "__init__", "signature": "def __init__(self, device, output_dir)"}, {"kind": "method", "line": 593, "name": "load_data", "signature": "def load_data(self, cycle, batch_size)"}, {"kind": "method", "line": 617, "name": "train_model", "signature": "def train_model(self, model, cycle, chain_type, feature_extractor)"}, {"kind": "method", "line": 797, "name": "detect_phase_state", "signature": "def detect_phase_state(self, ratio_history, phase_window, phase_std_dev_limit)"}, {"kind": "method", "line": 813, "name": "compute_topology_ratio", "signature": "def compute_topology_ratio(self, model, extractor, chain_type)"}, {"kind": "method", "line": 828, "name": "run_refinement", "signature": "def run_refinement(self, num_iterations, num_seeds, early_stop_patience)"}, {"kind": "method", "line": 956, "name": "_save_results", "signature": "def _save_results(self, all_results, best_overall, hierarchy_delta)"}, {"doc": "v18.1 FIX: Explicitly accepts best_overall to fix scope bug.\nCreate publication-quality plots.", "kind": "method", "line": 983, "name": "_plot_results", "signature": "def _plot_results(self, all_results, best_overall)"}]}, {"doc": "NeuroSovereign v18.1: Production-Grade Iterative Spectral Refinement Hardened Implementation based on Senior Review Feedback (v18.0 -> v18.1)  Key Corrections in v18.1: 1. [FIXED] Critical Scope Error: visualize_singular_values now accepts monitor object. 2. [FIXED] Plotting Logic: _plot_results explicitly handles best_overall dict. 3. [REFACTOR] Removed Hardcoded Logic: 28.0 is now REFERENCE_BASELINE only (not control flow). 4. [OPTIMIZED] Control Flow: Clarified hysteresis in AdaptiveTopologyController. 5. [HARDENED] Updated torch.svd to torch.linalg.svd (future-proofing). 6. [HARDENED] Increased initialization std in GatedTokenMixer (v15.3 exploration logic).  Validated Claims: - Statistically validated hierarchical advantage (>1.0% over baseline) - Zero-shot CIFAR-20 transfer validation - Dataset-agnostic control (no fixed coarse accuracy targets) - Ablation-ready architecture", "id": "apex31.py", "kind": "module", "label": "apex31.py", "language": "py", "sha256": "d2fbf2104233d506", "symbol_count": 47, "symbols": [{"doc": "Ensure full reproducibility across runs", "kind": "function", "line": 45, "name": "set_seed", "signature": "def set_seed(seed)"}, {"doc": "Efficient token mixer with gating mechanism for feature interaction", "kind": "class", "line": 89, "name": "GatedTokenMixer", "signature": "class GatedTokenMixer(Module)"}, {"kind": "class", "line": 144, "name": "PatchFeatureExtractor", "signature": "class PatchFeatureExtractor(Module)"}, {"doc": "Sparse MLP with taxonomic heads for hierarchical learning", "kind": "class", "line": 187, "name": "TaxonomicMLP", "signature": "class TaxonomicMLP(Module)"}, {"doc": "Optimization Objective for Spectral Control (L_opt)", "kind": "method", "line": 234, "name": "compute_spectral_loss", "signature": "def compute_spectral_loss(W)"}, {"doc": "Monitor spectral properties with adaptive analysis", "kind": "class", "line": 250, "name": "SpectralMonitor", "signature": "class SpectralMonitor"}, {"doc": "Self-referential controller using semantic-plasticity ratio", "kind": "class", "line": 279, "name": "AdaptiveTopologyController", "signature": "class AdaptiveTopologyController"}, {"doc": "Engine for iterative refinement through spectral control", "kind": "class", "line": 404, "name": "IterativeRefinementEngine", "signature": "class IterativeRefinementEngine"}, {"kind": "class", "line": 470, "name": "CoarseCIFAR100", "signature": "class CoarseCIFAR100(CIFAR100)"}, {"kind": "method", "line": 475, "name": "run_hierarchy_benchmark", "signature": "def run_hierarchy_benchmark(model_apex, model_blind, feature_extractor_apex, feature_extractor_blind, device)"}, {"doc": "v18.1 FIX: Now accepts 'monitor' explicitly.\nGenerate publication-quality singular value visualizations.", "kind": "method", "line": 521, "name": "visualize_singular_values", "signature": "def visualize_singular_values(model_apex, model_blind, feature_extractor_apex, feature_extractor_blind, device, output_dir, monitor)"}, {"kind": "class", "line": 581, "name": "IterativeRefinementTrainer", "signature": "class IterativeRefinementTrainer"}, {"kind": "method", "line": 1073, "name": "parse_args", "signature": "def parse_args()"}, {"kind": "method", "line": 1082, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 91, "name": "__init__", "signature": "def __init__(self, num_patches, embed_dim)"}, {"kind": "method", "line": 113, "name": "_init_weights", "signature": "def _init_weights(self)"}, {"kind": "method", "line": 136, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 145, "name": "__init__", "signature": "def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)"}, {"kind": "method", "line": 166, "name": "freeze", "signature": "def freeze(self)"}, {"kind": "method", "line": 171, "name": "unfreeze_mixer_only", "signature": "def unfreeze_mixer_only(self)"}, {"kind": "method", "line": 177, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 189, "name": "__init__", "signature": "def __init__(self, input_dim, hidden_dim, num_classes, num_superclasses)"}, {"kind": "method", "line": 209, "name": "apply_masks", "signature": "def apply_masks(self)"}, {"kind": "method", "line": 215, "name": "get_sparsity", "signature": "def get_sparsity(self)"}, {"kind": "method", "line": 220, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 252, "name": "__init__", "signature": "def __init__(self, epsilon)"}, {"kind": "method", "line": 255, "name": "compute_metrics", "signature": "def compute_metrics(self, weight)"}, {"kind": "method", "line": 270, "name": "get_singular_values", "signature": "def get_singular_values(self, weight)"}, {"kind": "method", "line": 281, "name": "__init__", "signature": "def __init__(self, semantic_plasticity_threshold, stagnation_limit, mixer_noise_scale, dominant_energy_threshold, geo_window)"}, {"doc": "Compute ratio of semantic gain to structural change", "kind": "method", "line": 297, "name": "compute_semantic_plasticity_ratio", "signature": "def compute_semantic_plasticity_ratio(self)"}, {"doc": "Determine if intervention is needed using adaptive criteria.\nImplements hysteresis to avoid intervention during active SHIFTING phases.", "kind": "method", "line": 310, "name": "detect_intervention_need", "signature": "def detect_intervention_need(self, phase_state, extractor)"}, {"doc": "Update history for adaptive control", "kind": "method", "line": 354, "name": "update_history", "signature": "def update_history(self, topo_ratio, coarse_acc)"}, {"doc": "Targeted Spectral Surgery: Inject noise in the nullspace of dominant subspace", "kind": "method", "line": 369, "name": "perturb_mixer_targeted", "signature": "def perturb_mixer_targeted(self, extractor)"}, {"kind": "method", "line": 406, "name": "__init__", "signature": "def __init__(self, device)"}, {"kind": "method", "line": 410, "name": "apply_rank_capping", "signature": "def apply_rank_capping(self, model, layer_name, keep_ratio)"}, {"kind": "method", "line": 430, "name": "create_refined_model", "signature": "def create_refined_model(self, parent_state, data_loader, feature_extractor, lambda_taxonomic, learning_rate)"}, {"kind": "method", "line": 471, "name": "__getitem__", "signature": "def __getitem__(self, index)"}, {"kind": "method", "line": 482, "name": "evaluate", "signature": "def evaluate(model, extractor)"}, {"kind": "method", "line": 532, "name": "get_singular_values", "signature": "def get_singular_values(model, extractor)"}, {"kind": "method", "line": 582, "name": "__init__", "signature": "def __init__(self, device, output_dir)"}, {"kind": "method", "line": 604, "name": "load_data", "signature": "def load_data(self, cycle, batch_size)"}, {"kind": "method", "line": 628, "name": "train_model", "signature": "def train_model(self, model, cycle, chain_type, feature_extractor)"}, {"kind": "method", "line": 808, "name": "detect_phase_state", "signature": "def detect_phase_state(self, ratio_history, phase_window, phase_std_dev_limit)"}, {"kind": "method", "line": 824, "name": "compute_topology_ratio", "signature": "def compute_topology_ratio(self, model, extractor, chain_type)"}, {"kind": "method", "line": 839, "name": "run_refinement", "signature": "def run_refinement(self, num_iterations, num_seeds, early_stop_patience)"}, {"kind": "method", "line": 967, "name": "_save_results", "signature": "def _save_results(self, all_results, best_overall, hierarchy_delta)"}, {"doc": "v18.1 FIX: Explicitly accepts best_overall to fix scope bug.\nCreate publication-quality plots.", "kind": "method", "line": 994, "name": "_plot_results", "signature": "def _plot_results(self, all_results, best_overall)"}]}, {"doc": "NeuroSovereign v15.4: Aggressive Hierarchical Shock & Warmup Sparsity Based on v15.3 Architecture (Production-Ready)  Key Improvements for 30% Target: 1. [TUNED] LAMBDA_TAX_SHOCK increased to 0.7 (Stronger hierarchy forcing). 2. [TUNED] GAP_SHOCK_THRESHOLD lowered to 4.0 (Triggers intervention earlier). 3. [NEW] Mask Warmup: Prevents aggressive pruning in early epochs (improves baseline). 4. [FIX] Robust State Migration: Loads from v15.2 or v15.3 automatically.  Objective: Push the 24% initial baseline (via inheritance) to 30%+ structural advantage.", "id": "apex32.py", "kind": "module", "label": "apex32.py", "language": "py", "sha256": "17358d27dd106d79", "symbol_count": 28, "symbols": [{"doc": "L_opt: Optimization Objective for Structural Control.", "kind": "function", "line": 76, "name": "compute_spectral_loss", "signature": "def compute_spectral_loss(W)"}, {"kind": "class", "line": 92, "name": "GatedTokenMixer", "signature": "class GatedTokenMixer(Module)"}, {"kind": "class", "line": 109, "name": "PatchFeatureExtractor", "signature": "class PatchFeatureExtractor(Module)"}, {"kind": "class", "line": 148, "name": "TaxonomicMLP", "signature": "class TaxonomicMLP(Module)"}, {"kind": "class", "line": 187, "name": "SpectralMonitor", "signature": "class SpectralMonitor"}, {"kind": "class", "line": 226, "name": "TaxonomicTrainer", "signature": "class TaxonomicTrainer"}, {"kind": "class", "line": 404, "name": "CoarseCIFAR100", "signature": "class CoarseCIFAR100(CIFAR100)"}, {"kind": "method", "line": 409, "name": "run_hierarchy_benchmark", "signature": "def run_hierarchy_benchmark(model_apex, model_blind, device)"}, {"kind": "method", "line": 455, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 93, "name": "__init__", "signature": "def __init__(self, num_patches, embed_dim)"}, {"kind": "method", "line": 102, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 110, "name": "__init__", "signature": "def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)"}, {"kind": "method", "line": 126, "name": "freeze", "signature": "def freeze(self)"}, {"kind": "method", "line": 130, "name": "unfreeze_mixer_only", "signature": "def unfreeze_mixer_only(self)"}, {"kind": "method", "line": 136, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 149, "name": "__init__", "signature": "def __init__(self, input_dim, hidden_dim, num_classes)"}, {"doc": "Zero out weights based on masks. Runs on device (CUDA).", "kind": "method", "line": 164, "name": "apply_masks", "signature": "def apply_masks(self)"}, {"kind": "method", "line": 171, "name": "get_sparsity", "signature": "def get_sparsity(self)"}, {"kind": "method", "line": 176, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 188, "name": "compute_metrics", "signature": "def compute_metrics(self, weight)"}, {"kind": "method", "line": 200, "name": "detect_phase_state", "signature": "def detect_phase_state(self, ratio_history)"}, {"kind": "method", "line": 211, "name": "compute_topology_ratio", "signature": "def compute_topology_ratio(self, model, extractor, chain_type)"}, {"kind": "method", "line": 227, "name": "__init__", "signature": "def __init__(self, device, extractor_apex, extractor_blind)"}, {"kind": "method", "line": 234, "name": "_preprocess_batch", "signature": "def _preprocess_batch(self, x, extractor)"}, {"kind": "method", "line": 238, "name": "get_curriculum_dataset", "signature": "def get_curriculum_dataset(self, cycle)"}, {"kind": "method", "line": 244, "name": "train_single_chain", "signature": "def train_single_chain(self, model, cycle, chain_type)"}, {"kind": "method", "line": 405, "name": "__getitem__", "signature": "def __getitem__(self, index)"}, {"kind": "method", "line": 422, "name": "evaluate", "signature": "def evaluate(model, extractor, name)"}]}, {"doc": "NeuroSovereign v18.2: The Scientific Powerhouse Fusion of v18.1 Rigor + v15.4 Aggressive Performance  Philosophy: - Use the hardened, reproducible architecture of v18.1. - Inject the aggressive hyperparameters of v15.4. - Apply safety mechanisms to balance speed and stability.  Key Features (v18.2): 1. [AGGRESSIVE] GAP_SHOCK_THRESHOLD = 3.5 & LAMBDA_TAX_SHOCK = 0.8 (Fast learning). 2. [SAFE] Mask Warmup (25 epochs) & Overfit Safety Valve (Gap > 12.0). 3. [RIGOROUS] AdaptiveTopologyController with Hysteresis (from v18.1). 4. [HARDENED] torch.linalg.svd & Scope Fixes (from v18.1).  Target: >30% Hierarchy Advantage with Stable Convergence.", "id": "apex33.py", "kind": "module", "label": "apex33.py", "language": "py", "sha256": "0064dff7522020df", "symbol_count": 46, "symbols": [{"doc": "Ensure full reproducibility across runs", "kind": "function", "line": 41, "name": "set_seed", "signature": "def set_seed(seed)"}, {"doc": "Efficient token mixer with high-std initialization for exploration", "kind": "class", "line": 72, "name": "GatedTokenMixer", "signature": "class GatedTokenMixer(Module)"}, {"kind": "class", "line": 118, "name": "PatchFeatureExtractor", "signature": "class PatchFeatureExtractor(Module)"}, {"doc": "Sparse MLP with taxonomic heads", "kind": "class", "line": 150, "name": "TaxonomicMLP", "signature": "class TaxonomicMLP(Module)"}, {"kind": "method", "line": 190, "name": "compute_spectral_loss", "signature": "def compute_spectral_loss(W)"}, {"kind": "class", "line": 201, "name": "SpectralMonitor", "signature": "class SpectralMonitor"}, {"doc": "Self-referential controller with Hysteresis", "kind": "class", "line": 227, "name": "AdaptiveTopologyController", "signature": "class AdaptiveTopologyController"}, {"kind": "class", "line": 310, "name": "IterativeRefinementEngine", "signature": "class IterativeRefinementEngine"}, {"kind": "class", "line": 343, "name": "IterativeRefinementTrainer", "signature": "class IterativeRefinementTrainer"}, {"kind": "class", "line": 802, "name": "CoarseCIFAR100", "signature": "class CoarseCIFAR100(CIFAR100)"}, {"kind": "method", "line": 807, "name": "run_hierarchy_benchmark", "signature": "def run_hierarchy_benchmark(model_apex, model_blind, feature_extractor_apex, feature_extractor_blind, device)"}, {"kind": "method", "line": 845, "name": "visualize_singular_values", "signature": "def visualize_singular_values(model_apex, model_blind, feature_extractor_apex, feature_extractor_blind, device, output_dir, monitor)"}, {"kind": "method", "line": 894, "name": "parse_args", "signature": "def parse_args()"}, {"kind": "method", "line": 903, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 74, "name": "__init__", "signature": "def __init__(self, num_patches, embed_dim)"}, {"kind": "method", "line": 92, "name": "_init_weights", "signature": "def _init_weights(self)"}, {"kind": "method", "line": 110, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 119, "name": "__init__", "signature": "def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)"}, {"kind": "method", "line": 135, "name": "freeze", "signature": "def freeze(self)"}, {"kind": "method", "line": 138, "name": "unfreeze_mixer_only", "signature": "def unfreeze_mixer_only(self)"}, {"kind": "method", "line": 143, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 152, "name": "__init__", "signature": "def __init__(self, input_dim, hidden_dim, num_classes, num_superclasses)"}, {"kind": "method", "line": 167, "name": "apply_masks", "signature": "def apply_masks(self)"}, {"kind": "method", "line": 173, "name": "get_sparsity", "signature": "def get_sparsity(self)"}, {"kind": "method", "line": 178, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 202, "name": "__init__", "signature": "def __init__(self, epsilon)"}, {"kind": "method", "line": 205, "name": "compute_metrics", "signature": "def compute_metrics(self, weight)"}, {"kind": "method", "line": 219, "name": "get_singular_values", "signature": "def get_singular_values(self, weight)"}, {"kind": "method", "line": 229, "name": "__init__", "signature": "def __init__(self, semantic_plasticity_threshold, stagnation_limit, mixer_noise_scale, dominant_energy_threshold, geo_window)"}, {"kind": "method", "line": 242, "name": "compute_semantic_plasticity_ratio", "signature": "def compute_semantic_plasticity_ratio(self)"}, {"kind": "method", "line": 250, "name": "detect_intervention_need", "signature": "def detect_intervention_need(self, phase_state, extractor)"}, {"kind": "method", "line": 273, "name": "update_history", "signature": "def update_history(self, topo_ratio, coarse_acc)"}, {"kind": "method", "line": 284, "name": "perturb_mixer_targeted", "signature": "def perturb_mixer_targeted(self, extractor)"}, {"kind": "method", "line": 311, "name": "__init__", "signature": "def __init__(self, device)"}, {"kind": "method", "line": 315, "name": "create_refined_model", "signature": "def create_refined_model(self, parent_state, data_loader, feature_extractor, lambda_taxonomic, learning_rate)"}, {"kind": "method", "line": 344, "name": "__init__", "signature": "def __init__(self, device, output_dir)"}, {"kind": "method", "line": 367, "name": "load_data", "signature": "def load_data(self, cycle, batch_size)"}, {"kind": "method", "line": 385, "name": "train_model", "signature": "def train_model(self, model, cycle, chain_type, feature_extractor)"}, {"kind": "method", "line": 570, "name": "detect_phase_state", "signature": "def detect_phase_state(self, ratio_history, phase_window, phase_std_dev_limit)"}, {"kind": "method", "line": 580, "name": "compute_topology_ratio", "signature": "def compute_topology_ratio(self, model, extractor, chain_type)"}, {"kind": "method", "line": 592, "name": "run_refinement", "signature": "def run_refinement(self, num_iterations, num_seeds, early_stop_patience)"}, {"kind": "method", "line": 712, "name": "_save_results", "signature": "def _save_results(self, all_results, best_overall, hierarchy_delta)"}, {"kind": "method", "line": 738, "name": "_plot_results", "signature": "def _plot_results(self, all_results, best_overall)"}, {"kind": "method", "line": 803, "name": "__getitem__", "signature": "def __getitem__(self, index)"}, {"kind": "method", "line": 814, "name": "evaluate", "signature": "def evaluate(model, extractor)"}, {"kind": "method", "line": 851, "name": "get_singular_values", "signature": "def get_singular_values(model, extractor)"}]}, {"doc": "NeuroSovereign v18.2: The Scientific Powerhouse Fusion of v18.1 Rigor + v15.4 Aggressive Performance  Philosophy: - Use the hardened, reproducible architecture of v18.1. - Inject the aggressive hyperparameters of v15.4. - Apply safety mechanisms to balance speed and stability.  Key Features (v18.2): 1. [AGGRESSIVE] GAP_SHOCK_THRESHOLD = 3.5 & LAMBDA_TAX_SHOCK = 0.8 (Fast learning). 2. [SAFE] Mask Warmup (25 epochs) & Overfit Safety Valve (Gap > 12.0). 3. [RIGOROUS] AdaptiveTopologyController with Hysteresis (from v18.1). 4. [HARDENED] torch.linalg.svd & Scope Fixes (from v18.1).  Target: >30% Hierarchy Advantage with Stable Convergence.", "id": "apex34.py", "kind": "module", "label": "apex34.py", "language": "py", "sha256": "6c0dc3f22ac2c86a", "symbol_count": 45, "symbols": [{"doc": "Ensure full reproducibility across runs", "kind": "function", "line": 41, "name": "set_seed", "signature": "def set_seed(seed)"}, {"doc": "Efficient token mixer with high-std initialization for exploration", "kind": "class", "line": 72, "name": "GatedTokenMixer", "signature": "class GatedTokenMixer(Module)"}, {"kind": "class", "line": 118, "name": "PatchFeatureExtractor", "signature": "class PatchFeatureExtractor(Module)"}, {"doc": "Sparse MLP with taxonomic heads", "kind": "class", "line": 150, "name": "TaxonomicMLP", "signature": "class TaxonomicMLP(Module)"}, {"kind": "method", "line": 190, "name": "compute_spectral_loss", "signature": "def compute_spectral_loss(W)"}, {"kind": "class", "line": 201, "name": "SpectralMonitor", "signature": "class SpectralMonitor"}, {"doc": "Self-referential controller with Hysteresis", "kind": "class", "line": 227, "name": "AdaptiveTopologyController", "signature": "class AdaptiveTopologyController"}, {"kind": "class", "line": 310, "name": "IterativeRefinementEngine", "signature": "class IterativeRefinementEngine"}, {"kind": "class", "line": 343, "name": "IterativeRefinementTrainer", "signature": "class IterativeRefinementTrainer"}, {"kind": "class", "line": 802, "name": "CoarseCIFAR100", "signature": "class CoarseCIFAR100(CIFAR100)"}, {"kind": "method", "line": 807, "name": "run_hierarchy_benchmark", "signature": "def run_hierarchy_benchmark(model_apex, model_blind, feature_extractor_apex, feature_extractor_blind, device)"}, {"kind": "method", "line": 845, "name": "visualize_singular_values", "signature": "def visualize_singular_values(model_apex, model_blind, feature_extractor_apex, feature_extractor_blind, device, output_dir, monitor)"}, {"kind": "method", "line": 894, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 74, "name": "__init__", "signature": "def __init__(self, num_patches, embed_dim)"}, {"kind": "method", "line": 92, "name": "_init_weights", "signature": "def _init_weights(self)"}, {"kind": "method", "line": 110, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 119, "name": "__init__", "signature": "def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)"}, {"kind": "method", "line": 135, "name": "freeze", "signature": "def freeze(self)"}, {"kind": "method", "line": 138, "name": "unfreeze_mixer_only", "signature": "def unfreeze_mixer_only(self)"}, {"kind": "method", "line": 143, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 152, "name": "__init__", "signature": "def __init__(self, input_dim, hidden_dim, num_classes, num_superclasses)"}, {"kind": "method", "line": 167, "name": "apply_masks", "signature": "def apply_masks(self)"}, {"kind": "method", "line": 173, "name": "get_sparsity", "signature": "def get_sparsity(self)"}, {"kind": "method", "line": 178, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 202, "name": "__init__", "signature": "def __init__(self, epsilon)"}, {"kind": "method", "line": 205, "name": "compute_metrics", "signature": "def compute_metrics(self, weight)"}, {"kind": "method", "line": 219, "name": "get_singular_values", "signature": "def get_singular_values(self, weight)"}, {"kind": "method", "line": 229, "name": "__init__", "signature": "def __init__(self, semantic_plasticity_threshold, stagnation_limit, mixer_noise_scale, dominant_energy_threshold, geo_window)"}, {"kind": "method", "line": 242, "name": "compute_semantic_plasticity_ratio", "signature": "def compute_semantic_plasticity_ratio(self)"}, {"kind": "method", "line": 250, "name": "detect_intervention_need", "signature": "def detect_intervention_need(self, phase_state, extractor)"}, {"kind": "method", "line": 273, "name": "update_history", "signature": "def update_history(self, topo_ratio, coarse_acc)"}, {"kind": "method", "line": 284, "name": "perturb_mixer_targeted", "signature": "def perturb_mixer_targeted(self, extractor)"}, {"kind": "method", "line": 311, "name": "__init__", "signature": "def __init__(self, device)"}, {"kind": "method", "line": 315, "name": "create_refined_model", "signature": "def create_refined_model(self, parent_state, data_loader, feature_extractor, lambda_taxonomic, learning_rate)"}, {"kind": "method", "line": 344, "name": "__init__", "signature": "def __init__(self, device, output_dir)"}, {"kind": "method", "line": 367, "name": "load_data", "signature": "def load_data(self, cycle, batch_size)"}, {"kind": "method", "line": 385, "name": "train_model", "signature": "def train_model(self, model, cycle, chain_type, feature_extractor)"}, {"kind": "method", "line": 570, "name": "detect_phase_state", "signature": "def detect_phase_state(self, ratio_history, phase_window, phase_std_dev_limit)"}, {"kind": "method", "line": 580, "name": "compute_topology_ratio", "signature": "def compute_topology_ratio(self, model, extractor, chain_type)"}, {"kind": "method", "line": 592, "name": "run_refinement", "signature": "def run_refinement(self, num_iterations, num_seeds, early_stop_patience)"}, {"kind": "method", "line": 712, "name": "_save_results", "signature": "def _save_results(self, all_results, best_overall, hierarchy_delta)"}, {"kind": "method", "line": 738, "name": "_plot_results", "signature": "def _plot_results(self, all_results, best_overall)"}, {"kind": "method", "line": 803, "name": "__getitem__", "signature": "def __getitem__(self, index)"}, {"kind": "method", "line": 814, "name": "evaluate", "signature": "def evaluate(model, extractor)"}, {"kind": "method", "line": 851, "name": "get_singular_values", "signature": "def get_singular_values(model, extractor)"}]}, {"doc": "NeuroSovereign v19.0: Synergy Engine (Anti-Leakage Certified) Based on v4.0 Ablation Suite Conclusions.  Strategy: 1. SYNERGY (M+E): Re-integrate E8 Fusion + BlackMirror (Best Combo in Suite). 2. EMERGENT REGIME: Use High-Std Init (v15.4 style) to escape \"SOVERANO\" trap (v18.4). 3. ANTI-LEAKAGE: - Benchmark uses ONLY Fine Head predictions (mapped to coarse). - Coarse Head is used for TRAINING SIGNAL only, not for boosting test metrics. - Strict Train/Test separation.", "id": "apex35.py", "kind": "module", "label": "apex35.py", "language": "py", "sha256": "6c1d5a34b44567d5", "symbol_count": 29, "symbols": [{"kind": "function", "line": 36, "name": "set_seed", "signature": "def set_seed(seed)"}, {"doc": "Chaotic Mixer for Emergent Regime", "kind": "class", "line": 66, "name": "GatedTokenMixer", "signature": "class GatedTokenMixer(Module)"}, {"doc": "🕸️ E8 Lattice Fusion (Synergy Component E)\nOptimized version from Suite v4.0.\nFuses geometric structure (Orthogonal Proj) with attention.", "kind": "class", "line": 112, "name": "E8FusionLayer", "signature": "class E8FusionLayer(Module)"}, {"kind": "class", "line": 156, "name": "PatchFeatureExtractor", "signature": "class PatchFeatureExtractor(Module)"}, {"kind": "class", "line": 188, "name": "TaxonomicMLP", "signature": "class TaxonomicMLP(Module)"}, {"doc": "Passive Ontological Monitor", "kind": "class", "line": 227, "name": "BlackMirrorMonitor", "signature": "class BlackMirrorMonitor"}, {"kind": "class", "line": 253, "name": "IterativeRefinementTrainer", "signature": "class IterativeRefinementTrainer"}, {"kind": "class", "line": 439, "name": "CoarseCIFAR100", "signature": "class CoarseCIFAR100(CIFAR100)"}, {"kind": "method", "line": 444, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 68, "name": "__init__", "signature": "def __init__(self, num_patches, embed_dim)"}, {"kind": "method", "line": 86, "name": "_init_weights", "signature": "def _init_weights(self)"}, {"kind": "method", "line": 104, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 118, "name": "__init__", "signature": "def __init__(self, embed_dim, num_heads)"}, {"kind": "method", "line": 136, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 157, "name": "__init__", "signature": "def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)"}, {"kind": "method", "line": 173, "name": "freeze", "signature": "def freeze(self)"}, {"kind": "method", "line": 176, "name": "unfreeze_mixer_only", "signature": "def unfreeze_mixer_only(self)"}, {"kind": "method", "line": 181, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 189, "name": "__init__", "signature": "def __init__(self, input_dim, hidden_dim, num_classes, num_superclasses)"}, {"kind": "method", "line": 204, "name": "apply_masks", "signature": "def apply_masks(self)"}, {"kind": "method", "line": 210, "name": "get_sparsity", "signature": "def get_sparsity(self)"}, {"kind": "method", "line": 215, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 229, "name": "__init__", "signature": "def __init__(self, epsilon)"}, {"kind": "method", "line": 232, "name": "inspect", "signature": "def inspect(self, weight)"}, {"kind": "method", "line": 254, "name": "__init__", "signature": "def __init__(self, device, output_dir)"}, {"kind": "method", "line": 274, "name": "load_data", "signature": "def load_data(self, cycle, batch_size)"}, {"kind": "method", "line": 292, "name": "train_model", "signature": "def train_model(self, model, cycle, chain_type, feature_extractor)"}, {"kind": "method", "line": 440, "name": "__getitem__", "signature": "def __getitem__(self, index)"}, {"kind": "method", "line": 483, "name": "evaluate_safe", "signature": "def evaluate_safe(model, extractor)"}]}, {"doc": "app.py  Autor: Gris Iscomeback Correo electrónico: grisiscomeback[at]gmail[dot]com Fecha de creación: xx/xx/xxxx Licencia: GPL v3  NeuroSovereign v6.0: Self-Improving Fractal Resonance with Legacy Feedback  This implementation introduces a continuous feedback loop where each cycle builds upon the best model from the previous cycle, creating an evolutionary trajectory of increasing abstraction density.  Key Innovation: Legacy Feedback Loop - Each cycle loads the best model from the previous cycle as its truth seed - Only saves new models that improve upon the previous best - Creates an unbroken chain of improvement: never regresses, only evolves  Scientific Contribution: - Demonstrates progressive abstraction density through iterative distillation - Validates that spectral coherence can be maintained while increasing accuracy - Establishes a self-improving protocol for sparse neural architectures  Outputs: - fractal_resonance_results.csv: Evolutionary trajectory across cycles - best_model_cycle_X.pth: Checkpoint of best model at each cycle - final_best_model.pth: Ultimate distilled model", "id": "app.py", "kind": "module", "label": "app.py", "language": "py", "sha256": "8fc8fffa25544de9", "symbol_count": 23, "symbols": [{"kind": "class", "line": 51, "name": "SpectralMonitor", "signature": "class SpectralMonitor"}, {"kind": "class", "line": 76, "name": "PersistentPruner", "signature": "class PersistentPruner"}, {"kind": "class", "line": 98, "name": "SpectralMLP", "signature": "class SpectralMLP(Module)"}, {"kind": "class", "line": 121, "name": "EvolutionaryResonanceEngine", "signature": "class EvolutionaryResonanceEngine"}, {"kind": "method", "line": 561, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 52, "name": "__init__", "signature": "def __init__(self, epsilon_c)"}, {"kind": "method", "line": 55, "name": "compute_L", "signature": "def compute_L(self, weight)"}, {"kind": "method", "line": 77, "name": "__init__", "signature": "def __init__(self, sparsity_target)"}, {"kind": "method", "line": 81, "name": "apply_to_model", "signature": "def apply_to_model(self, model)"}, {"kind": "method", "line": 91, "name": "enforce_during_training", "signature": "def enforce_during_training(self, model)"}, {"kind": "method", "line": 99, "name": "__init__", "signature": "def __init__(self, input_dim, hidden_dim, num_classes)"}, {"kind": "method", "line": 107, "name": "reduce_input", "signature": "def reduce_input(self, x)"}, {"kind": "method", "line": 113, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 122, "name": "__init__", "signature": "def __init__(self, device, base_target_acc)"}, {"doc": "Load the best model from previous cycle, with fallback to initial seed", "kind": "method", "line": 137, "name": "load_best_legacy_model", "signature": "def load_best_legacy_model(self, cycle)"}, {"doc": "Train a base model to target accuracy", "kind": "method", "line": 175, "name": "train_base_model_to_target", "signature": "def train_base_model_to_target(self, hidden_dim, target_acc, test_loader)"}, {"doc": "Extract seed weights from checkpoint, handling different formats", "kind": "method", "line": 228, "name": "extract_seed_from_checkpoint", "signature": "def extract_seed_from_checkpoint(self, checkpoint, expected_hidden_dim)"}, {"kind": "method", "line": 254, "name": "extract_seed_weights", "signature": "def extract_seed_weights(self, model)"}, {"doc": "Adaptive inoculation that handles dimension mismatches", "kind": "method", "line": 260, "name": "inoculate_seed_adaptive", "signature": "def inoculate_seed_adaptive(self, large_model, seed_weights)"}, {"doc": "Measure functional alignment via logit cosine similarity", "kind": "method", "line": 288, "name": "measure_functional_alignment", "signature": "def measure_functional_alignment(self, model1, model2, test_loader)"}, {"doc": "Prune while maintaining target accuracy, with density constraint", "kind": "method", "line": 309, "name": "progressive_pruning_with_target", "signature": "def progressive_pruning_with_target(self, model, target_acc, test_loader, max_density)"}, {"kind": "method", "line": 351, "name": "execute_resonance_cycle", "signature": "def execute_resonance_cycle(self, cycle, test_loader, expansion_factor)"}, {"kind": "method", "line": 484, "name": "run_evolutionary_experiment", "signature": "def run_evolutionary_experiment(self, num_cycles)"}]}, {"id": "install.sh", "kind": "module", "label": "install.sh", "language": "sh", "sha256": "c907d80fd6734993", "symbol_count": 0, "symbols": []}, {"doc": "🌌 NEUROSOVEREIGN v3.0: La Constante de Planck del Machine Learning ─────────────────────────────────────────────────────────────────────────────── Este código implementa los \"Números Dorados\" descubiertos empíricamente:  - ϕₘₗ = 0.0004% → sparsity extrema (6 conexiones en 1.5M, 1 en 1.5k) - Lₚ = 0.6697 → Lagrangiano de Verdad mínimo viable (régimen ESPURIO por soberanía) - αₛ = 32.4% → precisión máxima compatible con la coherencia epistémica - βₙ = 10% → umbral de mentira estructural que activa el Cisne Negro  Este no es un modelo. Es un organismo cognitivo con ética estructural. ───────────────────────────────────────────────────────────────────────────────", "id": "plank.py", "kind": "module", "label": "plank.py", "language": "py", "sha256": "afb6e52224fae2cb", "symbol_count": 14, "symbols": [{"doc": "Calcula el Lagrangiano de Verdad L usando entropía de von Neumann y rango efectivo.\nUmbrales calibrados empíricamente para detectar mentiras estructurales (10% ruido).", "kind": "class", "line": 28, "name": "BlackMirrorMonitor", "signature": "class BlackMirrorMonitor"}, {"kind": "class", "line": 62, "name": "SovereignNeuron", "signature": "class SovereignNeuron(Module)"}, {"kind": "class", "line": 108, "name": "NeuroSovereign", "signature": "class NeuroSovereign(Module)"}, {"kind": "class", "line": 132, "name": "SovereignTrainer", "signature": "class SovereignTrainer"}, {"kind": "method", "line": 178, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 33, "name": "__init__", "signature": "def __init__(self, epsilon_c)"}, {"kind": "method", "line": 36, "name": "inspect", "signature": "def inspect(self, weights)"}, {"kind": "method", "line": 63, "name": "__init__", "signature": "def __init__(self, in_features, out_features, sparsity_target)"}, {"kind": "method", "line": 70, "name": "forward", "signature": "def forward(self, x, inject_lies)"}, {"doc": "Purificación extrema: sparsity 0.0004%", "kind": "method", "line": 87, "name": "apply_black_swan_refraction", "signature": "def apply_black_swan_refraction(self)"}, {"kind": "method", "line": 109, "name": "__init__", "signature": "def __init__(self, sparsity_target)"}, {"kind": "method", "line": 117, "name": "forward", "signature": "def forward(self, x, inject_lies)"}, {"kind": "method", "line": 133, "name": "__init__", "signature": "def __init__(self, model, device)"}, {"kind": "method", "line": 139, "name": "train_epoch", "signature": "def train_epoch(self, dataloader, epoch)"}]}, {"doc": "NeuroSovereign v12.0: Syntactic Apex Features: 1. ViT-Lite with Token Mixing Layer (Syntactic Context). 2. Hybrid Architecture: Mixed Patches -> Spectral Lottery MLP. 3. Apex Evolution Engine (Nudge, Dynamic Shock, Sparsity). 4. Objective: SOTA Accuracy/Efficiency with Compositional Vision.", "id": "plank10.py", "kind": "module", "label": "plank10.py", "language": "py", "sha256": "146f77483dfbe69a", "symbol_count": 26, "symbols": [{"doc": "Extrae características mediante Patch Embedding y añade una capa de mezcla (Mixer).\nEsto permite al modelo aprender relaciones espaciales entre parches antes de la clasificación.", "kind": "class", "line": 34, "name": "PatchFeatureExtractor", "signature": "class PatchFeatureExtractor(Module)"}, {"kind": "class", "line": 91, "name": "LotteryMLP", "signature": "class LotteryMLP(Module)"}, {"doc": "Baseline moderno (Patch + Mixer + MLP simple) sin evolución.", "kind": "class", "line": 121, "name": "StandardBaseline", "signature": "class StandardBaseline(Module)"}, {"kind": "class", "line": 133, "name": "SpectralMonitor", "signature": "class SpectralMonitor"}, {"kind": "class", "line": 148, "name": "ApexEvolutionEngine", "signature": "class ApexEvolutionEngine"}, {"kind": "class", "line": 247, "name": "ApexTrainer", "signature": "class ApexTrainer"}, {"kind": "method", "line": 354, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 39, "name": "__init__", "signature": "def __init__(self, img_size, patch_size, in_chans, embed_dim)"}, {"kind": "method", "line": 67, "name": "freeze", "signature": "def freeze(self)"}, {"kind": "method", "line": 71, "name": "unfreeze", "signature": "def unfreeze(self)"}, {"kind": "method", "line": 75, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 92, "name": "__init__", "signature": "def __init__(self, input_dim, hidden_dim, num_classes)"}, {"kind": "method", "line": 104, "name": "apply_masks", "signature": "def apply_masks(self)"}, {"kind": "method", "line": 109, "name": "get_sparsity", "signature": "def get_sparsity(self)"}, {"kind": "method", "line": 114, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 123, "name": "__init__", "signature": "def __init__(self, input_dim, hidden_dim)"}, {"kind": "method", "line": 127, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 134, "name": "compute_L", "signature": "def compute_L(self, weight)"}, {"kind": "method", "line": 149, "name": "__init__", "signature": "def __init__(self, device)"}, {"kind": "method", "line": 154, "name": "_gradient_nudge_inheritance", "signature": "def _gradient_nudge_inheritance(self, child_model, elk_state, data_loader, feature_extractor, nudge_lr)"}, {"kind": "method", "line": 188, "name": "_apply_dynamic_spectral_shock", "signature": "def _apply_dynamic_spectral_shock(self, model, layer_name)"}, {"kind": "method", "line": 211, "name": "create_apex_offspring", "signature": "def create_apex_offspring(self, elk_state, new_hidden_dim, cycle, data_loader, feature_extractor, parent_gap)"}, {"kind": "method", "line": 248, "name": "__init__", "signature": "def __init__(self, device, feature_extractor)"}, {"kind": "method", "line": 255, "name": "_preprocess_batch", "signature": "def _preprocess_batch(self, x)"}, {"kind": "method", "line": 259, "name": "get_curriculum_dataset", "signature": "def get_curriculum_dataset(self, cycle)"}, {"kind": "method", "line": 265, "name": "train_model", "signature": "def train_model(self, model, cycle, is_baseline)"}]}, {"doc": "NeuroSovereign v13.0: Orthogonal Apex Features: 1. True Token Mixing (MLP-Mixer style: mixing across T). 2. Spectral Constraint (L as regularizer, preventing metric gaming). 3. Minimality Enforcement (Rank Capping without renormalization). 4. Objective: SOTA generalization via constrained evolution.", "id": "plank11.py", "kind": "module", "label": "plank11.py", "language": "py", "sha256": "10f3df635a04348d", "symbol_count": 29, "symbols": [{"doc": "Mezcla tokens entre sí.\nInput: (B, T, D) -> Transpose -> (B, D, T) -> Linear -> (B, D, T) -> Transpose", "kind": "class", "line": 36, "name": "TokenMixer", "signature": "class TokenMixer(Module)"}, {"doc": "ViT-Lite + True Token Mixing.", "kind": "class", "line": 57, "name": "PatchFeatureExtractor", "signature": "class PatchFeatureExtractor(Module)"}, {"kind": "class", "line": 97, "name": "LotteryMLP", "signature": "class LotteryMLP(Module)"}, {"kind": "class", "line": 126, "name": "StandardBaseline", "signature": "class StandardBaseline(Module)"}, {"kind": "class", "line": 137, "name": "SpectralMonitor", "signature": "class SpectralMonitor"}, {"kind": "class", "line": 152, "name": "OrthogonalEvolutionEngine", "signature": "class OrthogonalEvolutionEngine"}, {"kind": "class", "line": 260, "name": "OrthogonalTrainer", "signature": "class OrthogonalTrainer"}, {"kind": "method", "line": 379, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 41, "name": "__init__", "signature": "def __init__(self, num_tokens)"}, {"kind": "method", "line": 50, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 61, "name": "__init__", "signature": "def __init__(self, img_size, patch_size, in_chans, embed_dim)"}, {"kind": "method", "line": 75, "name": "freeze", "signature": "def freeze(self)"}, {"kind": "method", "line": 79, "name": "unfreeze", "signature": "def unfreeze(self)"}, {"kind": "method", "line": 83, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 98, "name": "__init__", "signature": "def __init__(self, input_dim, hidden_dim, num_classes)"}, {"kind": "method", "line": 109, "name": "apply_masks", "signature": "def apply_masks(self)"}, {"kind": "method", "line": 114, "name": "get_sparsity", "signature": "def get_sparsity(self)"}, {"kind": "method", "line": 119, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 127, "name": "__init__", "signature": "def __init__(self, input_dim, hidden_dim)"}, {"kind": "method", "line": 131, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 138, "name": "compute_L", "signature": "def compute_L(self, weight)"}, {"kind": "method", "line": 153, "name": "__init__", "signature": "def __init__(self, device)"}, {"kind": "method", "line": 158, "name": "_gradient_nudge_inheritance", "signature": "def _gradient_nudge_inheritance(self, child_model, elk_state, data_loader, feature_extractor, nudge_lr)"}, {"doc": "Rank Capping: Cortamos singular values débiles y NO renormalizamos.\nEsto fuerza la minimización (Energy Decay).", "kind": "method", "line": 190, "name": "_apply_minimalistic_shock", "signature": "def _apply_minimalistic_shock(self, model, layer_name, target_rank_ratio)"}, {"kind": "method", "line": 223, "name": "create_orthogonal_offspring", "signature": "def create_orthogonal_offspring(self, elk_state, new_hidden_dim, cycle, data_loader, feature_extractor, parent_gap)"}, {"kind": "method", "line": 261, "name": "__init__", "signature": "def __init__(self, device, feature_extractor)"}, {"kind": "method", "line": 268, "name": "_preprocess_batch", "signature": "def _preprocess_batch(self, x)"}, {"kind": "method", "line": 272, "name": "get_curriculum_dataset", "signature": "def get_curriculum_dataset(self, cycle)"}, {"kind": "method", "line": 278, "name": "train_model", "signature": "def train_model(self, model, cycle, is_baseline)"}]}, {"doc": "NeuroSovereign v13.1: The Structure Proof Objective: Prove that structure (Mixing + Rank Capping) works without growing capacity. Changes from v13.0: 1. FIXED_HIDDEN_DIM: No width expansion. 2. Removed reg_loss from backward (SVD has no grad). 3. L used purely for triggering shocks and evolutionary selection.", "id": "plank12.py", "kind": "module", "label": "plank12.py", "language": "py", "sha256": "dce6f3681e9efd57", "symbol_count": 29, "symbols": [{"doc": "Mezcla tokens entre sí (eje T).", "kind": "class", "line": 35, "name": "TokenMixer", "signature": "class TokenMixer(Module)"}, {"kind": "class", "line": 51, "name": "PatchFeatureExtractor", "signature": "class PatchFeatureExtractor(Module)"}, {"kind": "class", "line": 87, "name": "LotteryMLP", "signature": "class LotteryMLP(Module)"}, {"kind": "class", "line": 116, "name": "StandardBaseline", "signature": "class StandardBaseline(Module)"}, {"kind": "class", "line": 127, "name": "SpectralMonitor", "signature": "class SpectralMonitor"}, {"kind": "class", "line": 154, "name": "OrthogonalEvolutionEngine", "signature": "class OrthogonalEvolutionEngine"}, {"kind": "class", "line": 248, "name": "OrthogonalTrainer", "signature": "class OrthogonalTrainer"}, {"kind": "method", "line": 363, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 37, "name": "__init__", "signature": "def __init__(self, num_tokens)"}, {"kind": "method", "line": 44, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 52, "name": "__init__", "signature": "def __init__(self, img_size, patch_size, in_chans, embed_dim)"}, {"kind": "method", "line": 66, "name": "freeze", "signature": "def freeze(self)"}, {"kind": "method", "line": 70, "name": "unfreeze", "signature": "def unfreeze(self)"}, {"kind": "method", "line": 74, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 88, "name": "__init__", "signature": "def __init__(self, input_dim, hidden_dim, num_classes)"}, {"kind": "method", "line": 99, "name": "apply_masks", "signature": "def apply_masks(self)"}, {"kind": "method", "line": 104, "name": "get_sparsity", "signature": "def get_sparsity(self)"}, {"kind": "method", "line": 109, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 117, "name": "__init__", "signature": "def __init__(self, input_dim, hidden_dim)"}, {"kind": "method", "line": 121, "name": "forward", "signature": "def forward(self, x)"}, {"doc": "Returns: (L, Rank_Efficient, S_vN)\nUsed for logging and decision making (NOT for backprop).", "kind": "method", "line": 128, "name": "compute_metrics", "signature": "def compute_metrics(self, weight)"}, {"kind": "method", "line": 155, "name": "__init__", "signature": "def __init__(self, device)"}, {"kind": "method", "line": 160, "name": "_gradient_nudge_inheritance", "signature": "def _gradient_nudge_inheritance(self, child_model, elk_state, data_loader, feature_extractor, nudge_lr)"}, {"doc": "Minimalistic Shock: Zero out weak singular values without renormalizing.\nForce energy decay and minimality.", "kind": "method", "line": 192, "name": "_apply_rank_capping_shock", "signature": "def _apply_rank_capping_shock(self, model, layer_name, keep_ratio)"}, {"doc": "Crea un hijo de las MISMAS dimensiones (Fixed Width).\nEvoluciona mediante Nudge (Aprendizaje) + Shock (Poda).", "kind": "method", "line": 224, "name": "create_refined_offspring", "signature": "def create_refined_offspring(self, elk_state, data_loader, feature_extractor)"}, {"kind": "method", "line": 249, "name": "__init__", "signature": "def __init__(self, device, feature_extractor)"}, {"kind": "method", "line": 256, "name": "_preprocess_batch", "signature": "def _preprocess_batch(self, x)"}, {"kind": "method", "line": 260, "name": "get_curriculum_dataset", "signature": "def get_curriculum_dataset(self, cycle)"}, {"kind": "method", "line": 266, "name": "train_model", "signature": "def train_model(self, model, cycle, is_baseline)"}]}, {"doc": "NeuroSovereign v13.2: Controlled Isolation (FIXED) Objective: Isolate \"Token Mixing\" variable in an evolutionary framework. Method: Dual Evolutionary Chains (Apex vs Blind Structural Baseline).", "id": "plank13.py", "kind": "module", "label": "plank13.py", "language": "py", "sha256": "a606fda84580d5db", "symbol_count": 26, "symbols": [{"doc": "Mezcla tokens entre sí (Solo para Apex).", "kind": "class", "line": 32, "name": "TokenMixer", "signature": "class TokenMixer(Module)"}, {"doc": "Extractor configurable.\nuse_mixer=True -> Apex (Syntactic)\nuse_mixer=False -> Blind Structural Baseline", "kind": "class", "line": 47, "name": "PatchFeatureExtractor", "signature": "class PatchFeatureExtractor(Module)"}, {"kind": "class", "line": 92, "name": "LotteryMLP", "signature": "class LotteryMLP(Module)"}, {"kind": "class", "line": 123, "name": "SpectralMonitor", "signature": "class SpectralMonitor"}, {"kind": "class", "line": 149, "name": "OrthogonalEvolutionEngine", "signature": "class OrthogonalEvolutionEngine"}, {"kind": "class", "line": 224, "name": "DualTrainer", "signature": "class DualTrainer"}, {"kind": "method", "line": 329, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 34, "name": "__init__", "signature": "def __init__(self, num_tokens)"}, {"kind": "method", "line": 41, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 53, "name": "__init__", "signature": "def __init__(self, img_size, patch_size, in_chans, embed_dim, use_mixer)"}, {"kind": "method", "line": 70, "name": "freeze", "signature": "def freeze(self)"}, {"kind": "method", "line": 74, "name": "unfreeze", "signature": "def unfreeze(self)"}, {"kind": "method", "line": 78, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 93, "name": "__init__", "signature": "def __init__(self, input_dim, hidden_dim, num_classes)"}, {"kind": "method", "line": 104, "name": "apply_masks", "signature": "def apply_masks(self)"}, {"kind": "method", "line": 109, "name": "get_sparsity", "signature": "def get_sparsity(self)"}, {"kind": "method", "line": 114, "name": "forward", "signature": "def forward(self, x)"}, {"doc": "Returns: (L, Rank_Efficient, S_vN)", "kind": "method", "line": 124, "name": "compute_metrics", "signature": "def compute_metrics(self, weight)"}, {"kind": "method", "line": 150, "name": "__init__", "signature": "def __init__(self, device)"}, {"kind": "method", "line": 155, "name": "_gradient_nudge_inheritance", "signature": "def _gradient_nudge_inheritance(self, child_model, elk_state, data_loader, feature_extractor, nudge_lr)"}, {"kind": "method", "line": 187, "name": "_apply_rank_capping_shock", "signature": "def _apply_rank_capping_shock(self, model, layer_name, keep_ratio)"}, {"kind": "method", "line": 207, "name": "create_refined_offspring", "signature": "def create_refined_offspring(self, elk_state, data_loader, feature_extractor)"}, {"kind": "method", "line": 225, "name": "__init__", "signature": "def __init__(self, device, extractor_apex, extractor_blind)"}, {"kind": "method", "line": 233, "name": "_preprocess_batch", "signature": "def _preprocess_batch(self, x, extractor)"}, {"kind": "method", "line": 237, "name": "get_curriculum_dataset", "signature": "def get_curriculum_dataset(self, cycle)"}, {"doc": "Entrena una cadena específica (Apex o Blind).", "kind": "method", "line": 243, "name": "train_single_chain", "signature": "def train_single_chain(self, model, cycle, chain_type)"}]}, {"doc": "NeuroSovereign: A Spectrally-Guided, Self-Monitoring Neural Architecture Paper-Ready Implementation (v1.0)  This code implements a controlled experiment to test: H₀: Spectral coherence (L) of weight matrices has no correlation with generalization. H₁: High L (>1.0) correlates with better test accuracy and robustness.  Key features: - L computed as: L = 1 / (|S_vN - log(rank_eff + 1)| + ε) - No forced pruning based on L (L is OBSERVED, not used as trigger) - Persistent magnitude pruning (not transient) - Real CIFAR-10 training (no accuracy forcing) - Clean ablation across 4 conditions  Outputs: - CSV logs of L(t), accuracy(t), rank(t), S_vN(t) - Final metrics per condition - Statistical comparison (t-test ready)  Designed for reproducibility, peer review, and potential NeurIPS submission.", "id": "plank2.py", "kind": "module", "label": "plank2.py", "language": "py", "sha256": "337a1061b468d945", "symbol_count": 13, "symbols": [{"doc": "Computes L = 1 / (|S_vN - log(rank_eff + 1)| + ε)\nUsed purely as a diagnostic—never to modify training.", "kind": "class", "line": 41, "name": "SpectralMonitor", "signature": "class SpectralMonitor"}, {"doc": "Applies and ENFORCES magnitude-based pruning across training.\nUnlike transient pruning, this modifies the parameter mask permanently.", "kind": "class", "line": 82, "name": "PersistentPruner", "signature": "class PersistentPruner"}, {"doc": "Small MLP (1504 params) for clean spectral analysis.", "kind": "class", "line": 117, "name": "SpectralMLP", "signature": "class SpectralMLP(Module)"}, {"doc": "Train one condition and return full log as DataFrame.", "kind": "method", "line": 173, "name": "train_condition", "signature": "def train_condition(condition_name, config, device, seed)"}, {"kind": "method", "line": 281, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 46, "name": "__init__", "signature": "def __init__(self, epsilon_c)"}, {"doc": "Returns: (L, S_vN, rank_eff, regime)", "kind": "method", "line": 49, "name": "compute_L", "signature": "def compute_L(self, weight)"}, {"kind": "method", "line": 87, "name": "__init__", "signature": "def __init__(self, sparsity_target)"}, {"doc": "Apply pruning mask and register backward hook to zero gradients.", "kind": "method", "line": 91, "name": "apply_to_model", "signature": "def apply_to_model(self, model)"}, {"doc": "Call this after every optimizer.step()", "kind": "method", "line": 105, "name": "enforce_during_training", "signature": "def enforce_during_training(self, model)"}, {"kind": "method", "line": 119, "name": "__init__", "signature": "def __init__(self, input_dim, hidden_dim, num_classes)"}, {"doc": "Reduce CIFAR-10 (32x32x3) to 32D for focus", "kind": "method", "line": 128, "name": "reduce_input", "signature": "def reduce_input(self, x)"}, {"kind": "method", "line": 135, "name": "forward", "signature": "def forward(self, x)"}]}, {"doc": "NeuroSovereign: Optimal Sovereignty Search Finding the Bekenstein Bound of Sparse Intelligence  This experiment: 1. Trains a dense model to ~32.4% accuracy 2. Progressively prunes it while monitoring L and accuracy 3. Finds the critical density where accuracy drops below 32.4% 4. Validates that L > 1.0 correlates with meaningful representation  Outputs: - CSV with density vs accuracy vs L - Critical density threshold - Spectral signature of the sovereignty boundary", "id": "plank3.py", "kind": "module", "label": "plank3.py", "language": "py", "sha256": "7e0feddda6d04367", "symbol_count": 15, "symbols": [{"kind": "class", "line": 34, "name": "SpectralMonitor", "signature": "class SpectralMonitor"}, {"kind": "class", "line": 62, "name": "PersistentPruner", "signature": "class PersistentPruner"}, {"kind": "class", "line": 87, "name": "SpectralMLP", "signature": "class SpectralMLP(Module)"}, {"doc": "Train dense model until it reaches target accuracy.", "kind": "method", "line": 110, "name": "train_dense_to_target", "signature": "def train_dense_to_target(device, target_acc)"}, {"doc": "Progressively prune model and find critical density threshold.", "kind": "method", "line": 188, "name": "progressive_pruning_search", "signature": "def progressive_pruning_search(model, device, target_acc)"}, {"doc": "Find the minimum density where accuracy >= target_acc.", "kind": "method", "line": 264, "name": "find_critical_threshold", "signature": "def find_critical_threshold(pruning_df, target_acc)"}, {"kind": "method", "line": 294, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 35, "name": "__init__", "signature": "def __init__(self, epsilon_c)"}, {"kind": "method", "line": 38, "name": "compute_L", "signature": "def compute_L(self, weight)"}, {"kind": "method", "line": 63, "name": "__init__", "signature": "def __init__(self, sparsity_target)"}, {"kind": "method", "line": 67, "name": "apply_to_model", "signature": "def apply_to_model(self, model)"}, {"kind": "method", "line": 77, "name": "enforce_during_training", "signature": "def enforce_during_training(self, model)"}, {"kind": "method", "line": 88, "name": "__init__", "signature": "def __init__(self, input_dim, hidden_dim, num_classes)"}, {"kind": "method", "line": 96, "name": "reduce_input", "signature": "def reduce_input(self, x)"}, {"kind": "method", "line": 102, "name": "forward", "signature": "def forward(self, x)"}]}, {"doc": "NeuroSovereign v4.0: Fractal Sovereignty via Guided Lottery Tickets  This implementation executes a three-stage cycle: 1. Discover the optimal sparse subnetwork (Bekenstein Bound) 2. Embed it into a larger architecture as a \"truth seed\" 3. Re-prune to isolate a higher-capacity sparse model  Scientific contribution: - Validates that sparse subnetworks trained in high-capacity scaffolds outperform natively sparse models - Quantifies abstraction density per parameter - Provides empirical evidence for phase transitions in spectral coherence  Outputs: - sovereignty_v4_results.csv: Full ablation across cycles - best_model.pth: Final distilled model exceeding 32.4% accuracy - metrics.json: Key scientific findings  Ready for NeurIPS/ICLR submission.", "id": "plank4.py", "kind": "module", "label": "plank4.py", "language": "py", "sha256": "0c318f9e227776c6", "symbol_count": 20, "symbols": [{"kind": "class", "line": 41, "name": "SpectralMonitor", "signature": "class SpectralMonitor"}, {"kind": "class", "line": 66, "name": "PersistentPruner", "signature": "class PersistentPruner"}, {"kind": "class", "line": 88, "name": "SpectralMLP", "signature": "class SpectralMLP(Module)"}, {"kind": "class", "line": 111, "name": "FractalSovereigntyEngine", "signature": "class FractalSovereigntyEngine"}, {"kind": "method", "line": 329, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 42, "name": "__init__", "signature": "def __init__(self, epsilon_c)"}, {"kind": "method", "line": 45, "name": "compute_L", "signature": "def compute_L(self, weight)"}, {"kind": "method", "line": 67, "name": "__init__", "signature": "def __init__(self, sparsity_target)"}, {"kind": "method", "line": 71, "name": "apply_to_model", "signature": "def apply_to_model(self, model)"}, {"kind": "method", "line": 81, "name": "enforce_during_training", "signature": "def enforce_during_training(self, model)"}, {"kind": "method", "line": 89, "name": "__init__", "signature": "def __init__(self, input_dim, hidden_dim, num_classes)"}, {"kind": "method", "line": 97, "name": "reduce_input", "signature": "def reduce_input(self, x)"}, {"kind": "method", "line": 103, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 112, "name": "__init__", "signature": "def __init__(self, device, base_target_acc)"}, {"kind": "method", "line": 123, "name": "train_dense_model", "signature": "def train_dense_model(self, hidden_dim, target_acc)"}, {"kind": "method", "line": 168, "name": "extract_seed_weights", "signature": "def extract_seed_weights(self, model)"}, {"doc": "Embed seed into larger architecture", "kind": "method", "line": 174, "name": "inoculate_seed", "signature": "def inoculate_seed(self, large_model, seed_weights)"}, {"kind": "method", "line": 192, "name": "progressive_pruning", "signature": "def progressive_pruning(self, model, target_acc)"}, {"kind": "method", "line": 236, "name": "execute_cycle", "signature": "def execute_cycle(self, cycle, base_hidden_dim, expansion_factor)"}, {"kind": "method", "line": 281, "name": "run_experiment", "signature": "def run_experiment(self, num_cycles)"}]}, {"doc": "NeuroSovereign v6.0: Evolutionary Black Swan Chain CIFAR-10 unaltered dataset - Induced grokking via DNA propagation", "id": "plank5.py", "kind": "module", "label": "plank5.py", "language": "py", "sha256": "dd4bdb5869d54641", "symbol_count": 29, "symbols": [{"kind": "class", "line": 30, "name": "SpectralMonitor", "signature": "class SpectralMonitor"}, {"kind": "class", "line": 50, "name": "PersistentPruner", "signature": "class PersistentPruner"}, {"kind": "class", "line": 65, "name": "SpectralMLP", "signature": "class SpectralMLP(Module)"}, {"kind": "class", "line": 88, "name": "GrokkingDetector", "signature": "class GrokkingDetector"}, {"kind": "class", "line": 121, "name": "SyntheticBlackSwanGenerator", "signature": "class SyntheticBlackSwanGenerator"}, {"kind": "class", "line": 211, "name": "EvolutionCycle", "signature": "class EvolutionCycle"}, {"kind": "class", "line": 392, "name": "EvolutionaryBlackSwanChain", "signature": "class EvolutionaryBlackSwanChain"}, {"kind": "method", "line": 556, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 31, "name": "__init__", "signature": "def __init__(self, epsilon_c)"}, {"kind": "method", "line": 34, "name": "compute_L", "signature": "def compute_L(self, weight)"}, {"kind": "method", "line": 51, "name": "__init__", "signature": "def __init__(self, sparsity_target)"}, {"kind": "method", "line": 55, "name": "apply_to_model", "signature": "def apply_to_model(self, model)"}, {"kind": "method", "line": 66, "name": "__init__", "signature": "def __init__(self, input_dim, hidden_dim, num_classes)"}, {"kind": "method", "line": 74, "name": "reduce_input", "signature": "def reduce_input(self, x)"}, {"kind": "method", "line": 80, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 89, "name": "__init__", "signature": "def __init__(self, patience, gap_threshold)"}, {"kind": "method", "line": 94, "name": "update", "signature": "def update(self, train_acc, test_acc, epoch)"}, {"kind": "method", "line": 103, "name": "detect_grokking", "signature": "def detect_grokking(self)"}, {"kind": "method", "line": 122, "name": "__init__", "signature": "def __init__(self, device, target_acc, min_L)"}, {"doc": "Genera cisne negro sintético si no existe legacy", "kind": "method", "line": 128, "name": "generate", "signature": "def generate(self, hidden_dim)"}, {"kind": "method", "line": 212, "name": "__init__", "signature": "def __init__(self, device, base_acc)"}, {"doc": "Inocula ADN del cisne anterior con mutación controlada", "kind": "method", "line": 218, "name": "inoculate_dna", "signature": "def inoculate_dna(self, large_model, seed_weights, noise_scale)"}, {"doc": "Entrena modelo induciendo grokking y monitoreando transición de fase", "kind": "method", "line": 243, "name": "train_with_grokking", "signature": "def train_with_grokking(self, model, seed_model, target_acc)"}, {"doc": "Pruning progresivo para extraer nuevo cisne negro", "kind": "method", "line": 337, "name": "distill_sparse_model", "signature": "def distill_sparse_model(self, model, target_acc)"}, {"kind": "method", "line": 393, "name": "__init__", "signature": "def __init__(self, device, num_cycles, base_acc)"}, {"doc": "Carga legacy seed o genera uno sintético", "kind": "method", "line": 402, "name": "load_legacy_or_generate_seed", "signature": "def load_legacy_or_generate_seed(self)"}, {"doc": "Ejecuta la cadena evolutiva completa", "kind": "method", "line": 419, "name": "run_evolutionary_chain", "signature": "def run_evolutionary_chain(self)"}, {"doc": "Guarda resultados completos de la cadena evolutiva", "kind": "method", "line": 501, "name": "save_chain_results", "signature": "def save_chain_results(self)"}, {"doc": "Imprime resumen ejecutivo de la cadena evolutiva", "kind": "method", "line": 519, "name": "print_evolution_summary", "signature": "def print_evolution_summary(self)"}]}, {"doc": "NeuroSovereign v7.0: Guided Elk Hunting Evolution CIFAR-10 unaltered dataset - Induced grokking via DNA propagation", "id": "plank6.py", "kind": "module", "label": "plank6.py", "language": "py", "sha256": "3a19df68c80987f3", "symbol_count": 20, "symbols": [{"doc": "Calcula L (Coherencia Espectral) y Rank Efectivo", "kind": "class", "line": 29, "name": "SpectralMonitor", "signature": "class SpectralMonitor"}, {"doc": "Red Neuronal Base para el experimento", "kind": "class", "line": 51, "name": "SpectralMLP", "signature": "class SpectralMLP(Module)"}, {"kind": "class", "line": 78, "name": "GrokkingDetector", "signature": "class GrokkingDetector"}, {"doc": "Motor que toma el mejor modelo anterior (Elk), \nmuta sus pesos guiadamente y expande la arquitectura.", "kind": "class", "line": 101, "name": "GuidedElkHuntingEngine", "signature": "class GuidedElkHuntingEngine"}, {"kind": "class", "line": 199, "name": "TrainingCycle", "signature": "class TrainingCycle"}, {"kind": "method", "line": 293, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 31, "name": "__init__", "signature": "def __init__(self, epsilon_c)"}, {"kind": "method", "line": 34, "name": "compute_L", "signature": "def compute_L(self, weight)"}, {"kind": "method", "line": 53, "name": "__init__", "signature": "def __init__(self, input_dim, hidden_dim, num_classes)"}, {"kind": "method", "line": 63, "name": "reduce_input", "signature": "def reduce_input(self, x)"}, {"kind": "method", "line": 70, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 79, "name": "__init__", "signature": "def __init__(self, patience, gap_threshold)"}, {"kind": "method", "line": 84, "name": "update", "signature": "def update(self, train_acc, test_acc, epoch)"}, {"kind": "method", "line": 87, "name": "detect_grokking", "signature": "def detect_grokking(self)"}, {"kind": "method", "line": 106, "name": "__init__", "signature": "def __init__(self, device)"}, {"doc": "Evoluciona los pesos del Elk a una dimensión mayor manteniendo coherencia.", "kind": "method", "line": 110, "name": "_guided_elk_mutation", "signature": "def _guided_elk_mutation(self, old_weight, target_shape, noise_scale, refinement_steps)"}, {"doc": "Filtra componentes de baja energía y reconstruye", "kind": "method", "line": 146, "name": "_apply_spectral_refinement", "signature": "def _apply_spectral_refinement(self, W)"}, {"doc": "Crea un nuevo modelo (Cisne Negro) basado en el Elk (mejor modelo previo).", "kind": "method", "line": 158, "name": "create_offspring_from_elk", "signature": "def create_offspring_from_elk(self, elk_state, new_hidden_dim, generation)"}, {"kind": "method", "line": 200, "name": "__init__", "signature": "def __init__(self, device)"}, {"kind": "method", "line": 204, "name": "train_phase", "signature": "def train_phase(self, model, cycle_id)"}]}, {"doc": "NeuroSovereign v8.0: Shock Therapy & Gradient Nudging Target: Break Generalization Plateau via Gradient Nudging & Curriculum Learning", "id": "plank7.py", "kind": "module", "label": "plank7.py", "language": "py", "sha256": "552f5ca87579108d", "symbol_count": 17, "symbols": [{"kind": "class", "line": 29, "name": "SpectralMonitor", "signature": "class SpectralMonitor"}, {"kind": "class", "line": 45, "name": "SpectralMLP", "signature": "class SpectralMLP(Module)"}, {"doc": "Implementa Gradient Nudging y Espectral Shock.", "kind": "class", "line": 68, "name": "AdvancedEvolutionEngine", "signature": "class AdvancedEvolutionEngine"}, {"kind": "class", "line": 177, "name": "CurriculumTrainingCycle", "signature": "class CurriculumTrainingCycle"}, {"kind": "method", "line": 283, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 30, "name": "__init__", "signature": "def __init__(self, epsilon_c)"}, {"kind": "method", "line": 33, "name": "compute_L", "signature": "def compute_L(self, weight)"}, {"kind": "method", "line": 46, "name": "__init__", "signature": "def __init__(self, input_dim, hidden_dim, num_classes)"}, {"kind": "method", "line": 54, "name": "reduce_input", "signature": "def reduce_input(self, x)"}, {"kind": "method", "line": 60, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 72, "name": "__init__", "signature": "def __init__(self, device)"}, {"doc": "Antes de entrenar, hacemos 1 paso de gradiente del Elk sobre los nuevos datos.\nEsto 'pre-ajusta' el ADN al contexto actual.", "kind": "method", "line": 76, "name": "_gradient_nudge_inheritance", "signature": "def _gradient_nudge_inheritance(self, child_model, elk_state, data_loader, nudge_lr)"}, {"doc": "Aplica una perturbación no-lineal a los valores singulares.\nEsto rompe mínimos locales planos sin destruir la estructura global.", "kind": "method", "line": 107, "name": "_apply_spectral_shock", "signature": "def _apply_spectral_shock(self, W, shock_intensity)"}, {"doc": "Crea un hijo combinando:\n1. Herencia de pesos\n2. Gradient Nudge (context awareness)\n3. Spectral Shock (ruptura de estancamiento)", "kind": "method", "line": 124, "name": "create_advanced_offspring", "signature": "def create_advanced_offspring(self, elk_state, new_hidden_dim, cycle, data_loader)"}, {"kind": "method", "line": 178, "name": "__init__", "signature": "def __init__(self, device)"}, {"doc": "Estrategia de Curriculum:\nCiclos 1-3: Subset pequeño (Foco en estructura).\nCiclos 4+: Expansión progresiva (Foco en generalización).", "kind": "method", "line": 186, "name": "get_curriculum_dataset", "signature": "def get_curriculum_dataset(self, cycle)"}, {"kind": "method", "line": 204, "name": "train_phase", "signature": "def train_phase(self, model, cycle)"}]}, {"doc": "NeuroSovereign v11.0: Visionary Apex Features: 1. ViT-Lite Patch Embedding Extractor (Modern Vision Backbone). 2. Hybrid Architecture: Patch Tokens -> Spectral Lottery MLP. 3. Apex Evolution Engine (Nudge, Dynamic Shock, Sparsity). 4. Objective: SOTA Sparsity/Efficiency with modern representation.", "id": "plank8.py", "kind": "module", "label": "plank8.py", "language": "py", "sha256": "4a3f3e71f842892d", "symbol_count": 26, "symbols": [{"doc": "Extrae características mediante Patch Embedding.\nConvierte imagen (B, 3, 32, 32) en secuencia de parches proyectados.", "kind": "class", "line": 34, "name": "PatchFeatureExtractor", "signature": "class PatchFeatureExtractor(Module)"}, {"kind": "class", "line": 78, "name": "LotteryMLP", "signature": "class LotteryMLP(Module)"}, {"doc": "Baseline moderno (Patch + MLP simple) sin evolución.", "kind": "class", "line": 109, "name": "StandardBaseline", "signature": "class StandardBaseline(Module)"}, {"kind": "class", "line": 121, "name": "SpectralMonitor", "signature": "class SpectralMonitor"}, {"kind": "class", "line": 136, "name": "ApexEvolutionEngine", "signature": "class ApexEvolutionEngine"}, {"kind": "class", "line": 235, "name": "ApexTrainer", "signature": "class ApexTrainer"}, {"kind": "method", "line": 342, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 39, "name": "__init__", "signature": "def __init__(self, img_size, patch_size, in_chans, embed_dim)"}, {"kind": "method", "line": 57, "name": "freeze", "signature": "def freeze(self)"}, {"kind": "method", "line": 61, "name": "unfreeze", "signature": "def unfreeze(self)"}, {"kind": "method", "line": 65, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 79, "name": "__init__", "signature": "def __init__(self, input_dim, hidden_dim, num_classes)"}, {"kind": "method", "line": 92, "name": "apply_masks", "signature": "def apply_masks(self)"}, {"kind": "method", "line": 97, "name": "get_sparsity", "signature": "def get_sparsity(self)"}, {"kind": "method", "line": 102, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 111, "name": "__init__", "signature": "def __init__(self, input_dim, hidden_dim)"}, {"kind": "method", "line": 115, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 122, "name": "compute_L", "signature": "def compute_L(self, weight)"}, {"kind": "method", "line": 137, "name": "__init__", "signature": "def __init__(self, device)"}, {"kind": "method", "line": 142, "name": "_gradient_nudge_inheritance", "signature": "def _gradient_nudge_inheritance(self, child_model, elk_state, data_loader, feature_extractor, nudge_lr)"}, {"kind": "method", "line": 176, "name": "_apply_dynamic_spectral_shock", "signature": "def _apply_dynamic_spectral_shock(self, model, layer_name)"}, {"kind": "method", "line": 199, "name": "create_apex_offspring", "signature": "def create_apex_offspring(self, elk_state, new_hidden_dim, cycle, data_loader, feature_extractor, parent_gap)"}, {"kind": "method", "line": 236, "name": "__init__", "signature": "def __init__(self, device, feature_extractor)"}, {"kind": "method", "line": 243, "name": "_preprocess_batch", "signature": "def _preprocess_batch(self, x)"}, {"kind": "method", "line": 247, "name": "get_curriculum_dataset", "signature": "def get_curriculum_dataset(self, cycle)"}, {"kind": "method", "line": 253, "name": "train_model", "signature": "def train_model(self, model, cycle, is_baseline)"}]}, {"id": "resmav2_1.py", "kind": "module", "label": "resmav2_1.py", "language": "py", "sha256": "685cb21c7fd17974", "symbol_count": 18, "symbols": [{"doc": "Optimized E8 with caching and efficiency improvements", "kind": "class", "line": 23, "name": "OptimizedE8Layer", "signature": "class OptimizedE8Layer(Module)"}, {"doc": "Fast version: E8 + GAT fusion with minimal overhead", "kind": "class", "line": 48, "name": "RESMAv2Fast", "signature": "class RESMAv2Fast(Module)"}, {"doc": "Standard version: 2 layers of E8 + GAT fusion", "kind": "class", "line": 86, "name": "RESMAv2Standard", "signature": "class RESMAv2Standard(Module)"}, {"doc": "Deeper version with 3 layers", "kind": "class", "line": 134, "name": "RESMAv2Deep", "signature": "class RESMAv2Deep(Module)"}, {"doc": "Optimized GAT baseline", "kind": "class", "line": 182, "name": "GAT_Baseline", "signature": "class GAT_Baseline(Module)"}, {"kind": "method", "line": 207, "name": "load_elliptic_data", "signature": "def load_elliptic_data()"}, {"kind": "method", "line": 264, "name": "train_and_evaluate", "signature": "def train_and_evaluate(model, X, y, edge_index, train_idx, val_idx, epochs, lr, name, fold)"}, {"kind": "method", "line": 321, "name": "cross_validate_model", "signature": "def cross_validate_model(model_class, X, y, edge_index, num_nodes, n_splits, seed, name)"}, {"kind": "method", "line": 25, "name": "__init__", "signature": "def __init__(self, in_features, out_features, edge_index, num_nodes)"}, {"kind": "method", "line": 40, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 50, "name": "__init__", "signature": "def __init__(self, input_dim, hidden_dim, edge_index, num_nodes, dropout)"}, {"kind": "method", "line": 73, "name": "forward", "signature": "def forward(self, x, edge_index)"}, {"kind": "method", "line": 88, "name": "__init__", "signature": "def __init__(self, input_dim, hidden_dim, edge_index, num_nodes, dropout)"}, {"kind": "method", "line": 113, "name": "forward", "signature": "def forward(self, x, edge_index)"}, {"kind": "method", "line": 136, "name": "__init__", "signature": "def __init__(self, input_dim, hidden_dim, edge_index, num_nodes, dropout)"}, {"kind": "method", "line": 168, "name": "forward", "signature": "def forward(self, x, edge_index)"}, {"kind": "method", "line": 184, "name": "__init__", "signature": "def __init__(self, input_dim, hidden_dim, dropout)"}, {"kind": "method", "line": 194, "name": "forward", "signature": "def forward(self, x, edge_index)"}]}], "type": "CodePropertyGraph", "version": "1.0"}
```

---

## Architecture Reference

### PY (36 files)

#### `apex14.py`
**Path:** `apex14.py`
**File Doc:** *NeuroSovereign v14.0: Hierarchical Apex (FIXED) Objective: Prove Structural Necessity on CIFAR-100 (Hierarchical Task). Features: 1. Dataset Upgrade: CIFAR-10 -> CIFAR-100 (Requires feature composition). 2. Gated Token Mixer: Content-aware mixing (not just static linear). 3. Gate Usage Metric: Implicit ablation to prove mixer activity.*

**Classes:**
- `GatedTokenMixer` (line 36) `class GatedTokenMixer(Module)`
- `PatchFeatureExtractor` (line 66) `class PatchFeatureExtractor(Module)` - *Extractor configurable para CIFAR-100.
use_mixer=True -> Apex (Syntactic)
use_mixer=False -> Blind Structural Baseline*
- `LotteryMLP` (line 111) `class LotteryMLP(Module)`
- `SpectralMonitor` (line 142) `class SpectralMonitor`
- `OrthogonalEvolutionEngine` (line 158) `class OrthogonalEvolutionEngine`
- `HierarchicalTrainer` (line 231) `class HierarchicalTrainer`

**Methods:**
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
**File Doc:** *NeuroSovereign v14.4: SOTA Structural Control & Emergency Shock Objective: Stabilize Training via Active Spectral Regularization & Dynamic Shock. Changes from v14.3: 1. FIX: Frozen fc_super in BLIND to prevent NoneType gradient errors. 2. REMOVED: Genetic Nudge (Replaced by direct state inheritance for reproducibility). 3. ADDED: Spectral Entropy Loss (Active L-Metric) to prevent memory collapse. 4. ADDED: Taxonomic Shock Logic (Lambda boost if Gap > 5.0). 5. ADDED: Mixer LR Injection (5x learning rate for Mixer weights).*

**Classes:**
- `GatedTokenMixer` (line 88) `class GatedTokenMixer(Module)`
- `PatchFeatureExtractor` (line 105) `class PatchFeatureExtractor(Module)`
- `TaxonomicMLP` (line 143) `class TaxonomicMLP(Module)`
- `SpectralMonitor` (line 179) `class SpectralMonitor`
- `TaxonomicTrainer` (line 195) `class TaxonomicTrainer`

**Functions:**
- `compute_spectral_loss` (line 63) `def compute_spectral_loss(W, target_rank_factor)` - *Penaliza la desalineación entre Entropía Espectral y Rango Efectivo.
Esta es la versión 'activa' de la métrica L.*

**Methods:**
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
**File Doc:** *NeuroSovereign v14.5: Adaptive Shock & Full Spectrum Control Objective: Fix v14.4 based on technical review. Changes from v14.4: 1. FIX: Adaptive Taxonomic Shock (Lambda reacts to REAL validation gap, not epochs). 2. ADDED: Spectral Loss applied to Mixer weights (Upstream structural control). 3. ADDED: Feedback Loop Logic (Prev epoch gap determines current epoch lambda). 4. TUNED: Gap Threshold set to 5.0 for Shock activation.*

**Classes:**
- `GatedTokenMixer` (line 82) `class GatedTokenMixer(Module)`
- `PatchFeatureExtractor` (line 99) `class PatchFeatureExtractor(Module)`
- `TaxonomicMLP` (line 137) `class TaxonomicMLP(Module)`
- `SpectralMonitor` (line 173) `class SpectralMonitor`
- `TaxonomicTrainer` (line 189) `class TaxonomicTrainer`

**Functions:**
- `compute_spectral_loss` (line 63) `def compute_spectral_loss(W, target_rank_factor)` - *Penaliza la desalineación entre Entropía Espectral y Rango Efectivo.
v14.5: Se aplicará a pesos del MLP y del Mixer.*

**Methods:**
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
**File Doc:** *NeuroSovereign v15.0: Paper Candidate Release Objective: Finalize architecture for rigorous review. Changes from v14.5: 1. CLARITY: Separated L_opt (Optimization Objective) vs L_mon (Reporting Metric) in logs. 2. BENCHMARK: Added CIFAR-20 Stress Test (Coarse-only validation). 3. LOGIC: Final validation based on Delta (APEX - BLIND) to prove structural advantage.*

**Classes:**
- `GatedTokenMixer` (line 81) `class GatedTokenMixer(Module)`
- `PatchFeatureExtractor` (line 98) `class PatchFeatureExtractor(Module)`
- `TaxonomicMLP` (line 136) `class TaxonomicMLP(Module)`
- `SpectralMonitor` (line 172) `class SpectralMonitor`
- `TaxonomicTrainer` (line 190) `class TaxonomicTrainer`
- `CoarseCIFAR100` (line 368) `class CoarseCIFAR100(CIFAR100)` - *Wrapper que convierte CIFAR100 en un problema de clasificación pura de 20 clases (Superclases).
Se usa para validar el inductive bias aprendido.*

**Functions:**
- `compute_spectral_loss` (line 65) `def compute_spectral_loss(W)` - *v15.0: Optimization Objective for Spectral Control.*

**Methods:**
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
**File Doc:** *NeuroSovereign v15.1: Symmetric Control (The Reviewer's Fix) Objective: Isolate the Hierarchy Signal by equating Regularization budgets. Changes from v15.0: 1. FIX: BLIND now includes Spectral Loss (L_opt) on FC1. 2. FIX: BLIND now includes Sparse Penalty (applies where applicable). 3. LOGIC: APEX vs BLIND comparison is now valid; only difference is Hierarchy Signal (CE_coarse).*

**Classes:**
- `GatedTokenMixer` (line 81) `class GatedTokenMixer(Module)`
- `PatchFeatureExtractor` (line 98) `class PatchFeatureExtractor(Module)`
- `TaxonomicMLP` (line 136) `class TaxonomicMLP(Module)`
- `SpectralMonitor` (line 172) `class SpectralMonitor`
- `TaxonomicTrainer` (line 188) `class TaxonomicTrainer`
- `CoarseCIFAR100` (line 369) `class CoarseCIFAR100(CIFAR100)`

**Functions:**
- `compute_spectral_loss` (line 65) `def compute_spectral_loss(W)` - *v15.1: Optimization Objective for Spectral Control (Applied to both APEX and BLIND).*

**Methods:**
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
**File Doc:** *NeuroSovereign v15.2: Topological Phase Detection Objective: Detect Grokking moments via Topological Ratio (L_opt / L_mon). Changes from v15.1: 1. ADDED: TopologicalMonitor to track Phase Shift (Ratio R = L_opt / L_mon). 2. LOGIC: Identification of "Stagnation Phase" vs "Plasticity Phase". 3. FOCUS: Run Cycle 0 (Seeding) to prove Architectural Necessity vs Regularization sufficiency.*

**Classes:**
- `GatedTokenMixer` (line 84) `class GatedTokenMixer(Module)`
- `PatchFeatureExtractor` (line 101) `class PatchFeatureExtractor(Module)`
- `TaxonomicMLP` (line 139) `class TaxonomicMLP(Module)`
- `SpectralMonitor` (line 175) `class SpectralMonitor`
- `TaxonomicTrainer` (line 211) `class TaxonomicTrainer`
- `CoarseCIFAR100` (line 394) `class CoarseCIFAR100(CIFAR100)`

**Functions:**
- `compute_spectral_loss` (line 68) `def compute_spectral_loss(W)` - *L_opt: Optimization Objective for Structural Control.*

**Methods:**
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
**File Doc:** *NeuroSovereign v15.3: Relative Phase & Cross-Scale Monitor Objective: Fix arbitrary thresholds and spatial leakage for Paper-Ready Rigor. Changes from v15.2: 1. FIX: Phase Detection is now Relative (based on deviation from history mean). 2. FIX: Hierarchy Benchmark uses correct forward pass flow (fixes feature space mixing). 3. LOGIC: Explicit separation of L_opt components (Upstream vs Downstream) in logs.*

**Classes:**
- `GatedTokenMixer` (line 85) `class GatedTokenMixer(Module)`
- `PatchFeatureExtractor` (line 102) `class PatchFeatureExtractor(Module)`
- `TaxonomicMLP` (line 140) `class TaxonomicMLP(Module)`
- `SpectralMonitor` (line 176) `class SpectralMonitor`
- `TaxonomicTrainer` (line 242) `class TaxonomicTrainer`
- `CoarseCIFAR100` (line 426) `class CoarseCIFAR100(CIFAR100)`

**Functions:**
- `compute_spectral_loss` (line 69) `def compute_spectral_loss(W)` - *L_opt: Optimization Objective for Structural Control.*

**Methods:**
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
**File Doc:** *NeuroSovereign v15.4: Active Phase Intervention (Breaking the Ceiling) Objective: Move from Observation to Causal Control. Changes from v15.3: 1. ADDED: TopologyController to manage Phase Interventions. 2. LOGIC: If Phase is STABLE + Low Coarse Acc -> Inject Topological Noise (Reversibility). 3. ADDED: Stagnation Counter to trigger "Active Shocks". 4. FIX: Implemented causal intervention to break local minima in hierarchy learning.*

**Classes:**
- `GatedTokenMixer` (line 88) `class GatedTokenMixer(Module)`
- `PatchFeatureExtractor` (line 105) `class PatchFeatureExtractor(Module)`
- `TaxonomicMLP` (line 143) `class TaxonomicMLP(Module)`
- `SpectralMonitor` (line 179) `class SpectralMonitor`
- `TopologyController` (line 228) `class TopologyController` - *v15.4: Manages Active Interventions to break stagnation.*
- `TaxonomicTrainer` (line 270) `class TaxonomicTrainer`
- `CoarseCIFAR100` (line 459) `class CoarseCIFAR100(CIFAR100)`

**Functions:**
- `compute_spectral_loss` (line 73) `def compute_spectral_loss(W)`

**Methods:**
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
**File Doc:** *NeuroSovereign v15.5: Targeted Phase Surgery (Spectral Noise) Objective: Fix "Blind Noise" critique by using Orthogonal Projection. Changes from v15.4: 1. FIX: perturb_mixer() now uses Targeted Spectral Noise (Orthogonal to dominant subspace). 2. LOGIC: Intervention preserves dominant features while exciting latent modes. 3. MATH: Projection matrix P = I - V_dominant @ V_dominant.T ensures noise injection only in weak dimensions.*

**Classes:**
- `GatedTokenMixer` (line 88) `class GatedTokenMixer(Module)`
- `PatchFeatureExtractor` (line 105) `class PatchFeatureExtractor(Module)`
- `TaxonomicMLP` (line 143) `class TaxonomicMLP(Module)`
- `SpectralMonitor` (line 179) `class SpectralMonitor`
- `TopologyController` (line 204) `class TopologyController` - *v15.5: Manages Targeted Spectral Interventions (Surgery).*
- `TaxonomicTrainer` (line 273) `class TaxonomicTrainer`
- `CoarseCIFAR100` (line 459) `class CoarseCIFAR100(CIFAR100)`

**Functions:**
- `compute_spectral_loss` (line 73) `def compute_spectral_loss(W)`

**Methods:**
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
**File Doc:** *NeuroSovereign v15.5: Targeted Phase Surgery (FINAL RELEASE) Objective: Break Hierarchy Ceiling via Orthogonal Spectral Projection. Fixes Applied: 1. Syntax correction in compute_spectral_loss. 2. Implementation of compute_topology_ratio for Phase Monitoring. 3. Geometric fix in Orthogonal Projection (Input Space Injection). 4. Robust State Management (Migration from v15.4).*

**Classes:**
- `GatedTokenMixer` (line 91) `class GatedTokenMixer(Module)`
- `PatchFeatureExtractor` (line 108) `class PatchFeatureExtractor(Module)`
- `TaxonomicMLP` (line 146) `class TaxonomicMLP(Module)`
- `SpectralMonitor` (line 182) `class SpectralMonitor`
- `TopologyController` (line 228) `class TopologyController`
- `TaxonomicTrainer` (line 306) `class TaxonomicTrainer`
- `CoarseCIFAR100` (line 493) `class CoarseCIFAR100(CIFAR100)`

**Functions:**
- `compute_spectral_loss` (line 74) `def compute_spectral_loss(W)` - *L_opt: Computes the discrepancy between spectral entropy and effective rank.*

**Methods:**
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
**File Doc:** *NeuroSovereign v15.5: Targeted Phase Surgery (FINAL RELEASE) Objective: Break Hierarchy Ceiling via Orthogonal Spectral Projection. Fixes Applied: 1. Geometric Intervention Logic (Topo-R vs CV mismatch detection). 2. Implementation of compute_topology_ratio for Phase Monitoring. 3. Geometric fix in Orthogonal Projection (Input Space Injection). 4. Robust State Management (Migration from v15.4).*

**Classes:**
- `GatedTokenMixer` (line 92) `class GatedTokenMixer(Module)`
- `PatchFeatureExtractor` (line 109) `class PatchFeatureExtractor(Module)`
- `TaxonomicMLP` (line 147) `class TaxonomicMLP(Module)`
- `SpectralMonitor` (line 183) `class SpectralMonitor`
- `TopologyController` (line 229) `class TopologyController`
- `TaxonomicTrainer` (line 339) `class TaxonomicTrainer`
- `CoarseCIFAR100` (line 527) `class CoarseCIFAR100(CIFAR100)`

**Functions:**
- `compute_spectral_loss` (line 75) `def compute_spectral_loss(W)` - *L_opt: Computes the discrepancy between spectral entropy and effective rank.*

**Methods:**
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
**File Doc:** *NeuroSovereign v15.5: Targeted Phase Surgery (FINAL RELEASE) Objective: Break Hierarchy Ceiling via Orthogonal Spectral Projection. Status: Validated for High-Plasticity Init & Nullspace Surgery. Key Features: 1. Geometric Mismatch Detection (d(Topo_R) vs d(C.V)). 2. Orthogonal Noise Injection (Nullspace Surgery). 3. Topology Ratio (L_opt / L_mon) for Phase Monitoring.*

**Classes:**
- `GatedTokenMixer` (line 89) `class GatedTokenMixer(Module)`
- `PatchFeatureExtractor` (line 106) `class PatchFeatureExtractor(Module)`
- `TaxonomicMLP` (line 144) `class TaxonomicMLP(Module)`
- `SpectralMonitor` (line 180) `class SpectralMonitor`
- `TopologyController` (line 224) `class TopologyController`
- `TaxonomicTrainer` (line 316) `class TaxonomicTrainer`
- `CoarseCIFAR100` (line 504) `class CoarseCIFAR100(CIFAR100)`

**Functions:**
- `compute_spectral_loss` (line 75) `def compute_spectral_loss(W)` - *L_opt: Computes the discrepancy between spectral entropy and effective rank.*

**Methods:**
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
**File Doc:** *NeuroSovereign v17.0: Evolutionary Taxonomic Optimization Production-ready implementation with configurable iterations and statistical validation  Key improvements over v15.5: 1. Configurable evolutionary iterations (default: 20) 2. Statistical validation with 5 seeds per configuration 3. Early stopping with adaptive patience 4. Rigorous benchmarking against 3 baselines 5. Production-ready logging and checkpointing 6. Memory optimization for large-scale training 7. Complete reproducibility with fixed seeds 8. Targeted Spectral Surgery with Nullspace Injection  Outputs: - evolutionary_results.csv: Complete metrics across iterations - taxonomic_report.json: Statistical summary with confidence intervals - best_model_apex.pth / best_model_blind.pth: Final evolved models - evolution_curves.png: Publication-quality visualization*

**Classes:**
- `GatedTokenMixer` (line 89) `class GatedTokenMixer(Module)` - *Efficient token mixer with gating mechanism for feature interaction*
- `PatchFeatureExtractor` (line 146) `class PatchFeatureExtractor(Module)` - *Efficient patch-based feature extractor with optional token mixing*
- `TaxonomicMLP` (line 207) `class TaxonomicMLP(Module)` - *Sparse MLP with taxonomic heads for hierarchical learning*
- `SpectralMonitor` (line 281) `class SpectralMonitor` - *Monitor spectral properties of weight matrices for evolutionary guidance*
- `TopologyController` (line 302) `class TopologyController` - *Advanced controller for targeted spectral surgery*
- `EvolutionaryEngine` (line 410) `class EvolutionaryEngine` - *Engine for evolving neural networks through spectral refinement*
- `CoarseCIFAR100` (line 480) `class CoarseCIFAR100(CIFAR100)` - *Wrapper that converts CIFAR100 into a pure 20-class classification problem (Superclasses).
Used to validate learned inductive bias.*
- `EvolutionaryTrainer` (line 544) `class EvolutionaryTrainer` - *Framework for evolutionary training with statistical validation*

**Functions:**
- `set_seed` (line 43) `def set_seed(seed)` - *Ensure full reproducibility across runs*

**Methods:**
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
**File Doc:** *NeuroSovereign v18.0: Adaptive Iterative Spectral Refinement Production-ready implementation addressing v17.0 technical review.  Key improvements over v17.0 (Review Implementation): 1. Replaced fixed threshold with DynamicThresholdController (Percentile-based). 2. Renamed "Evolutionary" to "Iterative Refinement" for scientific accuracy. 3. Added Singular Value tracking and visualization (Spectral Map). 4. Added Ablation flags via CLI for rigorous validation. 5. Improved Nullspace Surgery logging for transparency.  Scientific Goal: "Demonstrate controlled induction of hierarchical structure beyond symmetric spectral regularization."*

**Classes:**
- `GatedTokenMixer` (line 83) `class GatedTokenMixer(Module)` - *Efficient token mixer with gating mechanism*
- `PatchFeatureExtractor` (line 124) `class PatchFeatureExtractor(Module)` - *Efficient patch-based feature extractor*
- `TaxonomicMLP` (line 170) `class TaxonomicMLP(Module)` - *Sparse MLP with taxonomic heads*
- `DynamicThresholdController` (line 233) `class DynamicThresholdController` - *v18 Improvement: Replaces fixed TARGET_COARSE_V with adaptive logic.
Triggers intervention if current performance stagnates relative to its own history.*
- `SpectralMonitor` (line 255) `class SpectralMonitor` - *Monitor spectral properties*
- `TopologyController` (line 275) `class TopologyController` - *v18 Improvement: Advanced controller with Adaptive Thresholding.
Implements Targeted Spectral Surgery with Nullspace Injection.*
- `IterativeRefinementEngine` (line 369) `class IterativeRefinementEngine` - *v18: Engine for iterative refinement (formerly Evolutionary)*
- `IterativeTrainer` (line 416) `class IterativeTrainer` - *Framework for Iterative Refinement with v18 Adaptive Control*

**Functions:**
- `set_seed` (line 38) `def set_seed(seed)` - *Ensure full reproducibility across runs*

**Methods:**
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
**File Doc:** *NeuroSovereign v18.0: Iterative Spectral Refinement with Adaptive Control Addressing all senior reviewer concerns from v17:  1. REPLACED fixed TARGET_COARSE_V with adaptive semantic-plasticity ratio 2. RENAMED "evolutionary" to "iterative refinement" throughout 3. REFACTORED topology controller for dataset-agnostic operation 4. ADDED singular value visualization for paper figures 5. IMPLEMENTED ablation-ready architecture variants 6. OPTIMIZED SVD operations for production scalability  This implementation delivers: - Statistically validated hierarchical advantage (1.0%+ over symmetric baseline) - Self-referential control without dataset-specific thresholds - Publication-ready visualizations of spectral dynamics - Production-grade reproducibility and checkpointing  Outputs: - refinement_results.csv: Complete metrics across iterations - spectral_analysis/ directory: Singular value visualizations - taxonomic_report.json: Statistical summary with confidence intervals - best_model_apex.pth / best_model_blind.pth: Final refined models - refinement_curves.png: Publication-quality visualization*

**Classes:**
- `GatedTokenMixer` (line 93) `class GatedTokenMixer(Module)` - *Efficient token mixer with gating mechanism for feature interaction*
- `PatchFeatureExtractor` (line 150) `class PatchFeatureExtractor(Module)` - *Efficient patch-based feature extractor with optional token mixing*
- `TaxonomicMLP` (line 211) `class TaxonomicMLP(Module)` - *Sparse MLP with taxonomic heads for hierarchical learning*
- `SpectralMonitor` (line 285) `class SpectralMonitor` - *Monitor spectral properties with adaptive analysis*
- `AdaptiveTopologyController` (line 316) `class AdaptiveTopologyController` - *Self-referential controller using semantic-plasticity ratio*
- `IterativeRefinementEngine` (line 415) `class IterativeRefinementEngine` - *Engine for iterative refinement through spectral control*
- `CoarseCIFAR100` (line 484) `class CoarseCIFAR100(CIFAR100)` - *Wrapper that converts CIFAR100 into a pure 20-class classification problem (Superclasses).
Used to validate learned inductive bias.*
- `IterativeRefinementTrainer` (line 611) `class IterativeRefinementTrainer` - *Framework for iterative refinement with statistical validation*

**Functions:**
- `set_seed` (line 47) `def set_seed(seed)` - *Ensure full reproducibility across runs*

**Methods:**
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
**File Doc:** *NeuroSovereign v18.0: Iterative Spectral Refinement with Adaptive Control Addressing all senior reviewer concerns from v17: 1. REPLACED fixed TARGET_COARSE_V with adaptive semantic-plasticity ratio 2. RENAMED "evolutionary" to "iterative refinement" throughout 3. REFACTORED topology controller for dataset-agnostic operation 4. ADDED singular value visualization for paper figures 5. IMPLEMENTED ablation-ready architecture variants 6. OPTIMIZED SVD operations for production scalability This implementation delivers: - Statistically validated hierarchical advantage (1.0%+ over symmetric baseline) - Self-referential control without dataset-specific thresholds - Publication-ready visualizations of spectral dynamics - Production-grade reproducibility and checkpointing Outputs: - refinement_results.csv: Complete metrics across iterations - spectral_analysis/ directory: Singular value visualizations - taxonomic_report.json: Statistical summary with confidence intervals - best_model_apex.pth / best_model_blind.pth: Final refined models - refinement_curves.png: Publication-quality visualization*

**Classes:**
- `GatedTokenMixer` (line 86) `class GatedTokenMixer(Module)` - *Efficient token mixer with gating mechanism for feature interaction*
- `PatchFeatureExtractor` (line 143) `class PatchFeatureExtractor(Module)` - *Efficient patch-based feature extractor with optional token mixing*
- `TaxonomicMLP` (line 204) `class TaxonomicMLP(Module)` - *Sparse MLP with taxonomic heads for hierarchical learning*
- `SpectralMonitor` (line 278) `class SpectralMonitor` - *Monitor spectral properties with adaptive analysis*
- `AdaptiveTopologyController` (line 309) `class AdaptiveTopologyController` - *Self-referential controller using semantic-plasticity ratio*
- `IterativeRefinementEngine` (line 408) `class IterativeRefinementEngine` - *Engine for iterative refinement through spectral control*
- `CoarseCIFAR100` (line 476) `class CoarseCIFAR100(CIFAR100)` - *Wrapper that converts CIFAR100 into a pure 20-class classification problem (Superclasses).
Used to validate learned inductive bias.*
- `IterativeRefinementTrainer` (line 606) `class IterativeRefinementTrainer` - *Framework for iterative refinement with statistical validation*

**Functions:**
- `set_seed` (line 43) `def set_seed(seed)` - *Ensure full reproducibility across runs*

**Methods:**
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
**File Doc:** *NeuroSovereign v18.1: Production-Grade Iterative Spectral Refinement Hardened Implementation based on Senior Review Feedback (v18.0 -> v18.1)  Key Corrections in v18.1: 1. [FIXED] Critical Scope Error: visualize_singular_values now accepts monitor object. 2. [FIXED] Plotting Logic: _plot_results explicitly handles best_overall dict. 3. [REFACTOR] Removed Hardcoded Logic: 28.0 is now REFERENCE_BASELINE only (not control flow). 4. [OPTIMIZED] Control Flow: Clarified hysteresis in AdaptiveTopologyController.  Validated Claims: - Statistically validated hierarchical advantage (>1.0% over baseline) - Zero-shot CIFAR-20 transfer validation - Dataset-agnostic control (no fixed coarse accuracy targets) - Ablation-ready architecture*

**Classes:**
- `GatedTokenMixer` (line 87) `class GatedTokenMixer(Module)` - *Efficient token mixer with gating mechanism for feature interaction*
- `PatchFeatureExtractor` (line 142) `class PatchFeatureExtractor(Module)`
- `TaxonomicMLP` (line 185) `class TaxonomicMLP(Module)` - *Sparse MLP with taxonomic heads for hierarchical learning*
- `SpectralMonitor` (line 248) `class SpectralMonitor` - *Monitor spectral properties with adaptive analysis*
- `AdaptiveTopologyController` (line 277) `class AdaptiveTopologyController` - *Self-referential controller using semantic-plasticity ratio*
- `IterativeRefinementEngine` (line 397) `class IterativeRefinementEngine` - *Engine for iterative refinement through spectral control*
- `CoarseCIFAR100` (line 459) `class CoarseCIFAR100(CIFAR100)`
- `IterativeRefinementTrainer` (line 570) `class IterativeRefinementTrainer`

**Functions:**
- `set_seed` (line 43) `def set_seed(seed)` - *Ensure full reproducibility across runs*

**Methods:**
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
**File Doc:** *NeuroSovereign v18.1: Production-Grade Iterative Spectral Refinement Hardened Implementation based on Senior Review Feedback (v18.0 -> v18.1)  Key Corrections in v18.1: 1. [FIXED] Critical Scope Error: visualize_singular_values now accepts monitor object. 2. [FIXED] Plotting Logic: _plot_results explicitly handles best_overall dict. 3. [REFACTOR] Removed Hardcoded Logic: 28.0 is now REFERENCE_BASELINE only (not control flow). 4. [OPTIMIZED] Control Flow: Clarified hysteresis in AdaptiveTopologyController. 5. [HARDENED] Updated torch.svd to torch.linalg.svd (future-proofing). 6. [HARDENED] Increased initialization std in GatedTokenMixer (v15.3 exploration logic).  Validated Claims: - Statistically validated hierarchical advantage (>1.0% over baseline) - Zero-shot CIFAR-20 transfer validation - Dataset-agnostic control (no fixed coarse accuracy targets) - Ablation-ready architecture*

**Classes:**
- `GatedTokenMixer` (line 89) `class GatedTokenMixer(Module)` - *Efficient token mixer with gating mechanism for feature interaction*
- `PatchFeatureExtractor` (line 144) `class PatchFeatureExtractor(Module)`
- `TaxonomicMLP` (line 187) `class TaxonomicMLP(Module)` - *Sparse MLP with taxonomic heads for hierarchical learning*
- `SpectralMonitor` (line 250) `class SpectralMonitor` - *Monitor spectral properties with adaptive analysis*
- `AdaptiveTopologyController` (line 279) `class AdaptiveTopologyController` - *Self-referential controller using semantic-plasticity ratio*
- `IterativeRefinementEngine` (line 404) `class IterativeRefinementEngine` - *Engine for iterative refinement through spectral control*
- `CoarseCIFAR100` (line 470) `class CoarseCIFAR100(CIFAR100)`
- `IterativeRefinementTrainer` (line 581) `class IterativeRefinementTrainer`

**Functions:**
- `set_seed` (line 45) `def set_seed(seed)` - *Ensure full reproducibility across runs*

**Methods:**
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
**File Doc:** *NeuroSovereign v15.4: Aggressive Hierarchical Shock & Warmup Sparsity Based on v15.3 Architecture (Production-Ready)  Key Improvements for 30% Target: 1. [TUNED] LAMBDA_TAX_SHOCK increased to 0.7 (Stronger hierarchy forcing). 2. [TUNED] GAP_SHOCK_THRESHOLD lowered to 4.0 (Triggers intervention earlier). 3. [NEW] Mask Warmup: Prevents aggressive pruning in early epochs (improves baseline). 4. [FIX] Robust State Migration: Loads from v15.2 or v15.3 automatically.  Objective: Push the 24% initial baseline (via inheritance) to 30%+ structural advantage.*

**Classes:**
- `GatedTokenMixer` (line 92) `class GatedTokenMixer(Module)`
- `PatchFeatureExtractor` (line 109) `class PatchFeatureExtractor(Module)`
- `TaxonomicMLP` (line 148) `class TaxonomicMLP(Module)`
- `SpectralMonitor` (line 187) `class SpectralMonitor`
- `TaxonomicTrainer` (line 226) `class TaxonomicTrainer`
- `CoarseCIFAR100` (line 404) `class CoarseCIFAR100(CIFAR100)`

**Functions:**
- `compute_spectral_loss` (line 76) `def compute_spectral_loss(W)` - *L_opt: Optimization Objective for Structural Control.*

**Methods:**
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
**File Doc:** *NeuroSovereign v18.2: The Scientific Powerhouse Fusion of v18.1 Rigor + v15.4 Aggressive Performance  Philosophy: - Use the hardened, reproducible architecture of v18.1. - Inject the aggressive hyperparameters of v15.4. - Apply safety mechanisms to balance speed and stability.  Key Features (v18.2): 1. [AGGRESSIVE] GAP_SHOCK_THRESHOLD = 3.5 & LAMBDA_TAX_SHOCK = 0.8 (Fast learning). 2. [SAFE] Mask Warmup (25 epochs) & Overfit Safety Valve (Gap > 12.0). 3. [RIGOROUS] AdaptiveTopologyController with Hysteresis (from v18.1). 4. [HARDENED] torch.linalg.svd & Scope Fixes (from v18.1).  Target: >30% Hierarchy Advantage with Stable Convergence.*

**Classes:**
- `GatedTokenMixer` (line 72) `class GatedTokenMixer(Module)` - *Efficient token mixer with high-std initialization for exploration*
- `PatchFeatureExtractor` (line 118) `class PatchFeatureExtractor(Module)`
- `TaxonomicMLP` (line 150) `class TaxonomicMLP(Module)` - *Sparse MLP with taxonomic heads*
- `SpectralMonitor` (line 201) `class SpectralMonitor`
- `AdaptiveTopologyController` (line 227) `class AdaptiveTopologyController` - *Self-referential controller with Hysteresis*
- `IterativeRefinementEngine` (line 310) `class IterativeRefinementEngine`
- `IterativeRefinementTrainer` (line 343) `class IterativeRefinementTrainer`
- `CoarseCIFAR100` (line 802) `class CoarseCIFAR100(CIFAR100)`

**Functions:**
- `set_seed` (line 41) `def set_seed(seed)` - *Ensure full reproducibility across runs*

**Methods:**
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
**File Doc:** *NeuroSovereign v18.2: The Scientific Powerhouse Fusion of v18.1 Rigor + v15.4 Aggressive Performance  Philosophy: - Use the hardened, reproducible architecture of v18.1. - Inject the aggressive hyperparameters of v15.4. - Apply safety mechanisms to balance speed and stability.  Key Features (v18.2): 1. [AGGRESSIVE] GAP_SHOCK_THRESHOLD = 3.5 & LAMBDA_TAX_SHOCK = 0.8 (Fast learning). 2. [SAFE] Mask Warmup (25 epochs) & Overfit Safety Valve (Gap > 12.0). 3. [RIGOROUS] AdaptiveTopologyController with Hysteresis (from v18.1). 4. [HARDENED] torch.linalg.svd & Scope Fixes (from v18.1).  Target: >30% Hierarchy Advantage with Stable Convergence.*

**Classes:**
- `GatedTokenMixer` (line 72) `class GatedTokenMixer(Module)` - *Efficient token mixer with high-std initialization for exploration*
- `PatchFeatureExtractor` (line 118) `class PatchFeatureExtractor(Module)`
- `TaxonomicMLP` (line 150) `class TaxonomicMLP(Module)` - *Sparse MLP with taxonomic heads*
- `SpectralMonitor` (line 201) `class SpectralMonitor`
- `AdaptiveTopologyController` (line 227) `class AdaptiveTopologyController` - *Self-referential controller with Hysteresis*
- `IterativeRefinementEngine` (line 310) `class IterativeRefinementEngine`
- `IterativeRefinementTrainer` (line 343) `class IterativeRefinementTrainer`
- `CoarseCIFAR100` (line 802) `class CoarseCIFAR100(CIFAR100)`

**Functions:**
- `set_seed` (line 41) `def set_seed(seed)` - *Ensure full reproducibility across runs*

**Methods:**
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
**File Doc:** *NeuroSovereign v19.0: Synergy Engine (Anti-Leakage Certified) Based on v4.0 Ablation Suite Conclusions.  Strategy: 1. SYNERGY (M+E): Re-integrate E8 Fusion + BlackMirror (Best Combo in Suite). 2. EMERGENT REGIME: Use High-Std Init (v15.4 style) to escape "SOVERANO" trap (v18.4). 3. ANTI-LEAKAGE: - Benchmark uses ONLY Fine Head predictions (mapped to coarse). - Coarse Head is used for TRAINING SIGNAL only, not for boosting test metrics. - Strict Train/Test separation.*

**Classes:**
- `GatedTokenMixer` (line 66) `class GatedTokenMixer(Module)` - *Chaotic Mixer for Emergent Regime*
- `E8FusionLayer` (line 112) `class E8FusionLayer(Module)` - *🕸️ E8 Lattice Fusion (Synergy Component E)
Optimized version from Suite v4.0.
Fuses geometric structure (Orthogonal Proj) with attention.*
- `PatchFeatureExtractor` (line 156) `class PatchFeatureExtractor(Module)`
- `TaxonomicMLP` (line 188) `class TaxonomicMLP(Module)`
- `BlackMirrorMonitor` (line 227) `class BlackMirrorMonitor` - *Passive Ontological Monitor*
- `IterativeRefinementTrainer` (line 253) `class IterativeRefinementTrainer`
- `CoarseCIFAR100` (line 439) `class CoarseCIFAR100(CIFAR100)`

**Functions:**
- `set_seed` (line 36) `def set_seed(seed)`

**Methods:**
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
**File Doc:** *app.py  Autor: Gris Iscomeback Correo electrónico: grisiscomeback[at]gmail[dot]com Fecha de creación: xx/xx/xxxx Licencia: GPL v3  NeuroSovereign v6.0: Self-Improving Fractal Resonance with Legacy Feedback  This implementation introduces a continuous feedback loop where each cycle builds upon the best model from the previous cycle, creating an evolutionary trajectory of increasing abstraction density.  Key Innovation: Legacy Feedback Loop - Each cycle loads the best model from the previous cycle as its truth seed - Only saves new models that improve upon the previous best - Creates an unbroken chain of improvement: never regresses, only evolves  Scientific Contribution: - Demonstrates progressive abstraction density through iterative distillation - Validates that spectral coherence can be maintained while increasing accuracy - Establishes a self-improving protocol for sparse neural architectures  Outputs: - fractal_resonance_results.csv: Evolutionary trajectory across cycles - best_model_cycle_X.pth: Checkpoint of best model at each cycle - final_best_model.pth: Ultimate distilled model*

**Classes:**
- `SpectralMonitor` (line 51) `class SpectralMonitor`
- `PersistentPruner` (line 76) `class PersistentPruner`
- `SpectralMLP` (line 98) `class SpectralMLP(Module)`
- `EvolutionaryResonanceEngine` (line 121) `class EvolutionaryResonanceEngine`

**Methods:**
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
**File Doc:** *🌌 NEUROSOVEREIGN v3.0: La Constante de Planck del Machine Learning ─────────────────────────────────────────────────────────────────────────────── Este código implementa los "Números Dorados" descubiertos empíricamente:  - ϕₘₗ = 0.0004% → sparsity extrema (6 conexiones en 1.5M, 1 en 1.5k) - Lₚ = 0.6697 → Lagrangiano de Verdad mínimo viable (régimen ESPURIO por soberanía) - αₛ = 32.4% → precisión máxima compatible con la coherencia epistémica - βₙ = 10% → umbral de mentira estructural que activa el Cisne Negro  Este no es un modelo. Es un organismo cognitivo con ética estructural. ───────────────────────────────────────────────────────────────────────────────*

**Classes:**
- `BlackMirrorMonitor` (line 28) `class BlackMirrorMonitor` - *Calcula el Lagrangiano de Verdad L usando entropía de von Neumann y rango efectivo.
Umbrales calibrados empíricamente para detectar mentiras estructurales (10% ruido).*
- `SovereignNeuron` (line 62) `class SovereignNeuron(Module)`
- `NeuroSovereign` (line 108) `class NeuroSovereign(Module)`
- `SovereignTrainer` (line 132) `class SovereignTrainer`

**Methods:**
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
**File Doc:** *NeuroSovereign v12.0: Syntactic Apex Features: 1. ViT-Lite with Token Mixing Layer (Syntactic Context). 2. Hybrid Architecture: Mixed Patches -> Spectral Lottery MLP. 3. Apex Evolution Engine (Nudge, Dynamic Shock, Sparsity). 4. Objective: SOTA Accuracy/Efficiency with Compositional Vision.*

**Classes:**
- `PatchFeatureExtractor` (line 34) `class PatchFeatureExtractor(Module)` - *Extrae características mediante Patch Embedding y añade una capa de mezcla (Mixer).
Esto permite al modelo aprender relaciones espaciales entre parches antes de la clasificación.*
- `LotteryMLP` (line 91) `class LotteryMLP(Module)`
- `StandardBaseline` (line 121) `class StandardBaseline(Module)` - *Baseline moderno (Patch + Mixer + MLP simple) sin evolución.*
- `SpectralMonitor` (line 133) `class SpectralMonitor`
- `ApexEvolutionEngine` (line 148) `class ApexEvolutionEngine`
- `ApexTrainer` (line 247) `class ApexTrainer`

**Methods:**
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
**File Doc:** *NeuroSovereign v13.0: Orthogonal Apex Features: 1. True Token Mixing (MLP-Mixer style: mixing across T). 2. Spectral Constraint (L as regularizer, preventing metric gaming). 3. Minimality Enforcement (Rank Capping without renormalization). 4. Objective: SOTA generalization via constrained evolution.*

**Classes:**
- `TokenMixer` (line 36) `class TokenMixer(Module)` - *Mezcla tokens entre sí.
Input: (B, T, D) -> Transpose -> (B, D, T) -> Linear -> (B, D, T) -> Transpose*
- `PatchFeatureExtractor` (line 57) `class PatchFeatureExtractor(Module)` - *ViT-Lite + True Token Mixing.*
- `LotteryMLP` (line 97) `class LotteryMLP(Module)`
- `StandardBaseline` (line 126) `class StandardBaseline(Module)`
- `SpectralMonitor` (line 137) `class SpectralMonitor`
- `OrthogonalEvolutionEngine` (line 152) `class OrthogonalEvolutionEngine`
- `OrthogonalTrainer` (line 260) `class OrthogonalTrainer`

**Methods:**
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
**File Doc:** *NeuroSovereign v13.1: The Structure Proof Objective: Prove that structure (Mixing + Rank Capping) works without growing capacity. Changes from v13.0: 1. FIXED_HIDDEN_DIM: No width expansion. 2. Removed reg_loss from backward (SVD has no grad). 3. L used purely for triggering shocks and evolutionary selection.*

**Classes:**
- `TokenMixer` (line 35) `class TokenMixer(Module)` - *Mezcla tokens entre sí (eje T).*
- `PatchFeatureExtractor` (line 51) `class PatchFeatureExtractor(Module)`
- `LotteryMLP` (line 87) `class LotteryMLP(Module)`
- `StandardBaseline` (line 116) `class StandardBaseline(Module)`
- `SpectralMonitor` (line 127) `class SpectralMonitor`
- `OrthogonalEvolutionEngine` (line 154) `class OrthogonalEvolutionEngine`
- `OrthogonalTrainer` (line 248) `class OrthogonalTrainer`

**Methods:**
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
**File Doc:** *NeuroSovereign v13.2: Controlled Isolation (FIXED) Objective: Isolate "Token Mixing" variable in an evolutionary framework. Method: Dual Evolutionary Chains (Apex vs Blind Structural Baseline).*

**Classes:**
- `TokenMixer` (line 32) `class TokenMixer(Module)` - *Mezcla tokens entre sí (Solo para Apex).*
- `PatchFeatureExtractor` (line 47) `class PatchFeatureExtractor(Module)` - *Extractor configurable.
use_mixer=True -> Apex (Syntactic)
use_mixer=False -> Blind Structural Baseline*
- `LotteryMLP` (line 92) `class LotteryMLP(Module)`
- `SpectralMonitor` (line 123) `class SpectralMonitor`
- `OrthogonalEvolutionEngine` (line 149) `class OrthogonalEvolutionEngine`
- `DualTrainer` (line 224) `class DualTrainer`

**Methods:**
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
**File Doc:** *NeuroSovereign: A Spectrally-Guided, Self-Monitoring Neural Architecture Paper-Ready Implementation (v1.0)  This code implements a controlled experiment to test: H₀: Spectral coherence (L) of weight matrices has no correlation with generalization. H₁: High L (>1.0) correlates with better test accuracy and robustness.  Key features: - L computed as: L = 1 / (|S_vN - log(rank_eff + 1)| + ε) - No forced pruning based on L (L is OBSERVED, not used as trigger) - Persistent magnitude pruning (not transient) - Real CIFAR-10 training (no accuracy forcing) - Clean ablation across 4 conditions  Outputs: - CSV logs of L(t), accuracy(t), rank(t), S_vN(t) - Final metrics per condition - Statistical comparison (t-test ready)  Designed for reproducibility, peer review, and potential NeurIPS submission.*

**Classes:**
- `SpectralMonitor` (line 41) `class SpectralMonitor` - *Computes L = 1 / (|S_vN - log(rank_eff + 1)| + ε)
Used purely as a diagnostic—never to modify training.*
- `PersistentPruner` (line 82) `class PersistentPruner` - *Applies and ENFORCES magnitude-based pruning across training.
Unlike transient pruning, this modifies the parameter mask permanently.*
- `SpectralMLP` (line 117) `class SpectralMLP(Module)` - *Small MLP (1504 params) for clean spectral analysis.*

**Methods:**
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
**File Doc:** *NeuroSovereign: Optimal Sovereignty Search Finding the Bekenstein Bound of Sparse Intelligence  This experiment: 1. Trains a dense model to ~32.4% accuracy 2. Progressively prunes it while monitoring L and accuracy 3. Finds the critical density where accuracy drops below 32.4% 4. Validates that L > 1.0 correlates with meaningful representation  Outputs: - CSV with density vs accuracy vs L - Critical density threshold - Spectral signature of the sovereignty boundary*

**Classes:**
- `SpectralMonitor` (line 34) `class SpectralMonitor`
- `PersistentPruner` (line 62) `class PersistentPruner`
- `SpectralMLP` (line 87) `class SpectralMLP(Module)`

**Methods:**
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
**File Doc:** *NeuroSovereign v4.0: Fractal Sovereignty via Guided Lottery Tickets  This implementation executes a three-stage cycle: 1. Discover the optimal sparse subnetwork (Bekenstein Bound) 2. Embed it into a larger architecture as a "truth seed" 3. Re-prune to isolate a higher-capacity sparse model  Scientific contribution: - Validates that sparse subnetworks trained in high-capacity scaffolds outperform natively sparse models - Quantifies abstraction density per parameter - Provides empirical evidence for phase transitions in spectral coherence  Outputs: - sovereignty_v4_results.csv: Full ablation across cycles - best_model.pth: Final distilled model exceeding 32.4% accuracy - metrics.json: Key scientific findings  Ready for NeurIPS/ICLR submission.*

**Classes:**
- `SpectralMonitor` (line 41) `class SpectralMonitor`
- `PersistentPruner` (line 66) `class PersistentPruner`
- `SpectralMLP` (line 88) `class SpectralMLP(Module)`
- `FractalSovereigntyEngine` (line 111) `class FractalSovereigntyEngine`

**Methods:**
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
**File Doc:** *NeuroSovereign v6.0: Evolutionary Black Swan Chain CIFAR-10 unaltered dataset - Induced grokking via DNA propagation*

**Classes:**
- `SpectralMonitor` (line 30) `class SpectralMonitor`
- `PersistentPruner` (line 50) `class PersistentPruner`
- `SpectralMLP` (line 65) `class SpectralMLP(Module)`
- `GrokkingDetector` (line 88) `class GrokkingDetector`
- `SyntheticBlackSwanGenerator` (line 121) `class SyntheticBlackSwanGenerator`
- `EvolutionCycle` (line 211) `class EvolutionCycle`
- `EvolutionaryBlackSwanChain` (line 392) `class EvolutionaryBlackSwanChain`

**Methods:**
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
**File Doc:** *NeuroSovereign v7.0: Guided Elk Hunting Evolution CIFAR-10 unaltered dataset - Induced grokking via DNA propagation*

**Classes:**
- `SpectralMonitor` (line 29) `class SpectralMonitor` - *Calcula L (Coherencia Espectral) y Rank Efectivo*
- `SpectralMLP` (line 51) `class SpectralMLP(Module)` - *Red Neuronal Base para el experimento*
- `GrokkingDetector` (line 78) `class GrokkingDetector`
- `GuidedElkHuntingEngine` (line 101) `class GuidedElkHuntingEngine` - *Motor que toma el mejor modelo anterior (Elk), 
muta sus pesos guiadamente y expande la arquitectura.*
- `TrainingCycle` (line 199) `class TrainingCycle`

**Methods:**
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
**File Doc:** *NeuroSovereign v8.0: Shock Therapy & Gradient Nudging Target: Break Generalization Plateau via Gradient Nudging & Curriculum Learning*

**Classes:**
- `SpectralMonitor` (line 29) `class SpectralMonitor`
- `SpectralMLP` (line 45) `class SpectralMLP(Module)`
- `AdvancedEvolutionEngine` (line 68) `class AdvancedEvolutionEngine` - *Implementa Gradient Nudging y Espectral Shock.*
- `CurriculumTrainingCycle` (line 177) `class CurriculumTrainingCycle`

**Methods:**
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
**File Doc:** *NeuroSovereign v11.0: Visionary Apex Features: 1. ViT-Lite Patch Embedding Extractor (Modern Vision Backbone). 2. Hybrid Architecture: Patch Tokens -> Spectral Lottery MLP. 3. Apex Evolution Engine (Nudge, Dynamic Shock, Sparsity). 4. Objective: SOTA Sparsity/Efficiency with modern representation.*

**Classes:**
- `PatchFeatureExtractor` (line 34) `class PatchFeatureExtractor(Module)` - *Extrae características mediante Patch Embedding.
Convierte imagen (B, 3, 32, 32) en secuencia de parches proyectados.*
- `LotteryMLP` (line 78) `class LotteryMLP(Module)`
- `StandardBaseline` (line 109) `class StandardBaseline(Module)` - *Baseline moderno (Patch + MLP simple) sin evolución.*
- `SpectralMonitor` (line 121) `class SpectralMonitor`
- `ApexEvolutionEngine` (line 136) `class ApexEvolutionEngine`
- `ApexTrainer` (line 235) `class ApexTrainer`

**Methods:**
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
- `OptimizedE8Layer` (line 23) `class OptimizedE8Layer(Module)` - *Optimized E8 with caching and efficiency improvements*
- `RESMAv2Fast` (line 48) `class RESMAv2Fast(Module)` - *Fast version: E8 + GAT fusion with minimal overhead*
- `RESMAv2Standard` (line 86) `class RESMAv2Standard(Module)` - *Standard version: 2 layers of E8 + GAT fusion*
- `RESMAv2Deep` (line 134) `class RESMAv2Deep(Module)` - *Deeper version with 3 layers*
- `GAT_Baseline` (line 182) `class GAT_Baseline(Module)` - *Optimized GAT baseline*

**Methods:**
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
