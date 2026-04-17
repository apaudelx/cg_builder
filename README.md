# CG Molecule Builder

This project generates coarse-grained molecules, builds datasets, and analyzes molecular diversity.
Use `topo_generator.py` and `generator.py` for one molecule, then use the batch and analysis scripts for dataset-scale workflows.

## 1) Environment Setup

Install dependencies:

```bash
python3 -m pip install numpy networkx matplotlib plotly scikit-learn umap-learn
```

Key points:
- `scikit-learn` and `umap-learn` are required for `analyze_molecule_space.py`.
- If you only run single-molecule generation, you can work with `numpy networkx matplotlib plotly`.

## 2) Data Files Used by Generators

Default generator constraints are read from `data/`:
- `allowed_bonds.txt`: permitted bead-type bond pairs.
- `max_degree.txt`: maximum allowed connectivity per bead type.
- `bond_stats.txt`: mean and standard deviation of bond lengths by bead pair.

Key points:
- If a bead-pair length is missing, scripts fall back to `0.27 nm` and clip to internal min/max bounds.
- Most scripts are hardcoded to `data/...`, so update files there or edit script paths when using another library directory.

## 3) Single Molecule Workflow

### Step 3.1 Generate Topology (`generated.itp`)

Basic command:

```bash
python3 topo_generator.py --nbeads 12 --seed 1
```

What the main options do:
- `--nbeads INT`: total number of beads to generate (required).
- `--n3 --n4 --n5 --n6`: number of 3/4/5/6-member rings to enforce.
- `--bead-weights SC4:3,TC4:2`: force exact counts of specific bead types when adding beads.
- `--seed INT`: reproducible topology sampling.
- `--max-tries INT`: retry budget for constrained ring construction.
- `--output PATH`: output topology file (default `generated.itp`).

Key points:
- Ring bead requirements must fit inside `--nbeads`, otherwise generation stops with an error.
- If no rings are requested, the generator builds tree-like/branched topologies under bond and degree constraints.

### Step 3.2 Generate Coordinates + Visualizations

Command:

```bash
python3 generator.py
```

What it does:
- Reads `generated.itp`.
- Writes force-ready topology `res.itp` with bond equilibrium lengths.
- Builds 3D coordinates and writes `res.gro`.
- Writes static 2D projection `molecule.png` and interactive 3D `molecule_3d.html`.

Key points:
- This script currently expects `generated.itp` as input and writes fixed output names.
- Nonbonded overlaps are reduced during placement using an internal minimum distance threshold.

## 4) Batch Dataset Generation

Use this to produce many molecules with reproducible seeds and a manifest.

Example:

```bash
python3 batch_generate_molecules.py \
	--count 100 \
	--min-beads 6 \
	--max-beads 15 \
	--master-seed 42 \
	--prefix bigm \
	--outdir dataset_100 \
	--bead-allocation balanced
```

Useful options:
- `--count`: number of molecules.
- `--min-beads`, `--max-beads`: bead-count range.
- `--master-seed`: required, defines deterministic per-molecule seeds.
- `--prefix`: naming prefix (`bigm_001`, ...).
- `--bead-allocation random|balanced`: random sampling vs near-equal allocation by bead count.
- `--compare-dir PATH`: optional reference dataset; flags topology signature overlap.
- `--retry-count`: retries when generation fails for a molecule.

Outputs in `--outdir`:
- `<prefix>_XXX/<prefix>_XXX.itp`
- `<prefix>_XXX/<prefix>_XXX.gro`
- `manifest.csv`
- `overlap_report.csv` (only if `--compare-dir` is used)

Key points:
- `manifest.csv` is the source of truth for success/failure status, seeds, hashes, and errors.
- The script raises an error at the end if any molecule failed, even if many succeeded.

## 5) Diverse Pool Generation (Active Learning Style)

Use this when you want a final subset that is structurally diverse, not just randomly generated.

Example:

```bash
python3 generate_diverse_pool.py \
	--target 100 \
	--oversample 3.0 \
	--master-seed 42 \
	--outdir generated_pool \
	--compare-dir dataset_100
```

What the key options do:
- `--target`: final pool size after filtering.
- `--oversample`: raw generation multiplier before diversity selection.
- `--compare-dir PATH` (repeatable): avoid overlaps with one or more existing datasets.
- `--skip-diversity`: keep generated molecules without max-min selection.

Outputs in `--outdir`:
- `pool/`: final renumbered molecules (`mol_00001`, ...).
- `manifest.csv`: mapping from final names to originals and metadata.
- `generation_config.json`: reproducibility settings.

Key points:
- The script generates a temporary `_raw` pool, selects a diverse subset, then deletes `_raw`.
- Generation is stratified across tree/ring regimes to improve coverage before diversity filtering.

## 6) Rebuild or Expand the Constraint Library

Use this to mine existing `.itp` datasets and regenerate `allowed_bonds`, `max_degree`, and `bond_stats`.

Dry-run example:

```bash
python3 rebuild_library.py \
	--datasets ee_itp_667 dataset_100 \
	--existing-dir data \
	--outdir data_expanded \
	--dry-run
```

Write files example:

```bash
python3 rebuild_library.py \
	--datasets ee_itp_667 dataset_100 \
	--existing-dir data \
	--outdir data_expanded
```

Key points:
- Without `--existing-dir`, the library is built from dataset evidence only.
- With `--existing-dir`, mined statistics are merged with existing files and written to `--outdir`.

## 7) Molecule Space Analysis

This script computes feature-space uniqueness, UMAP embedding, and a greedy diverse subset.

Example:

```bash
python3 analyze_molecule_space.py \
	--datasets dataset_100 \
	--bead-params NBFIX_table \
	--outdir molecule_space_dataset100 \
	--k-neighbors 5 \
	--diverse-k 20 \
	--seed 42
```

Main options:
- `--datasets DIR [DIR ...]`: one or more dataset roots to scan for `.itp` (and optional `.gro`).
- `--bead-params PATH`: optional CSV (`bead_type,epsilon,sigma`) or NBFIX-style table.
- `--k-neighbors`: neighborhood size for uniqueness score.
- `--umap-neighbors`, `--umap-min-dist`: UMAP shape controls.
- `--diverse-k`: final diverse subset size.

Outputs in `--outdir`:
- `uniqueness_scores.csv`
- `embedding_umap.csv`
- `plot_umap_uniqueness.png`
- `diverse_subset_k<k>.csv`

Key points:
- Uniqueness ranks individually isolated molecules, while diverse subset selection avoids redundant picks.
- Use `diverse_subset_k<k>.csv` as the default shortlist for follow-up runs.

## 8) Practical End-to-End Commands

Minimal single molecule run:

```bash
python3 topo_generator.py --nbeads 12 --seed 1
python3 generator.py
```

Typical dataset + analysis run:

```bash
python3 batch_generate_molecules.py --count 100 --min-beads 6 --max-beads 15 --master-seed 42 --prefix bigm --outdir dataset_100 --bead-allocation balanced
python3 analyze_molecule_space.py --datasets dataset_100 --bead-params NBFIX_table --outdir molecule_space_dataset100 --seed 42 --diverse-k 20
```

Key points:
- Keep `--master-seed` and analysis `--seed` recorded for reproducible outputs.
- Keep generated manifests and analysis CSVs; they are the quickest audit trail for what was produced and selected.
