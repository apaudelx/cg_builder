# ANALYSIS

## Scope
This document summarizes the dataset generation and molecule-space analysis workflow used in this project, with a focus on the dataset in dataset_100.

## 1. How Data Was Generated
The dataset was generated programmatically using a batch pipeline that builds valid coarse-grained topologies and then generates 3D coordinates.

### Generation workflow
1. Topology generation (constraint-aware):
- Uses allowed bond rules from data/allowed_bonds.txt
- Uses degree constraints from data/max_degree.txt
- Produces per-molecule .itp topologies

2. Coordinate generation:
- Uses bond statistics from data/bond_stats.txt
- Produces per-molecule .gro coordinates

3. Batch dataset creation:
- 100 molecules total
- Bead count sampled uniformly in [6, 15]
- One reproducible master seed with deterministic per-molecule seeds
- Molecule folders and file names use prefix bigm_XXX

### Source of truth for generation metadata
See:
- dataset_100/notes/README.md
- dataset_100/notes/manifest.csv

## 2. What Data Was Generated
Main generated dataset:
- dataset_100/

Structure:
1. 100 molecule folders
- bigm_001 to bigm_100
- each folder contains:
  - bigm_XXX.itp
  - bigm_XXX.gro

2. Notes and run metadata
- dataset_100/notes/README.md
- dataset_100/notes/manifest.csv

## 3. Analysis Objective
Goal: identify molecules that are dissimilar from each other, not just individually unusual.

This was done by:
1. Building a fixed-length feature vector per molecule (aggregated node/edge/global descriptors)
2. Scaling features
3. Computing k-nearest-neighbor based uniqueness scores in high-dimensional feature space
4. Running UMAP for 2D visualization
5. Selecting a distance-diverse subset with greedy max-min distance

## 4. Analysis Inputs
Dataset input:
- dataset_100

Optional bead-parameter enrichment:
- NBFIX_table (used to include epsilon/sigma-derived features)

## 5. Where to Find Analysis Outputs
Current minimal analysis outputs are in:
- molecule_space_dataset100/

Files:
1. uniqueness_scores.csv
- Per-molecule uniqueness score (higher means more isolated in high-dimensional feature space)

2. embedding_umap.csv
- 2D UMAP coordinates for each molecule plus uniqueness score

3. plot_umap_uniqueness.png
- UMAP visualization colored by uniqueness

4. diverse_subset_k20.csv
- Distance-diverse shortlist (20 molecules) selected to be mutually dissimilar

## 6. How to Interpret the Outputs
Use this order:

1. Primary selection file: diverse_subset_k20.csv
- This is the recommended list if you need a non-redundant, structurally diverse subset.
- It avoids picking many near-duplicates.

2. Secondary ranking file: uniqueness_scores.csv
- Good for individual novelty ranking.
- Not sufficient alone for batch selection because high-uniqueness molecules can still be similar to each other.

3. Visualization support: plot_umap_uniqueness.png
- Helpful to visually inspect neighborhoods and outliers.
- Use as a qualitative map, not as the only decision criterion.

4. Coordinate table: embedding_umap.csv
- Useful for downstream plotting, annotation, and reproducibility of the visualization.

## 7. Practical Recommendation
For labeling or experimental follow-up, start with diverse_subset_k20.csv.
If budget allows more molecules, generate additional diverse subsets (for example k=30 or k=40) using the same workflow.

## 8. Reproducibility Command (Dataset 100 Analysis)
Use the common virtual environment and run:

source /Users/apaudel/common_venv/bin/activate
python analyze_molecule_space.py --datasets dataset_100 --bead-params NBFIX_table --outdir molecule_space_dataset100 --seed 42 --diverse-k 20

This reproduces the same minimal analysis outputs described above.
