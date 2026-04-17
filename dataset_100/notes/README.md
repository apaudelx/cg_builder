# Dataset README

## Overview
This folder contains a generated dataset of 100 coarse-grained molecules.
Each molecule has its own subfolder with one topology file (.itp) and one coordinate file (.gro).

## Generation Settings
- Molecule count: 100
- Bead range: 6 to 15 (uniform random sampling)
- Prefix: bigm
- Master seed: 42
- Base seed per molecule: master_seed + molecule_index
- Retry policy: deterministic retry with seed offsets (+1000 per attempt), maximum 5 retries

## Folder Structure
- One subfolder per molecule: bigm_001 through bigm_100
- Inside each subfolder:
  - bigm_XXX.itp
  - bigm_XXX.gro
- Dataset-level metadata file:
  - manifest.csv

## Manifest Schema
manifest.csv contains one row per molecule with these columns:
- molecule: molecule identifier (for example, bigm_001)
- nbeads: number of beads in that molecule
- base_seed: seed derived from master seed and index
- used_seed: final seed used (may differ from base_seed if retries were needed)
- status: generation result (ok or failed)
- error: error message if generation failed

## Reproducibility
This dataset is reproducible by re-running the batch generator with the same arguments:
- count 100
- min-beads 6
- max-beads 15
- master-seed 42
- prefix bigm

If used_seed equals base_seed, generation succeeded on the first try.
If used_seed differs from base_seed, generation used a deterministic retry seed and is still reproducible.

## Run Summary (Current Dataset)
- Successful molecules: 100
- Failed molecules: 0
- Molecules requiring retry seeds: 8

Retry cases:
- bigm_005 (base 47, used 2047)
- bigm_014 (base 56, used 1056)
- bigm_023 (base 65, used 1065)
- bigm_036 (base 78, used 1078)
- bigm_039 (base 81, used 1081)
- bigm_049 (base 91, used 1091)
- bigm_050 (base 92, used 1092)
- bigm_055 (base 97, used 1097)

## Bead Count Distribution
Observed distribution in this dataset:
- 6 beads: 9 molecules
- 7 beads: 14 molecules
- 8 beads: 8 molecules
- 9 beads: 16 molecules
- 10 beads: 11 molecules
- 11 beads: 11 molecules
- 12 beads: 9 molecules
- 13 beads: 6 molecules
- 14 beads: 8 molecules
- 15 beads: 8 molecules

## Notes
- The dataset uses random sampling and constraint-based graph construction.
- Small deviations from perfect uniformity are expected for 100 samples.
- For strictly balanced bins, use deterministic bead allocation per bead count instead of random sampling.
