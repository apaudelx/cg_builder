# Molecule Generator Usage

This repository contains two scripts:

- topo_generator.py → generates molecular topology (.itp)
- generator.py → generates coordinates (.gro) and visualization

---

## Requirements

Install dependencies:

pip install numpy networkx matplotlib plotly

---

## Step 1: Generate Topology

Run:

python topo_generator.py --nbeads 10

Optional arguments:

--n3 1   # 3-member ring  
--n4 1   # 4-member ring   

--bead-weights SC4:3,TC4:2  

--seed 42  

Output:

generated.itp

---

## Step 2: Generate Coordinates and Visualization

Run:

python generator.py

This will:

- read generated.itp  
- generate 3D coordinates  
- apply bond length statistics  
- avoid nonbonded overlaps  
- create visualization  

---

## Outputs

res.itp  
res.gro  
molecule.png  
molecule_3d.html  


## Typical Workflow

python topo_generator.py --nbeads 12 --seed 1  
python generator.py  

---

## Notes

- if no rings are specified, a linear/branched molecule is generated  
- bond lengths are sampled from bond_stats.txt  
- default bond length fallback = 0.27 nm  
- nonbonded clashes are avoided during coordinate placement  
