"""Generate a large, diverse pool of CG molecules for active learning.

Stratified generation → duplicate removal → diversity-based selection.

Strategy:
  1. Generate molecules in 4 strata (small/medium/large trees + ring molecules)
  2. Remove exact-signature duplicates and overlaps with existing datasets
  3. Compute feature vectors for all molecules
  4. Use greedy max-min selection to pick the most diverse final subset

Usage:
    # Generate 1000 diverse molecules from a raw pool of 3000
    python cg_builder/generate_diverse_pool.py --target 1000 --oversample 3.0

    # Generate 500 diverse molecules, checking against existing 667 base set
    python cg_builder/generate_diverse_pool.py --target 500 \
        --compare-dir data/ee_itp_667 \
        --compare-dir ee_itp_100_new

    # Quick test run
    python cg_builder/generate_diverse_pool.py --target 50 --oversample 2.0
"""

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import topo_generator as topo
from batch_generate_molecules import (
    build_coordinates,
    graph_signature,
    load_signature_hashes,
    read_bond_stats,
    write_gro,
    write_itp,
)

DATA_DIR = SCRIPT_DIR / "data"


# ═══════════════════════════════════════════════════════════════════════════════
# Stratum Definitions
# ═══════════════════════════════════════════════════════════════════════════════

STRATA = [
    {"name": "small_tree",  "min_beads": 2,  "max_beads": 5,  "rings": {}, "weight": 0.20},
    {"name": "medium_tree", "min_beads": 6,  "max_beads": 10, "rings": {}, "weight": 0.30},
    {"name": "large_tree",  "min_beads": 11, "max_beads": 15, "rings": {}, "weight": 0.20},
    {"name": "ring_small",  "min_beads": 5,  "max_beads": 8,  "rings": {"n3": 1}, "weight": 0.10},
    {"name": "ring_medium", "min_beads": 8,  "max_beads": 12, "rings": {"n4": 1}, "weight": 0.10},
    {"name": "ring_large",  "min_beads": 10, "max_beads": 15, "rings": {"n5": 1}, "weight": 0.05},
    {"name": "ring_6",      "min_beads": 10, "max_beads": 15, "rings": {"n6": 1}, "weight": 0.05},
]


def build_topology_with_rings(nbeads, seed, allowed_bonds, bead_pool, max_degree, ring_spec):
    """Build topology supporting optional ring structures."""
    random.seed(seed)

    n3 = ring_spec.get("n3", 0)
    n4 = ring_spec.get("n4", 0)
    n5 = ring_spec.get("n5", 0)
    n6 = ring_spec.get("n6", 0)

    ring_beads_needed = n3 * 3 + n4 * 4 + n5 * 5 + n6 * 6
    if ring_beads_needed > nbeads:
        raise ValueError(f"Ring beads ({ring_beads_needed}) exceed target ({nbeads})")

    beads, bonds, degree, bonds_set, rings = topo.build_rings(
        n3=n3, n4=n4, n5=n5, n6=n6,
        allowed_bonds=allowed_bonds,
        bead_pool=bead_pool,
        max_degree=max_degree,
        max_tries=10000,
    )

    if len(rings) > 1:
        topo.connect_rings(rings, beads, bonds, degree, bonds_set,
                           allowed_bonds, max_degree, set(bead_pool))

    if len(beads) < nbeads:
        topo.add_extra_beads(
            target_N=nbeads,
            beads=beads, bonds=bonds, degree=degree, bonds_set=bonds_set,
            allowed_bonds=allowed_bonds, bead_pool=bead_pool,
            max_degree=max_degree, original_beads=set(bead_pool),
            forced_beads=[],
        )

    return beads, bonds


# ═══════════════════════════════════════════════════════════════════════════════
# Feature Extraction (lightweight, no sklearn needed for generation)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_feature_vector(beads, bonds):
    """Compute a numeric feature vector for diversity comparison."""
    n = len(beads)
    m = len(bonds)

    type_counts = Counter(beads.values())
    all_types = sorted(type_counts.keys())

    degree = defaultdict(int)
    for i, j in bonds:
        degree[i] += 1
        degree[j] += 1
    degree_vals = [degree.get(i, 0) for i in sorted(beads.keys())]

    bond_types = Counter()
    for i, j in bonds:
        pair = tuple(sorted((beads[i], beads[j])))
        bond_types[pair] += 1

    edges = [(min(i, j), max(i, j)) for i, j in bonds]
    adj = defaultdict(set)
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    visited = set()
    components = 0
    for node in beads:
        if node in visited:
            continue
        components += 1
        stack = [node]
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            stack.extend(adj[cur] - visited)
    cyclomatic = max(0, m - n + components)

    features = {
        "num_atoms": n,
        "num_bonds": m,
        "avg_degree": np.mean(degree_vals) if degree_vals else 0,
        "max_degree": max(degree_vals) if degree_vals else 0,
        "graph_density": (2.0 * m) / (n * (n - 1)) if n > 1 else 0,
        "unique_bead_types": len(all_types),
        "cyclomatic_number": cyclomatic,
    }

    return features


# ═══════════════════════════════════════════════════════════════════════════════
# Generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_pool(total_count, master_seed, allowed_bonds, bead_pool, max_degree,
                  bond_stats, outdir, compare_hashes, retry_count=5):
    """Generate molecules across all strata."""
    rng = random.Random(master_seed)

    schedule = []
    for stratum in STRATA:
        count = max(1, int(total_count * stratum["weight"]))
        for _ in range(count):
            nbeads = rng.randint(stratum["min_beads"], stratum["max_beads"])
            schedule.append({
                "stratum": stratum["name"],
                "nbeads": nbeads,
                "rings": stratum["rings"],
            })

    rng.shuffle(schedule)
    # Trim or pad to exact count
    if len(schedule) > total_count:
        schedule = schedule[:total_count]
    while len(schedule) < total_count:
        s = rng.choice(STRATA)
        schedule.append({
            "stratum": s["name"],
            "nbeads": rng.randint(s["min_beads"], s["max_beads"]),
            "rings": s["rings"],
        })

    outdir.mkdir(parents=True, exist_ok=True)
    seen_hashes = set(compare_hashes)

    molecules = []
    failed = 0

    for idx, spec in enumerate(schedule, 1):
        mol_name = f"mol_{idx:05d}"
        mol_dir = outdir / mol_name
        mol_dir.mkdir(parents=True, exist_ok=True)

        base_seed = master_seed + idx
        success = False

        for attempt in range(retry_count + 1):
            candidate_seed = base_seed + attempt * 1000
            try:
                beads, bonds = build_topology_with_rings(
                    nbeads=spec["nbeads"],
                    seed=candidate_seed,
                    allowed_bonds=allowed_bonds,
                    bead_pool=bead_pool,
                    max_degree=max_degree,
                    ring_spec=spec["rings"],
                )

                _, sig_hash = graph_signature(beads, bonds)
                if sig_hash in seen_hashes:
                    continue
                seen_hashes.add(sig_hash)

                coords = build_coordinates(beads, bonds, candidate_seed, bond_stats)

                itp_path = mol_dir / f"{mol_name}.itp"
                gro_path = mol_dir / f"{mol_name}.gro"
                write_itp(beads, bonds, bond_stats, str(itp_path))
                write_gro(beads, coords, str(gro_path))

                features = compute_feature_vector(beads, bonds)

                molecules.append({
                    "name": mol_name,
                    "stratum": spec["stratum"],
                    "nbeads": spec["nbeads"],
                    "seed": candidate_seed,
                    "sig_hash": sig_hash,
                    "features": features,
                    "dir": mol_dir,
                    "has_ring": bool(spec["rings"]),
                })
                success = True
                break

            except Exception:
                continue

        if not success:
            shutil.rmtree(mol_dir, ignore_errors=True)
            failed += 1

        if idx % 500 == 0 or idx == len(schedule):
            print(f"  Generated {idx}/{len(schedule)} "
                  f"({len(molecules)} unique, {failed} failed)")

    return molecules


# ═══════════════════════════════════════════════════════════════════════════════
# Diversity Selection (greedy max-min)
# ═══════════════════════════════════════════════════════════════════════════════

def build_feature_matrix(molecules):
    """Build a normalized feature matrix from all molecules."""
    all_bead_types = set()
    all_bond_types = set()
    for mol in molecules:
        beads_path = mol["dir"] / f"{mol['name']}.itp"
        beads, bonds = _parse_itp_quick(beads_path)
        all_bead_types.update(beads.values())
        for i, j in bonds:
            pair = tuple(sorted((beads[i], beads[j])))
            all_bond_types.add(pair)

    bead_vocab = sorted(all_bead_types)
    bond_vocab = sorted(all_bond_types)

    rows = []
    for mol in molecules:
        beads_path = mol["dir"] / f"{mol['name']}.itp"
        beads, bonds = _parse_itp_quick(beads_path)

        f = mol["features"]
        row = [
            f["num_atoms"], f["num_bonds"], f["avg_degree"],
            f["max_degree"], f["graph_density"], f["unique_bead_types"],
            f["cyclomatic_number"],
        ]

        type_counts = Counter(beads.values())
        n = max(1, len(beads))
        for bt in bead_vocab:
            row.append(type_counts.get(bt, 0) / n)

        bond_counts = Counter()
        for i, j in bonds:
            pair = tuple(sorted((beads[i], beads[j])))
            bond_counts[pair] += 1
        m = max(1, len(bonds))
        for bp in bond_vocab:
            row.append(bond_counts.get(bp, 0) / m)

        rows.append(row)

    X = np.array(rows, dtype=float)

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    X_scaled = (X - mean) / std

    return X_scaled


def _parse_itp_quick(itp_path):
    """Minimal .itp parser for bead types and bonds."""
    beads = {}
    bonds = []
    section = None
    with open(itp_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            if line.startswith("["):
                section = line.lower()
                continue
            if ";" in line:
                line = line.split(";", 1)[0].strip()
                if not line:
                    continue
            parts = line.split()
            if section == "[ atoms ]" and len(parts) >= 2:
                beads[int(parts[0])] = parts[1]
            elif section == "[ bonds ]" and len(parts) >= 2:
                bonds.append((int(parts[0]), int(parts[1])))
    return beads, bonds


def greedy_maxmin_select(X_scaled, target_k):
    """Select the most diverse subset using greedy max-min distance."""
    n = len(X_scaled)
    k = min(target_k, n)

    print(f"  Running greedy max-min selection ({n} → {k})...")

    # Start from the molecule closest to the centroid (most "average")
    centroid = X_scaled.mean(axis=0)
    dists_to_centroid = np.linalg.norm(X_scaled - centroid, axis=1)
    start_idx = int(np.argmin(dists_to_centroid))

    selected = [start_idx]
    min_d = np.full(n, np.inf)

    # Update distances from start point
    dists = np.linalg.norm(X_scaled - X_scaled[start_idx], axis=1)
    min_d = np.minimum(min_d, dists)

    for step in range(1, k):
        min_d_copy = min_d.copy()
        for s in selected:
            min_d_copy[s] = -1

        idx = int(np.argmax(min_d_copy))
        selected.append(idx)

        dists = np.linalg.norm(X_scaled - X_scaled[idx], axis=1)
        min_d = np.minimum(min_d, dists)

        if (step + 1) % 500 == 0:
            print(f"    Selected {step + 1}/{k}")

    return selected


# ═══════════════════════════════════════════════════════════════════════════════
# Summary Stats
# ═══════════════════════════════════════════════════════════════════════════════

def print_pool_stats(molecules, label="Pool"):
    """Print distribution statistics for a molecule set."""
    strata_counts = Counter(m["stratum"] for m in molecules)
    bead_counts = [m["nbeads"] for m in molecules]
    ring_count = sum(1 for m in molecules if m["has_ring"])

    print(f"\n  {label} Statistics ({len(molecules)} molecules):")
    print(f"    Bead count range: {min(bead_counts)}–{max(bead_counts)} "
          f"(mean {np.mean(bead_counts):.1f})")
    print(f"    With rings: {ring_count} ({100*ring_count/len(molecules):.1f}%)")
    print(f"    Strata distribution:")
    for s, c in sorted(strata_counts.items()):
        print(f"      {s}: {c}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate a large, diverse pool of CG molecules for active learning."
    )
    parser.add_argument("--target", type=int, required=True,
                        help="Number of diverse molecules in the final pool")
    parser.add_argument("--oversample", type=float, default=3.0,
                        help="Oversample factor: generate target*oversample, then select "
                             "the most diverse (default: 3.0)")
    parser.add_argument("--master-seed", type=int, default=42)
    parser.add_argument("--outdir", default="generated_pool",
                        help="Output directory for the diverse pool")
    parser.add_argument("--compare-dir", action="append", default=[],
                        help="Existing dataset directory to avoid overlaps "
                             "(can be specified multiple times)")
    parser.add_argument("--retry-count", type=int, default=5)
    parser.add_argument("--skip-diversity", action="store_true",
                        help="Skip diversity selection, keep all generated molecules")

    args = parser.parse_args()

    raw_count = int(args.target * args.oversample)
    print("=" * 60)
    print("DIVERSE MOLECULE POOL GENERATION")
    print("=" * 60)
    print(f"  Target pool size:  {args.target}")
    print(f"  Raw generation:    {raw_count} (oversample {args.oversample}x)")
    print(f"  Master seed:       {args.master_seed}")
    print(f"  Compare dirs:      {args.compare_dir or 'none'}")
    print()

    # Load constraints
    allowed_bonds, bead_pool = topo.read_allowed_bonds(str(DATA_DIR / "allowed_bonds.txt"))
    max_degree = topo.read_max_degree(str(DATA_DIR / "max_degree.txt"))
    bond_stats = read_bond_stats(str(DATA_DIR / "bond_stats.txt"))

    print(f"  Bead types: {len(bead_pool)}, Allowed bonds: {len(allowed_bonds)}, "
          f"Degree rules: {len(max_degree)}")

    # Load comparison hashes
    compare_hashes = set()
    for cdir in args.compare_dir:
        h = load_signature_hashes(cdir)
        print(f"  Loaded {len(h)} signatures from {cdir}")
        compare_hashes.update(h)

    # Stage 1: Generate raw pool
    raw_dir = Path(args.outdir) / "_raw"
    print(f"\n{'─'*60}")
    print(f"Stage 1: Generating {raw_count} raw molecules")
    print(f"{'─'*60}")

    molecules = generate_pool(
        total_count=raw_count,
        master_seed=args.master_seed,
        allowed_bonds=allowed_bonds,
        bead_pool=bead_pool,
        max_degree=max_degree,
        bond_stats=bond_stats,
        outdir=raw_dir,
        compare_hashes=compare_hashes,
        retry_count=args.retry_count,
    )

    print_pool_stats(molecules, "Raw Pool")

    if len(molecules) <= args.target or args.skip_diversity:
        selected = molecules
        print(f"\n  Using all {len(molecules)} molecules (no diversity filtering needed)")
    else:
        # Stage 2: Diversity selection
        print(f"\n{'─'*60}")
        print(f"Stage 2: Diversity-based selection ({len(molecules)} → {args.target})")
        print(f"{'─'*60}")

        X_scaled = build_feature_matrix(molecules)
        selected_idx = greedy_maxmin_select(X_scaled, args.target)
        selected = [molecules[i] for i in selected_idx]

        print_pool_stats(selected, "Diverse Subset")

    # Stage 3: Copy selected molecules to final output
    final_dir = Path(args.outdir) / "pool"
    final_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'─'*60}")
    print(f"Stage 3: Assembling final pool in {final_dir}")
    print(f"{'─'*60}")

    manifest_rows = []
    for new_idx, mol in enumerate(selected, 1):
        final_name = f"mol_{new_idx:05d}"
        src = mol["dir"]
        dst = final_dir / final_name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

        # Rename files inside to match new name
        for old_file in dst.iterdir():
            if old_file.stem == mol["name"]:
                new_file = dst / f"{final_name}{old_file.suffix}"
                old_file.rename(new_file)

        manifest_rows.append({
            "molecule": final_name,
            "original": mol["name"],
            "stratum": mol["stratum"],
            "nbeads": mol["nbeads"],
            "has_ring": mol["has_ring"],
            "seed": mol["seed"],
            "sig_hash": mol["sig_hash"],
        })

    manifest_path = Path(args.outdir) / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    # Save generation config for reproducibility
    config = {
        "target": args.target,
        "oversample": args.oversample,
        "raw_generated": len(molecules),
        "final_selected": len(selected),
        "master_seed": args.master_seed,
        "compare_dirs": args.compare_dir,
        "strata": STRATA,
    }
    with open(Path(args.outdir) / "generation_config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    # Clean up raw directory
    print(f"  Cleaning up raw generation directory...")
    shutil.rmtree(raw_dir, ignore_errors=True)

    # Final summary
    strata_dist = Counter(m["stratum"] for m in selected)
    bead_dist = Counter(m["nbeads"] for m in selected)

    print(f"\n{'='*60}")
    print(f"GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Final pool: {len(selected)} diverse molecules")
    print(f"  Output:     {final_dir}/")
    print(f"  Manifest:   {manifest_path}")
    print(f"  Config:     {Path(args.outdir) / 'generation_config.json'}")
    print()
    print(f"  Bead count distribution:")
    for nbeads in sorted(bead_dist.keys()):
        bar = "█" * bead_dist[nbeads]
        print(f"    {nbeads:>2} beads: {bead_dist[nbeads]:>4}  {bar}")
    print()
    print(f"  To use with active learning:")
    print(f"    1. Copy {final_dir}/* to ee_itp_100_new/")
    print(f"    2. Run: bash active_learning/submit_bootstrap.sh")
    print(f"    3. Run: bash active_learning/submit_multi_iteration.sh 5")


if __name__ == "__main__":
    main()
