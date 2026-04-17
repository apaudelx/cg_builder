"""Rebuild the generator's data library from existing molecule datasets.

Parses all .itp files in one or more dataset directories to extract:
  1. allowed_bonds.txt  — all observed bead-type pairs that form bonds
  2. max_degree.txt     — maximum graph degree observed for each bead type
  3. bond_stats.txt     — mean, std, and count of bond lengths per pair

This expands the generator's vocabulary to cover ALL bead types present in
the training data, not just the subset the original library was built from.

Usage:
    # Rebuild from the 667 training set
    python cg_builder/rebuild_library.py \
        --datasets data/ee_itp_667 \
        --outdir cg_builder/data

    # Rebuild from multiple datasets, merging with existing library
    python cg_builder/rebuild_library.py \
        --datasets data/ee_itp_667 ee_itp_100_new \
        --existing-dir cg_builder/data \
        --outdir cg_builder/data_expanded

    # Dry run: show stats without writing
    python cg_builder/rebuild_library.py \
        --datasets data/ee_itp_667 \
        --dry-run
"""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def parse_itp(itp_path):
    """Parse atoms (id, type) and bonds (ai, aj, b0) from an .itp file."""
    atoms = {}
    bonds = []
    section = None

    with open(itp_path) as f:
        for raw in f:
            line = raw.strip()
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
                atom_id = int(parts[0])
                bead_type = parts[1]
                atoms[atom_id] = bead_type

            elif section == "[ bonds ]" and len(parts) >= 2:
                ai, aj = int(parts[0]), int(parts[1])
                b0 = float(parts[3]) if len(parts) >= 4 else None
                bonds.append((ai, aj, b0))

    return atoms, bonds


def find_itp_files(dataset_dirs):
    """Find all .itp files across dataset directories."""
    itp_files = []
    for d in dataset_dirs:
        root = Path(d)
        if not root.exists():
            print(f"  Warning: {root} does not exist, skipping")
            continue
        for itp in root.rglob("*.itp"):
            itp_files.append(itp)
    return sorted(itp_files)


def mine_datasets(itp_files):
    """Extract bond pairs, degree stats, and bond lengths from all .itp files."""
    bond_pairs = set()
    degree_per_type = defaultdict(list)
    bond_lengths = defaultdict(list)
    all_bead_types = set()

    parsed = 0
    errors = 0

    for itp_path in itp_files:
        try:
            atoms, bonds = parse_itp(itp_path)
        except Exception:
            errors += 1
            continue

        if not atoms:
            continue

        parsed += 1
        all_bead_types.update(atoms.values())

        degree = Counter()
        for ai, aj, b0 in bonds:
            ti = atoms.get(ai)
            tj = atoms.get(aj)
            if ti is None or tj is None:
                continue

            pair = frozenset((ti, tj))
            bond_pairs.add(pair)

            degree[ai] += 1
            degree[aj] += 1

            canonical = tuple(sorted((ti, tj)))
            if b0 is not None and b0 > 0:
                bond_lengths[canonical].append(b0)

        for atom_id, bead_type in atoms.items():
            degree_per_type[bead_type].append(degree.get(atom_id, 0))

    return {
        "bond_pairs": bond_pairs,
        "degree_per_type": degree_per_type,
        "bond_lengths": bond_lengths,
        "all_bead_types": all_bead_types,
        "parsed": parsed,
        "errors": errors,
    }


def load_existing_library(existing_dir):
    """Load existing allowed_bonds, max_degree, and bond_stats."""
    existing_dir = Path(existing_dir)
    existing = {
        "bond_pairs": set(),
        "max_degree": {},
        "bond_stats": {},
    }

    ab_path = existing_dir / "allowed_bonds.txt"
    if ab_path.exists():
        with open(ab_path) as f:
            for line in f:
                parts = line.split()
                if len(parts) == 2:
                    existing["bond_pairs"].add(frozenset(parts))

    md_path = existing_dir / "max_degree.txt"
    if md_path.exists():
        with open(md_path) as f:
            for line in f:
                parts = line.split()
                if len(parts) == 2:
                    existing["max_degree"][parts[0]] = int(parts[1])

    bs_path = existing_dir / "bond_stats.txt"
    if bs_path.exists():
        with open(bs_path) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 4:
                    pair = (parts[0], parts[1])
                    existing["bond_stats"][pair] = {
                        "mean": float(parts[2]),
                        "std": float(parts[3]),
                        "count": int(parts[4]) if len(parts) >= 5 else 1,
                    }

    return existing


def merge_and_write(mined, existing, outdir, dry_run=False):
    """Merge mined data with existing library and write output files."""
    outdir = Path(outdir)

    # ── Allowed bonds ─────────────────────────────────────────────────────────
    all_pairs = mined["bond_pairs"] | existing["bond_pairs"]

    # ── Max degree ────────────────────────────────────────────────────────────
    max_deg = dict(existing["max_degree"])
    for btype, degrees in mined["degree_per_type"].items():
        observed_max = max(degrees) if degrees else 0
        if btype in max_deg:
            max_deg[btype] = max(max_deg[btype], observed_max)
        else:
            max_deg[btype] = observed_max

    for btype in mined["all_bead_types"]:
        if btype not in max_deg:
            max_deg[btype] = 1

    # ── Bond stats ────────────────────────────────────────────────────────────
    bond_stats = {}

    for pair_tuple, lengths in mined["bond_lengths"].items():
        if len(lengths) > 0:
            bond_stats[pair_tuple] = {
                "mean": float(np.mean(lengths)),
                "std": float(np.std(lengths)) if len(lengths) > 1 else 0.01,
                "count": len(lengths),
            }

    for pair_tuple, stats in existing["bond_stats"].items():
        if pair_tuple not in bond_stats:
            bond_stats[pair_tuple] = stats
        else:
            m = bond_stats[pair_tuple]
            e = stats
            total = m["count"] + e["count"]
            combined_mean = (m["mean"] * m["count"] + e["mean"] * e["count"]) / total
            combined_var = (
                (m["count"] * (m["std"]**2 + m["mean"]**2) +
                 e["count"] * (e["std"]**2 + e["mean"]**2)) / total
                - combined_mean**2
            )
            bond_stats[pair_tuple] = {
                "mean": combined_mean,
                "std": max(0.001, float(np.sqrt(max(0, combined_var)))),
                "count": total,
            }

    # For allowed bond pairs with no length data, add a default
    for pair in all_pairs:
        items = sorted(pair)
        canonical = (items[0], items[0]) if len(items) == 1 else (items[0], items[1])
        if canonical not in bond_stats:
            bond_stats[canonical] = {
                "mean": 0.27,
                "std": 0.03,
                "count": 0,
            }

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"LIBRARY SUMMARY")
    print(f"{'='*60}")

    existing_types = set(existing["max_degree"].keys())
    mined_types = mined["all_bead_types"]
    new_types = mined_types - existing_types

    existing_pairs = existing["bond_pairs"]
    new_pairs = mined["bond_pairs"] - existing_pairs

    print(f"  Bead types:    {len(max_deg)} total "
          f"({len(existing_types)} existing + {len(new_types)} new)")
    print(f"  Allowed bonds: {len(all_pairs)} total "
          f"({len(existing_pairs)} existing + {len(new_pairs)} new)")
    print(f"  Bond stats:    {len(bond_stats)} total")
    print()

    if new_types:
        print(f"  New bead types ({len(new_types)}):")
        for t in sorted(new_types):
            print(f"    {t} (max_degree={max_deg[t]})")
        print()

    # Degree distribution
    degree_dist = Counter(max_deg.values())
    print(f"  Degree distribution:")
    for deg in sorted(degree_dist.keys()):
        count = degree_dist[deg]
        print(f"    degree {deg}: {count} bead types")
    print()

    if dry_run:
        print("  [DRY RUN] No files written.")
        return

    # ── Write files ───────────────────────────────────────────────────────────
    outdir.mkdir(parents=True, exist_ok=True)

    # allowed_bonds.txt
    ab_path = outdir / "allowed_bonds.txt"
    with open(ab_path, "w") as f:
        for pair in sorted(all_pairs, key=lambda p: tuple(sorted(p))):
            items = sorted(pair)
            if len(items) == 1:
                f.write(f"{items[0]} {items[0]}\n")
            else:
                f.write(f"{items[0]} {items[1]}\n")
    print(f"  Written: {ab_path} ({len(all_pairs)} pairs)")

    # max_degree.txt
    md_path = outdir / "max_degree.txt"
    with open(md_path, "w") as f:
        for btype in sorted(max_deg.keys()):
            f.write(f"{btype} {max_deg[btype]}\n")
    print(f"  Written: {md_path} ({len(max_deg)} types)")

    # bond_stats.txt
    bs_path = outdir / "bond_stats.txt"
    with open(bs_path, "w") as f:
        for pair in sorted(bond_stats.keys()):
            s = bond_stats[pair]
            f.write(f"{pair[0]} {pair[1]} {s['mean']:.4f} {s['std']:.4f} {s['count']}\n")
    print(f"  Written: {bs_path} ({len(bond_stats)} entries)")


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild generator data library from molecule datasets."
    )
    parser.add_argument("--datasets", nargs="+", required=True,
                        help="Dataset directories containing molecule subfolders with .itp files")
    parser.add_argument("--existing-dir", default=None,
                        help="Existing data directory to merge with (default: start fresh)")
    parser.add_argument("--outdir", default="cg_builder/data_expanded",
                        help="Output directory for expanded data files")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show stats without writing files")

    args = parser.parse_args()

    print("=" * 60)
    print("REBUILD GENERATOR LIBRARY")
    print("=" * 60)
    print(f"  Datasets:     {args.datasets}")
    print(f"  Existing dir: {args.existing_dir or 'none (fresh build)'}")
    print(f"  Output dir:   {args.outdir}")
    print()

    # Find all .itp files
    itp_files = find_itp_files(args.datasets)
    print(f"  Found {len(itp_files)} .itp files")

    if not itp_files:
        print("  Error: no .itp files found")
        return

    # Mine the datasets
    print(f"\n  Mining bond pairs, degrees, and bond lengths...")
    mined = mine_datasets(itp_files)
    print(f"  Parsed {mined['parsed']} molecules ({mined['errors']} errors)")
    print(f"  Found {len(mined['all_bead_types'])} unique bead types")
    print(f"  Found {len(mined['bond_pairs'])} unique bond pairs")
    print(f"  Found {len(mined['bond_lengths'])} bond-pair length distributions")

    # Load existing library
    if args.existing_dir:
        print(f"\n  Loading existing library from {args.existing_dir}...")
        existing = load_existing_library(args.existing_dir)
        print(f"  Existing: {len(existing['bond_pairs'])} pairs, "
              f"{len(existing['max_degree'])} types, "
              f"{len(existing['bond_stats'])} bond stats")
    else:
        existing = {"bond_pairs": set(), "max_degree": {}, "bond_stats": {}}

    # Merge and write
    merge_and_write(mined, existing, args.outdir, dry_run=args.dry_run)

    if not args.dry_run:
        print(f"\nDone. Updated library in: {args.outdir}/")
        print(f"\nTo use with the generator, either:")
        print(f"  1. Copy files to cg_builder/data/")
        print(f"  2. Or update generator scripts to point to {args.outdir}/")


if __name__ == "__main__":
    main()
