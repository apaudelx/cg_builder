import argparse
import csv
import hashlib
import random
from collections import Counter
from pathlib import Path

import numpy as np
import networkx as nx

import topo_generator as topo


MIN_BOND = 0.2
MAX_BOND = 0.4
MIN_NONBONDED = 0.3


def read_bond_stats(filename="data/bond_stats.txt"):
    stats = {}
    with open(filename) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            a, b = parts[0], parts[1]
            mean = float(parts[2])
            std = float(parts[3])
            stats[frozenset((a, b))] = (mean, std)
    return stats


def get_bond_length(ti, tj, bond_stats):
    key = frozenset((ti, tj))
    if key in bond_stats:
        d = bond_stats[key][0]
    else:
        d = 0.27
    return max(MIN_BOND, min(MAX_BOND, d))


def sample_bond_length(ti, tj, bond_stats):
    key = frozenset((ti, tj))
    if key in bond_stats:
        mean, std = bond_stats[key]
        d = random.gauss(mean, std)
    else:
        d = 0.27
    return max(MIN_BOND, min(MAX_BOND, d))


def random_unit_vector():
    v = np.random.normal(size=3)
    return v / np.linalg.norm(v)


def is_valid_position(new_pos, coords, exclude_ids=None):
    for k, pos in coords.items():
        if exclude_ids and k in exclude_ids:
            continue
        if np.linalg.norm(new_pos - pos) < MIN_NONBONDED:
            return False
    return True


def generate_coordinates(beads, bonds, bond_stats):
    g = nx.Graph()
    g.add_edges_from(bonds)

    coords = {}
    start = list(beads.keys())[0]
    coords[start] = np.array([0.0, 0.0, 0.0])

    visited = {start}
    queue = [start]

    while queue:
        u = queue.pop(0)
        for v in g.neighbors(u):
            if v in visited:
                continue

            d = sample_bond_length(beads[u], beads[v], bond_stats)
            placed = False

            for _ in range(50):
                trial = coords[u] + d * random_unit_vector()
                if is_valid_position(trial, coords, exclude_ids={u}):
                    coords[v] = trial
                    placed = True
                    break

            if not placed:
                coords[v] = coords[u] + d * random_unit_vector()

            visited.add(v)
            queue.append(v)

    return coords


def get_mass(bead_type):
    if bead_type.startswith("S"):
        return 54.0
    if bead_type.startswith("T"):
        return 36.0
    return 72.0


def write_itp(beads, bonds, bond_stats, filename, k=20000):
    with open(filename, "w") as f:
        f.write("[ moleculetype ]\n")
        f.write("; name        nrexcl\n")
        f.write("res           1\n\n")

        f.write("[ atoms ]\n")
        f.write(";    nr  type  resnr resid  atom  cgnr     charge       mass\n")

        for i in sorted(beads):
            btype = beads[i]
            resid = f"C{i}"
            mass = get_mass(btype)
            f.write(
                f"{i:7d} {btype:6s} {1:6d} res   {resid:5s} {i:6d} {0.0:10.3f} {mass:10.3f}\n"
            )

        f.write("\n[ bonds ]\n")
        f.write(";  ai    aj   funct      b0(nm)          k(kJ/mol/nm^2)\n")
        for i, j in bonds:
            r0 = get_bond_length(beads[i], beads[j], bond_stats)
            f.write(f"{i:6d} {j:6d} 1 {r0:12.4f} {k:16.1f}\n")


def write_gro(beads, coords, filename):
    with open(filename, "w") as f:
        f.write("Generated molecule\n")
        f.write(f"{len(beads)}\n")
        for i in sorted(beads):
            x, y, z = coords[i]
            atom_name = f"C{i}"
            f.write(
                f"{1:5d}{'res':<5s}{atom_name:>5s}{i:5d}{x:8.3f}{y:8.3f}{z:8.3f}\n"
            )
        f.write(f"{10.0:10.5f}{10.0:10.5f}{10.0:10.5f}\n")


def build_topology(nbeads, seed, allowed_bonds, bead_pool, max_degree):
    random.seed(seed)

    beads, bonds, degree, bonds_set, rings = topo.build_rings(
        n3=0,
        n4=0,
        n5=0,
        n6=0,
        allowed_bonds=allowed_bonds,
        bead_pool=bead_pool,
        max_degree=max_degree,
        max_tries=10000,
    )

    topo.add_extra_beads(
        target_N=nbeads,
        beads=beads,
        bonds=bonds,
        degree=degree,
        bonds_set=bonds_set,
        allowed_bonds=allowed_bonds,
        bead_pool=bead_pool,
        max_degree=max_degree,
        original_beads=set(bead_pool),
        forced_beads=[],
    )

    return beads, bonds


def build_coordinates(beads, bonds, seed, bond_stats):
    random.seed(seed)
    np.random.seed(seed)
    return generate_coordinates(beads, bonds, bond_stats)


def ensure_empty_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def build_bead_schedule(count, min_beads, max_beads, mode, rng):
    if mode == "random":
        return [rng.randint(min_beads, max_beads) for _ in range(count)]

    bead_values = list(range(min_beads, max_beads + 1))
    n_bins = len(bead_values)
    base = count // n_bins
    remainder = count % n_bins

    schedule = []
    for b in bead_values:
        schedule.extend([b] * base)

    # Distribute remainder deterministically with seeded RNG.
    extra = bead_values[:]
    rng.shuffle(extra)
    schedule.extend(extra[:remainder])

    rng.shuffle(schedule)
    return schedule


def parse_itp_beads_bonds(itp_path):
    beads = {}
    bonds = []
    section = None

    with open(itp_path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith(";"):
                continue

            if line.startswith("[") and line.endswith("]"):
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
                i, j = int(parts[0]), int(parts[1])
                bonds.append((i, j))

    return beads, bonds


def graph_signature(beads, bonds):
    node_counter = Counter(beads.values())

    edge_counter = Counter()
    for i, j in bonds:
        ti = beads.get(i, "UNK")
        tj = beads.get(j, "UNK")
        pair = tuple(sorted((ti, tj)))
        edge_counter[pair] += 1

    node_part = ";".join(f"{k}:{v}" for k, v in sorted(node_counter.items()))
    edge_part = ";".join(f"{a}-{b}:{c}" for (a, b), c in sorted(edge_counter.items()))
    signature = f"N|{node_part}|E|{edge_part}"
    sig_hash = hashlib.sha1(signature.encode("utf-8")).hexdigest()
    return signature, sig_hash


def load_signature_hashes(dataset_dir):
    dataset_root = Path(dataset_dir)
    if not dataset_root.exists():
        raise FileNotFoundError(f"--compare-dir does not exist: {dataset_root}")

    hashes = set()
    for itp_path in dataset_root.rglob("*.itp"):
        beads, bonds = parse_itp_beads_bonds(itp_path)
        if not beads:
            continue
        _, sig_hash = graph_signature(beads, bonds)
        hashes.add(sig_hash)

    return hashes


def main():
    parser = argparse.ArgumentParser(
        description="Generate a structured dataset of molecules with reproducible randomness."
    )
    parser.add_argument("--count", type=int, default=100, help="Number of molecules to generate")
    parser.add_argument("--min-beads", type=int, default=6, help="Minimum bead count")
    parser.add_argument("--max-beads", type=int, default=15, help="Maximum bead count")
    parser.add_argument("--master-seed", type=int, required=True, help="Master seed for reproducibility")
    parser.add_argument("--prefix", type=str, default="bigm", help="Prefix for molecule folder/file names")
    parser.add_argument("--outdir", type=str, default="generated_dataset", help="Output dataset folder")
    parser.add_argument(
        "--bead-allocation",
        choices=["random", "balanced"],
        default="random",
        help="random: uniform random bead count; balanced: near-equal count per bead size",
    )
    parser.add_argument(
        "--compare-dir",
        type=str,
        default=None,
        help="Optional dataset directory to flag topology-signature overlaps",
    )
    parser.add_argument(
        "--retry-count",
        type=int,
        default=5,
        help="Max retry attempts if generation fails for a molecule",
    )

    args = parser.parse_args()

    if args.min_beads < 1:
        raise ValueError("--min-beads must be >= 1")
    if args.max_beads < args.min_beads:
        raise ValueError("--max-beads must be >= --min-beads")
    if args.count < 1:
        raise ValueError("--count must be >= 1")
    if args.retry_count < 0:
        raise ValueError("--retry-count must be >= 0")

    outdir = Path(args.outdir)
    ensure_empty_dir(outdir)

    allowed_bonds, bead_pool = topo.read_allowed_bonds("data/allowed_bonds.txt")
    max_degree = topo.read_max_degree("data/max_degree.txt")
    bond_stats = read_bond_stats("data/bond_stats.txt")

    selector_rng = random.Random(args.master_seed)
    bead_schedule = build_bead_schedule(
        count=args.count,
        min_beads=args.min_beads,
        max_beads=args.max_beads,
        mode=args.bead_allocation,
        rng=selector_rng,
    )

    compare_hashes = set()
    if args.compare_dir:
        compare_hashes = load_signature_hashes(args.compare_dir)

    manifest_path = outdir / "manifest.csv"
    overlap_path = outdir / "overlap_report.csv"

    rows = []
    overlap_rows = []

    for idx in range(1, args.count + 1):
        mol_name = f"{args.prefix}_{idx:03d}"
        mol_dir = outdir / mol_name
        mol_dir.mkdir(parents=True, exist_ok=True)

        nbeads = bead_schedule[idx - 1]
        base_seed = args.master_seed + idx

        success = False
        last_error = ""
        used_seed = None
        sig_hash = ""
        duplicate_in_compare = ""

        for attempt in range(args.retry_count + 1):
            candidate_seed = base_seed + attempt * 1000

            try:
                beads, bonds = build_topology(
                    nbeads=nbeads,
                    seed=candidate_seed,
                    allowed_bonds=allowed_bonds,
                    bead_pool=bead_pool,
                    max_degree=max_degree,
                )

                coords = build_coordinates(
                    beads=beads,
                    bonds=bonds,
                    seed=candidate_seed,
                    bond_stats=bond_stats,
                )

                itp_path = mol_dir / f"{mol_name}.itp"
                gro_path = mol_dir / f"{mol_name}.gro"

                write_itp(
                    beads=beads,
                    bonds=bonds,
                    bond_stats=bond_stats,
                    filename=str(itp_path),
                )
                write_gro(beads=beads, coords=coords, filename=str(gro_path))

                _, sig_hash = graph_signature(beads, bonds)
                if args.compare_dir:
                    duplicate_in_compare = "yes" if sig_hash in compare_hashes else "no"

                used_seed = candidate_seed
                success = True
                break
            except Exception as exc:
                last_error = str(exc)

        rows.append(
            {
                "molecule": mol_name,
                "nbeads": nbeads,
                "base_seed": base_seed,
                "used_seed": used_seed if used_seed is not None else "",
                "status": "ok" if success else "failed",
                "signature_hash": sig_hash,
                "duplicate_in_compare": duplicate_in_compare,
                "error": "" if success else last_error,
            }
        )

        if success and args.compare_dir:
            overlap_rows.append(
                {
                    "molecule": mol_name,
                    "signature_hash": sig_hash,
                    "duplicate_in_compare": duplicate_in_compare,
                }
            )

        status_msg = "ok" if success else "failed"
        print(f"[{idx:03d}/{args.count}] {mol_name} beads={nbeads} seed={used_seed} status={status_msg}")

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "molecule",
                "nbeads",
                "base_seed",
                "used_seed",
                "status",
                "signature_hash",
                "duplicate_in_compare",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    if args.compare_dir:
        with open(overlap_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["molecule", "signature_hash", "duplicate_in_compare"],
            )
            writer.writeheader()
            writer.writerows(overlap_rows)

    failed = sum(1 for r in rows if r["status"] != "ok")
    overlap_count = sum(1 for r in rows if r.get("duplicate_in_compare") == "yes")
    print(f"\nDone. Output: {outdir}")
    print(f"Manifest: {manifest_path}")
    print(f"Successful: {args.count - failed}, Failed: {failed}")
    if args.compare_dir:
        print(f"Overlap report: {overlap_path}")
        print(f"Topological overlaps with compare set: {overlap_count}")

    if failed:
        raise RuntimeError("Some molecules failed to generate. Check manifest.csv for details.")


if __name__ == "__main__":
    main()
