import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


try:
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
except Exception as exc:
    raise ImportError(
        "scikit-learn is required. Install with: python3 -m pip install scikit-learn"
    ) from exc

try:
    import matplotlib.pyplot as plt
except Exception as exc:
    raise ImportError(
        "matplotlib is required. Install with: python3 -m pip install matplotlib"
    ) from exc


def load_bead_params(path):
    if not path:
        return {}

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"bead-params file not found: {p}")

    bead_params = {}

    # Try CSV first: bead_type,epsilon,sigma
    try:
        with open(p, "r", newline="") as f:
            reader = csv.DictReader(f)
            required = {"bead_type", "epsilon", "sigma"}
            if reader.fieldnames and required.issubset(set(reader.fieldnames)):
                for row in reader:
                    b = row["bead_type"].strip()
                    e = float(row["epsilon"])
                    s = float(row["sigma"])
                    bead_params[b] = (e, s)
                if bead_params:
                    return bead_params
    except Exception:
        pass

    # Fallback: parse Martini-style NBFIX table lines: "TYPE TYPE epsilon sigma"
    with open(p, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(";"):
                continue
            if line.upper() == "NBFIX":
                continue

            parts = line.split()
            if len(parts) < 4:
                continue

            t1, t2 = parts[0], parts[1]
            try:
                eps = float(parts[2])
                sig = float(parts[3])
            except ValueError:
                continue

            # For per-bead features, keep direct self-parameter rows first.
            if t1 == t2:
                bead_params[t1] = (eps, sig)
            elif t1 not in bead_params:
                bead_params[t1] = (eps, sig)

    if not bead_params:
        raise ValueError(
            "Could not parse bead parameters. Provide CSV headers bead_type,epsilon,sigma "
            "or a whitespace NBFIX table."
        )

    return bead_params


def find_molecules(dataset_dirs):
    molecules = []

    for dataset_dir in dataset_dirs:
        root = Path(dataset_dir)
        if not root.exists():
            raise FileNotFoundError(f"Dataset directory not found: {root}")

        for itp_path in root.rglob("*.itp"):
            gro_path = itp_path.with_suffix(".gro")
            molecules.append(
                {
                    "dataset": root.name,
                    "molecule": itp_path.stem,
                    "itp_path": itp_path,
                    "gro_path": gro_path if gro_path.exists() else None,
                }
            )

    molecules.sort(key=lambda x: (x["dataset"], x["molecule"]))
    return molecules


def parse_itp(itp_path):
    atoms = []
    bonds = []

    section = None
    with open(itp_path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(";"):
                continue

            if line.startswith("[") and line.endswith("]"):
                section = line.lower()
                continue

            if ";" in line:
                line = line.split(";", 1)[0].strip()
                if not line:
                    continue

            parts = line.split()

            if section == "[ atoms ]":
                # Expected format: nr type resnr residue atom cgnr charge mass
                if len(parts) < 2:
                    continue
                atom_id = int(parts[0])
                bead_type = parts[1]

                charge = 0.0
                mass = 0.0
                if len(parts) >= 7:
                    try:
                        charge = float(parts[6])
                    except ValueError:
                        charge = 0.0
                if len(parts) >= 8:
                    try:
                        mass = float(parts[7])
                    except ValueError:
                        mass = 0.0

                atoms.append(
                    {
                        "id": atom_id,
                        "type": bead_type,
                        "charge": charge,
                        "mass": mass,
                    }
                )

            elif section == "[ bonds ]":
                # Typical: ai aj funct b0 k
                if len(parts) < 3:
                    continue

                ai = int(parts[0])
                aj = int(parts[1])
                funct = parts[2]
                b0 = float(parts[3]) if len(parts) >= 4 else 0.0
                k = float(parts[4]) if len(parts) >= 5 else 0.0

                bonds.append(
                    {
                        "ai": ai,
                        "aj": aj,
                        "funct": funct,
                        "b0": b0,
                        "k": k,
                    }
                )

    return atoms, bonds


def parse_gro(gro_path):
    if gro_path is None:
        return None

    coords = {}
    with open(gro_path, "r") as f:
        lines = f.readlines()

    if len(lines) < 3:
        return None

    try:
        n_atoms = int(lines[1].strip())
    except ValueError:
        return None

    for line in lines[2 : 2 + n_atoms]:
        # GRO fixed-width: atom index [15:20], x [20:28], y [28:36], z [36:44]
        try:
            idx = int(line[15:20])
            x = float(line[20:28])
            y = float(line[28:36])
            z = float(line[36:44])
            coords[idx] = np.array([x, y, z], dtype=float)
        except Exception:
            continue

    return coords if coords else None


def safe_mean(x):
    return float(np.mean(x)) if len(x) else 0.0


def safe_std(x):
    return float(np.std(x)) if len(x) else 0.0


def compute_graph_density(n, m):
    if n <= 1:
        return 0.0
    return (2.0 * m) / (n * (n - 1))


def compute_components(n_nodes, edges):
    if n_nodes == 0:
        return 0

    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    visited = set()
    components = 0

    node_ids = set(range(1, n_nodes + 1))
    for node in node_ids:
        if node in visited:
            continue

        components += 1
        stack = [node]
        visited.add(node)

        while stack:
            cur = stack.pop()
            for nei in adj[cur]:
                if nei not in visited:
                    visited.add(nei)
                    stack.append(nei)

    return components


def compute_geometry_features(coords):
    if not coords:
        return {
            "has_gro": 0.0,
            "rg": 0.0,
            "d_mean": 0.0,
            "d_std": 0.0,
            "d_max": 0.0,
        }

    arr = np.array(list(coords.values()))
    if len(arr) == 0:
        return {
            "has_gro": 0.0,
            "rg": 0.0,
            "d_mean": 0.0,
            "d_std": 0.0,
            "d_max": 0.0,
        }

    center = np.mean(arr, axis=0)
    rg = float(np.sqrt(np.mean(np.sum((arr - center) ** 2, axis=1))))

    dists = []
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            dists.append(float(np.linalg.norm(arr[i] - arr[j])))

    return {
        "has_gro": 1.0,
        "rg": rg,
        "d_mean": safe_mean(dists),
        "d_std": safe_std(dists),
        "d_max": max(dists) if dists else 0.0,
    }


def molecule_raw_features(mol, bead_params):
    atoms, bonds = parse_itp(mol["itp_path"])
    coords = parse_gro(mol["gro_path"])

    atom_types = [a["type"] for a in atoms]
    charges = [a["charge"] for a in atoms]
    masses = [a["mass"] for a in atoms]

    degree = defaultdict(int)
    bond_type_counts = Counter()
    funct_counts = Counter()
    b0_vals = []
    k_vals = []

    for b in bonds:
        i, j = b["ai"], b["aj"]
        degree[i] += 1
        degree[j] += 1

        ti = atoms[i - 1]["type"] if 1 <= i <= len(atoms) else "UNK"
        tj = atoms[j - 1]["type"] if 1 <= j <= len(atoms) else "UNK"
        pair = tuple(sorted((ti, tj)))
        bond_type_counts[pair] += 1

        funct_counts[str(b["funct"])] += 1
        b0_vals.append(float(b["b0"]))
        k_vals.append(float(b["k"]))

    degree_vals = [degree[a["id"]] for a in atoms]

    # Optional epsilon/sigma derived from bead type lookup table.
    eps_vals = []
    sig_vals = []
    for bt in atom_types:
        if bt in bead_params:
            e, s = bead_params[bt]
        else:
            e, s = 0.0, 0.0
        eps_vals.append(e)
        sig_vals.append(s)

    n = len(atoms)
    m = len(bonds)
    comps = compute_components(n, [(b["ai"], b["aj"]) for b in bonds])
    cyclomatic = max(0, m - n + comps)

    global_feats = {
        "num_atoms": float(n),
        "num_bonds": float(m),
        "avg_degree": safe_mean(degree_vals),
        "max_degree": float(max(degree_vals) if degree_vals else 0.0),
        "graph_density": compute_graph_density(n, m),
        "total_charge": float(sum(charges)),
        "charge_std": safe_std(charges),
        "unique_bead_types": float(len(set(atom_types))),
        "cyclomatic_number": float(cyclomatic),
    }

    node_numeric = {
        "mass_mean": safe_mean(masses),
        "mass_std": safe_std(masses),
        "charge_mean": safe_mean(charges),
        "charge_std_node": safe_std(charges),
        "degree_mean": safe_mean(degree_vals),
        "degree_std": safe_std(degree_vals),
        "epsilon_mean": safe_mean(eps_vals),
        "epsilon_std": safe_std(eps_vals),
        "sigma_mean": safe_mean(sig_vals),
        "sigma_std": safe_std(sig_vals),
    }

    edge_numeric = {
        "b0_mean": safe_mean(b0_vals),
        "b0_std": safe_std(b0_vals),
        "k_mean": safe_mean(k_vals),
        "k_std": safe_std(k_vals),
    }

    geometry = compute_geometry_features(coords)

    return {
        "meta": mol,
        "atom_type_counts": Counter(atom_types),
        "bond_type_counts": bond_type_counts,
        "funct_counts": funct_counts,
        "global": global_feats,
        "node_numeric": node_numeric,
        "edge_numeric": edge_numeric,
        "geometry": geometry,
    }


def build_feature_matrix(raw_list):
    bead_vocab = sorted({k for r in raw_list for k in r["atom_type_counts"].keys()})
    bond_vocab = sorted({k for r in raw_list for k in r["bond_type_counts"].keys()})
    funct_vocab = sorted({k for r in raw_list for k in r["funct_counts"].keys()})

    fixed_keys = (
        [
            "num_atoms",
            "num_bonds",
            "avg_degree",
            "max_degree",
            "graph_density",
            "total_charge",
            "charge_std",
            "unique_bead_types",
            "cyclomatic_number",
        ]
        + [
            "mass_mean",
            "mass_std",
            "charge_mean",
            "charge_std_node",
            "degree_mean",
            "degree_std",
            "epsilon_mean",
            "epsilon_std",
            "sigma_mean",
            "sigma_std",
        ]
        + ["b0_mean", "b0_std", "k_mean", "k_std"]
        + ["has_gro", "rg", "d_mean", "d_std", "d_max"]
    )

    feature_names = []
    feature_names.extend(fixed_keys)
    feature_names.extend([f"bead_frac::{b}" for b in bead_vocab])
    feature_names.extend([f"bond_frac::{a}-{b}" for (a, b) in bond_vocab])
    feature_names.extend([f"funct_frac::{f}" for f in funct_vocab])

    X = []
    meta_rows = []

    for r in raw_list:
        n_atoms = max(1.0, r["global"]["num_atoms"])
        n_bonds = max(1.0, r["global"]["num_bonds"])

        row = []

        for k in fixed_keys:
            if k in r["global"]:
                row.append(r["global"][k])
            elif k in r["node_numeric"]:
                row.append(r["node_numeric"][k])
            elif k in r["edge_numeric"]:
                row.append(r["edge_numeric"][k])
            else:
                row.append(r["geometry"][k])

        for b in bead_vocab:
            row.append(r["atom_type_counts"].get(b, 0.0) / n_atoms)

        for pair in bond_vocab:
            row.append(r["bond_type_counts"].get(pair, 0.0) / n_bonds)

        for f in funct_vocab:
            row.append(r["funct_counts"].get(f, 0.0) / n_bonds)

        X.append(row)

        meta_rows.append(
            {
                "dataset": r["meta"]["dataset"],
                "molecule": r["meta"]["molecule"],
                "itp_path": str(r["meta"]["itp_path"]),
                "gro_path": str(r["meta"]["gro_path"]) if r["meta"]["gro_path"] else "",
            }
        )

    X = np.array(X, dtype=float)
    return X, feature_names, meta_rows


def compute_uniqueness(X_scaled, k_neighbors):
    n = len(X_scaled)
    k = min(k_neighbors + 1, n)
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(X_scaled)
    dists, _ = nn.kneighbors(X_scaled)

    # Exclude self-distance at index 0.
    uniq = np.mean(dists[:, 1:], axis=1) if k > 1 else np.zeros(n)
    return uniq


def run_umap(X_scaled, n_neighbors, min_dist, random_state):
    try:
        import umap
    except Exception:
        raise ImportError(
            "UMAP is required for this minimal workflow. Install with: python3 -m pip install umap-learn"
        )

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=random_state,
    )
    return reducer.fit_transform(X_scaled)


def plot_umap_uniqueness(coords, uniqueness, out_path):
    plt.figure(figsize=(9, 7))

    x = coords[:, 0]
    y = coords[:, 1]

    sc = plt.scatter(x, y, c=uniqueness, cmap="viridis", s=30, alpha=0.9)
    cbar = plt.colorbar(sc)
    cbar.set_label("Uniqueness score (kNN distance)")

    plt.title("UMAP (colored by uniqueness)")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def greedy_maxmin_subset(X_scaled, start_idx, k):
    n = len(X_scaled)
    k = min(k, n)

    dmat = np.linalg.norm(X_scaled[:, None, :] - X_scaled[None, :, :], axis=2)

    selected = [start_idx]
    remaining = set(range(n))
    remaining.remove(start_idx)
    min_d = dmat[:, start_idx].copy()

    for _ in range(k - 1):
        idx = max(remaining, key=lambda i: min_d[i])
        selected.append(idx)
        remaining.remove(idx)
        min_d = np.minimum(min_d, dmat[:, idx])

    return selected


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Minimal molecule analysis: UMAP + uniqueness + diverse subset."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Dataset directories containing molecule subfolders with .itp files",
    )
    parser.add_argument(
        "--outdir",
        default="molecule_space_analysis",
        help="Output directory for minimal analysis outputs",
    )
    parser.add_argument(
        "--bead-params",
        default=None,
        help=(
            "Optional bead parameter file. Supported formats: "
            "CSV (bead_type,epsilon,sigma) or whitespace NBFIX table"
        ),
    )
    parser.add_argument("--k-neighbors", type=int, default=5, help="k for uniqueness score")
    parser.add_argument("--umap-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--diverse-k", type=int, default=20, help="Number of molecules in distance-diverse subset")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    bead_params = load_bead_params(args.bead_params)
    molecules = find_molecules(args.datasets)

    if not molecules:
        raise RuntimeError("No .itp files found in provided dataset directories.")

    raw = [molecule_raw_features(m, bead_params) for m in molecules]
    X, feature_names, meta_rows = build_feature_matrix(raw)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    uniqueness = compute_uniqueness(X_scaled, k_neighbors=args.k_neighbors)

    uniq_csv = outdir / "uniqueness_scores.csv"
    uniq_rows = []
    for i, m in enumerate(meta_rows):
        uniq_rows.append(
            [
                m["dataset"],
                m["molecule"],
                float(uniqueness[i]),
                m["itp_path"],
                m["gro_path"],
            ]
        )
    uniq_rows.sort(key=lambda r: r[2], reverse=True)
    write_csv(
        uniq_csv,
        ["dataset", "molecule", "uniqueness_score", "itp_path", "gro_path"],
        uniq_rows,
    )

    # UMAP + plot (required for this workflow)
    emb_umap = run_umap(
        X_scaled,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        random_state=args.seed,
    )

    umap_csv = outdir / "embedding_umap.csv"
    umap_rows = [
        [
            meta_rows[i]["dataset"],
            meta_rows[i]["molecule"],
            float(emb_umap[i, 0]),
            float(emb_umap[i, 1]),
            float(uniqueness[i]),
        ]
        for i in range(len(meta_rows))
    ]
    write_csv(umap_csv, ["dataset", "molecule", "x", "y", "uniqueness_score"], umap_rows)

    plot_umap_uniqueness(
        emb_umap,
        uniqueness,
        outdir / "plot_umap_uniqueness.png",
    )

    # Distance-diverse subset using greedy max-min selection.
    top_unique_name = uniq_rows[0][1]
    start_idx = next(i for i, m in enumerate(meta_rows) if m["molecule"] == top_unique_name)
    diverse_idx = greedy_maxmin_subset(X_scaled, start_idx=start_idx, k=args.diverse_k)

    diverse_rows = []
    for rank, idx in enumerate(diverse_idx, start=1):
        m = meta_rows[idx]
        diverse_rows.append(
            [
                rank,
                m["dataset"],
                m["molecule"],
                m["itp_path"],
                m["gro_path"],
                top_unique_name,
            ]
        )
    write_csv(
        outdir / f"diverse_subset_k{args.diverse_k}.csv",
        ["rank", "dataset", "molecule", "itp_path", "gro_path", "seed_molecule"],
        diverse_rows,
    )

    print(f"Analyzed molecules: {len(meta_rows)}")
    print(f"Outputs written to: {outdir}")
    print(f"Files: uniqueness_scores.csv, embedding_umap.csv, plot_umap_uniqueness.png, diverse_subset_k{args.diverse_k}.csv")


if __name__ == "__main__":
    main()
