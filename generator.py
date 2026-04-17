import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt

MIN_BOND = 0.2
MAX_BOND = 0.4
MIN_NONBONDED = 0.3

def read_itp(filename):
    beads = {}
    bonds = []

    section = None

    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";"):
                continue

            if line.startswith("["):
                section = line.lower()
                continue

            parts = line.split()

            if "[ atoms ]" in section:
                beads[int(parts[0])] = parts[1]

            elif "[ bonds ]" in section:
                bonds.append((int(parts[0]), int(parts[1])))

    return beads, bonds


def read_gro(filename):
    coords = {}

    with open(filename) as f:
        lines = f.readlines()

    natoms = int(lines[1].strip())

    for line in lines[2:2+natoms]:
        idx = int(line[15:20])
        x = float(line[20:28])
        y = float(line[28:36])
        z = float(line[36:44])
        coords[idx] = np.array([x, y, z])

    return coords


def is_valid_position(new_pos, coords, exclude_ids=None):
    for k, pos in coords.items():
        if exclude_ids and k in exclude_ids:
            continue
        if np.linalg.norm(new_pos - pos) < MIN_NONBONDED:
            return False
    return True


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


def write_itp(beads, bonds, bond_stats, filename="generated.itp", k=20000):
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
            ti = beads[i]
            tj = beads[j]
            r0 = get_bond_length(ti, tj, bond_stats)

            f.write(f"{i:6d} {j:6d} 1 {r0:12.4f} {k:16.1f}\n")


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


def generate_coordinates(beads, bonds, bond_stats):
    G = nx.Graph()
    G.add_edges_from(bonds)

    coords = {}

    start = list(beads.keys())[0]
    coords[start] = np.array([0.0, 0.0, 0.0])

    visited = set([start])
    queue = [start]

    while queue:
        u = queue.pop(0)

        for v in G.neighbors(u):
            if v in visited:
                continue

            ti = beads[u]
            tj = beads[v]
            d = sample_bond_length(ti, tj, bond_stats)

            placed = False

            for _ in range(50):
                direction = random_unit_vector()
                trial = coords[u] + d * direction

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
    elif bead_type.startswith("T"):
        return 36.0
    else:
        return 72.0


def write_gro(beads, coords, filename="generated.gro"):
    with open(filename, "w") as f:
        f.write("Generated molecule\n")
        f.write(f"{len(beads)}\n")

        for i in sorted(beads):
            x, y, z = coords[i]
            atom_name = f"C{i}"
            f.write(
                f"{1:5d}{'res':<5s}{atom_name:>5s}{i:5d}"
                f"{x:8.3f}{y:8.3f}{z:8.3f}\n"
            )

        f.write(f"{10.0:10.5f}{10.0:10.5f}{10.0:10.5f}\n")


def draw_molecule_3d_projection(itp_file, gro_file, out_png="molecule.png"):
    atom_types, bonds = read_itp(itp_file)
    coords = read_gro(gro_file)

    G = nx.Graph()

    for i in atom_types:
        G.add_node(i)

    for i, j in bonds:
        G.add_edge(i, j)

    pos = {i: coords[i][:2] for i in atom_types if i in coords}
    labels = {i: f"{i}\n{atom_types[i]}" for i in atom_types if i in coords}

    plt.figure(figsize=(6, 6))
    nx.draw(G, pos, with_labels=False, node_size=500)
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    plt.title("geometry-based projection")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    #plt.show()
    
import plotly.graph_objects as go


def draw_3d_interactive(itp_file, gro_file):
    atom_types, bonds = read_itp(itp_file)
    coords = read_gro(gro_file)

    xs, ys, zs = [], [], []
    labels = []

    for i in coords:
        x, y, z = coords[i]
        xs.append(x)
        ys.append(y)
        zs.append(z)
        labels.append(f"{i}:{atom_types.get(i,'')}")

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode='markers+text',
        text=labels,
        marker=dict(size=5)
    ))

    for i, j in bonds:
        if i in coords and j in coords:
            xi, yi, zi = coords[i]
            xj, yj, zj = coords[j]

            fig.add_trace(go.Scatter3d(
                x=[xi, xj],
                y=[yi, yj],
                z=[zi, zj],
                mode='lines'
            ))

    fig.write_html("molecule_3d.html")   # ← ADD THIS
    print("molecule_3d.html written")

if __name__ == "__main__":
    beads, bonds = read_itp("generated.itp")
    bond_stats = read_bond_stats("data/bond_stats.txt")

    write_itp(
        beads=beads,
        bonds=bonds,
        bond_stats=bond_stats,
        filename="res.itp"
    )

    coords = generate_coordinates(beads, bonds, bond_stats)
    write_gro(beads, coords, "res.gro")

    draw_molecule_3d_projection("res.itp", "res.gro", "molecule.png")
    draw_3d_interactive("res.itp", "res.gro")   # ← ADD THIS

    print("generated.gro written")
    print("molecule.png written")
