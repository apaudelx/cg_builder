import argparse
import random
from collections import defaultdict


# -------------------------
# READ LIBRARY DATA
# -------------------------
def read_allowed_bonds(filename="data/allowed_bonds.txt"):
    allowed = set()
    bead_types = set()

    with open(filename, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) != 2:
                continue
            a, b = parts
            allowed.add(frozenset((a, b)))
            bead_types.add(a)
            bead_types.add(b)

    return allowed, sorted(bead_types)


def read_max_degree(filename="data/max_degree.txt"):
    max_degree = {}
    with open(filename, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) != 2:
                continue
            bead, deg = parts
            max_degree[bead] = int(deg)
    return max_degree


# -------------------------
# CORE CHECKS
# -------------------------
def can_add_bond(i, j, ti, tj, degree, max_degree, allowed_bonds, bonds_set, original_beads):
    if i == j:
        return False

    if (min(i, j), max(i, j)) in bonds_set:
        return False

    # allow flexible bonding for newly introduced beads
    if frozenset((ti, tj)) not in allowed_bonds:
        if ti not in original_beads or tj not in original_beads:
            pass
        else:
            return False

    if degree[i] >= max_degree.get(ti, 0):
        return False

    if degree[j] >= max_degree.get(tj, 0):
        return False

    return True


def parse_bead_weights(s):
    weights = {}
    total = 0

    for item in s.split(","):
        t, n = item.split(":")
        n = int(n)
        weights[t] = n
        total += n

    return weights, total


# -------------------------
# RING GENERATION
# -------------------------
def generate_one_ring(ring_size, allowed_bonds, bead_pool, max_degree, max_tries):
    for _ in range(max_tries):
        types = [random.choice(bead_pool) for _ in range(ring_size)]

        if any(max_degree.get(t, 0) < 2 for t in types):
            continue

        ok = True
        for i in range(ring_size):
            a = types[i]
            b = types[(i + 1) % ring_size]
            if frozenset((a, b)) not in allowed_bonds:
                ok = False
                break

        if ok:
            return types

    return None


# -------------------------
# BUILD RINGS
# -------------------------
def build_rings(n3, n4, n5, n6, allowed_bonds, bead_pool, max_degree, max_tries):
    ring_specs = [(3, n3), (4, n4), (5, n5), (6, n6)]

    beads = {}
    bonds = []
    degree = defaultdict(int)
    bonds_set = set()

    current_id = 1
    rings = []

    for size, count in ring_specs:
        for _ in range(count):
            types = generate_one_ring(size, allowed_bonds, bead_pool, max_degree, max_tries)

            if types is None:
                raise RuntimeError(f"failed to generate {size}-ring")

            ids = []
            for t in types:
                beads[current_id] = t
                ids.append(current_id)
                current_id += 1

            for i in range(size):
                u = ids[i]
                v = ids[(i + 1) % size]

                bonds.append((u, v))
                bonds_set.add((min(u, v), max(u, v)))
                degree[u] += 1
                degree[v] += 1

            rings.append(ids)

    return beads, bonds, degree, bonds_set, rings


# -------------------------
# CONNECT RINGS
# -------------------------
def connect_rings(rings, beads, bonds, degree, bonds_set, allowed_bonds, max_degree, original_beads):
    for i in range(len(rings) - 1):
        r1 = rings[i]
        r2 = rings[i + 1]

        success = False

        for _ in range(5000):
            u = random.choice(r1)
            v = random.choice(r2)

            if can_add_bond(u, v, beads[u], beads[v], degree, max_degree,
                            allowed_bonds, bonds_set, original_beads):
                bonds.append((u, v))
                bonds_set.add((min(u, v), max(u, v)))
                degree[u] += 1
                degree[v] += 1
                success = True
                break

        if not success:
            raise RuntimeError("failed to connect rings")


# -------------------------
# ADD EXTRA BEADS
# -------------------------
def add_extra_beads(target_N, beads, bonds, degree, bonds_set,
                   allowed_bonds, bead_pool, max_degree,
                   original_beads, forced_beads=None):

# ---- FIX: initialize if empty ----
    if not beads:
        first_type = random.choice(bead_pool)
        beads[1] = first_type
        degree[1] = 0
        bonds_set.clear()
    
    next_id = max(beads) + 1

    next_id = max(beads) + 1 if beads else 1


    while len(beads) < target_N:
        success = False

        for _ in range(10000):
            attach = random.choice(list(beads.keys()))

            if forced_beads:
                new_type = forced_beads[0]
            else:
                new_type = random.choice(bead_pool)

            if max_degree.get(new_type, 0) < 1:
                continue

            if can_add_bond(
                attach, next_id,
                beads[attach], new_type,
                degree, max_degree, allowed_bonds, bonds_set, original_beads
            ):
                beads[next_id] = new_type
                bonds.append((attach, next_id))
                bonds_set.add((min(attach, next_id), max(attach, next_id)))
                degree[attach] += 1
                degree[next_id] += 1
                next_id += 1

                if forced_beads:
                    forced_beads.pop(0)

                success = True
                break

        if not success:
            raise RuntimeError("failed to add extra bead")


# -------------------------
# WRITE ITP
# -------------------------
def write_itp(beads, bonds, filename):
    with open(filename, "w") as f:
        f.write("[ moleculetype ]\nGENMOL 1\n\n")

        f.write("[ atoms ]\n")
        for i in sorted(beads):
            f.write(f"{i:5d} {beads[i]:6s} 1 MOL {beads[i]:6s} {i:5d} 0.0\n")

        f.write("\n[ bonds ]\n")
        for i, j in bonds:
            f.write(f"{i:5d} {j:5d} 1\n")


# -------------------------
# MAIN
# -------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--nbeads", type=int, required=True)
    parser.add_argument("--bead-weights", type=str, default=None)
    parser.add_argument("--n3", type=int, default=0)
    parser.add_argument("--n4", type=int, default=0)
    parser.add_argument("--n5", type=int, default=0)
    parser.add_argument("--n6", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-tries", type=int, default=10000)
    parser.add_argument("--output", default="generated.itp")

    args = parser.parse_args()

    allowed_bonds, bead_pool = read_allowed_bonds()
    max_degree = read_max_degree()

    original_beads = set(bead_pool)
    forced_beads = []

    if args.bead_weights:
        weights, total_fixed = parse_bead_weights(args.bead_weights)

        for t in weights:
            if t not in bead_pool:
                bead_pool.append(t)
                max_degree[t] = 1

        if total_fixed > args.nbeads:
            raise ValueError("fixed bead count exceeds total nbeads")

        for t, n in weights.items():
            forced_beads.extend([t] * n)

    if args.seed is not None:
        random.seed(args.seed)

    required = 3*args.n3 + 4*args.n4 + 5*args.n5 + 6*args.n6
    if required > args.nbeads:
        raise ValueError("ring beads exceed total nbeads")

    beads, bonds, degree, bonds_set, rings = build_rings(
        args.n3, args.n4, args.n5, args.n6,
        allowed_bonds, bead_pool, max_degree, args.max_tries
    )

    if len(rings) > 1:
        connect_rings(rings, beads, bonds, degree, bonds_set,
                      allowed_bonds, max_degree, original_beads)

    add_extra_beads(
        args.nbeads,
        beads, bonds, degree, bonds_set,
        allowed_bonds, bead_pool, max_degree,
        original_beads,
        forced_beads=forced_beads
    )

    write_itp(beads, bonds, args.output)

    print("done:", args.output)


if __name__ == "__main__":
    main()
