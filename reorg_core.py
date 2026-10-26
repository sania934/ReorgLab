"""Parse Gaussian log files and compute four-point (Nelsen) reorganization energies.

Every log is classified from its own content (charge, multiplicity, job type and
geometry), never from its file name:

    lambda_h = [E+(G0) - E+(G+)] + [E0(G+) - E0(G0)]
    lambda_e = [E-(G0) - E-(G-)] + [E0(G-) - E0(G0)]

with G0 / G+ / G- the optimized neutral / cation / anion geometries.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

HARTREE_TO_EV = 27.211386245988
HARTREE_TO_KCAL = 627.5094740631
RMSD_TOL = 1e-3  # Angstrom; geometries closer than this are considered identical

_SYMBOLS = (
    "X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
    "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr "
    "Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi"
).split()

_ROLE_WORDS = {
    "anion", "cation", "neutral", "energy", "positive", "negative", "opt", "sp",
    "log", "out", "radical", "ion", "geom", "geometry", "single", "point",
}


# --------------------------------------------------------------------------- parsing
@dataclass
class LogData:
    name: str
    charge: int
    mult: int
    route: str
    is_opt: bool
    energy: float            # last SCF energy (Hartree)
    first_energy: float      # first SCF energy (Hartree), i.e. at the input geometry
    terminated: bool
    converged: bool          # "Stationary point found" (always True for single points)
    s2: float
    restricted: bool
    z: np.ndarray
    first_xyz: np.ndarray
    last_xyz: np.ndarray

    @property
    def method_key(self) -> str:
        """Level of theory with job keywords and R/U prefixes stripped."""
        parts = []
        for tok in self.route.split():
            if "/" in tok:
                func, basis = tok.split("/", 1)
                parts.append(re.sub(r"^(ro|r|u)", "", func) + "/" + basis)
        parts += re.findall(r"scrf=\S+", self.route)
        return " ".join(sorted(parts))

    @property
    def has_solvent(self) -> bool:
        return "scrf" in self.route


def _orientations(text: str, kind: str) -> list[tuple[np.ndarray, np.ndarray]]:
    pattern = kind + r" orientation:\s*\n -+\n.*?\n.*?\n -+\n(.*?)\n -+\n"
    out = []
    for block in re.findall(pattern, text, re.S):
        rows = [line.split() for line in block.strip().splitlines()]
        out.append((
            np.array([int(r[1]) for r in rows]),
            np.array([[float(v) for v in r[3:6]] for r in rows]),
        ))
    return out


def parse_log(name: str, text: str) -> LogData:
    """Parse one Gaussian log. Raises ValueError if it is not a usable Gaussian job."""
    route_match = re.search(r"^ #.*?(?=\n ---)", text, re.M | re.S)
    cm_match = re.search(r"Charge =\s*(-?\d+) Multiplicity = (\d+)", text)
    scf = re.findall(r"SCF Done:\s+E\((\S+?)\)\s+=\s+(-?\d+\.\d+)", text)
    geoms = _orientations(text, "Input") or _orientations(text, "Standard")
    if not (route_match and cm_match and scf and geoms):
        raise ValueError("not a Gaussian output with an SCF energy and geometry")

    route = " ".join(route_match.group(0).split()).lower()
    is_opt = bool(re.search(r"(?:^|[\s#])opt(?:[\s=(]|$)", route))
    s2 = re.findall(r"S\*\*2 before annihilation\s+([\d.]+)", text)
    return LogData(
        name=name,
        charge=int(cm_match.group(1)),
        mult=int(cm_match.group(2)),
        route=route,
        is_opt=is_opt,
        energy=float(scf[-1][1]),
        first_energy=float(scf[0][1]),
        terminated="Normal termination" in text,
        converged=("Stationary point found" in text) if is_opt else True,
        s2=float(s2[-1]) if s2 else 0.0,
        restricted=scf[-1][0].upper().startswith("R"),
        z=geoms[0][0],
        first_xyz=geoms[0][1],
        last_xyz=geoms[-1][1],
    )


# --------------------------------------------------------------------------- geometry
def rmsd(p: np.ndarray, q: np.ndarray) -> float:
    """Kabsch RMSD (Angstrom) for two geometries with identical atom order."""
    p, q = p - p.mean(0), q - q.mean(0)
    u, _, vt = np.linalg.svd(p.T @ q)
    rot = u @ np.diag([1, 1, np.sign(np.linalg.det(u @ vt))]) @ vt
    return float(np.sqrt(((p @ rot - q) ** 2).sum(1).mean()))


def same_geometry(a: LogData, xyz_a: np.ndarray, b: LogData, xyz_b: np.ndarray) -> bool:
    return (len(a.z) == len(b.z) and np.array_equal(a.z, b.z)
            and rmsd(xyz_a, xyz_b) < RMSD_TOL)


def formula(z: np.ndarray) -> str:
    counts = Counter(_SYMBOLS[i] for i in z)
    order = ["C", "H"] + sorted(k for k in counts if k not in ("C", "H"))
    return "".join(f"{el}{counts[el] if counts[el] > 1 else ''}" for el in order if el in counts)


# --------------------------------------------------------------------------- grouping
def group_by_molecule(logs: list[LogData]) -> list[list[LogData]]:
    """Link logs that share any geometry (start or end); each connected set is one molecule."""
    parent = list(range(len(logs)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(logs):
        for j in range(i + 1, len(logs)):
            b = logs[j]
            if len(a.z) != len(b.z) or not np.array_equal(a.z, b.z):
                continue
            if any(same_geometry(a, xa, b, xb)
                   for xa in (a.first_xyz, a.last_xyz) for xb in (b.first_xyz, b.last_xyz)):
                parent[find(j)] = find(i)

    groups: dict[int, list[LogData]] = {}
    for i, log in enumerate(logs):
        groups.setdefault(find(i), []).append(log)
    return list(groups.values())


def molecule_label(logs: list[LogData]) -> str:
    """Most common file-name token left after removing role words, else the formula."""
    tokens = Counter()
    for log in logs:
        stem = re.sub(r"\.\w+$", "", log.name)
        rest = [t for t in re.split(r"[\s_\-]+", stem) if t and t.lower() not in _ROLE_WORDS]
        if rest:
            tokens["_".join(rest)] += 1
    return tokens.most_common(1)[0][0] if tokens else formula(logs[0].z)


# --------------------------------------------------------------------------- analysis
@dataclass
class Carrier:
    kind: str                       # "hole" or "electron"
    ion_opt: LogData | None = None
    ion_at_neutral: LogData | None = None
    ion_at_neutral_energy: float | None = None
    ion_at_neutral_source: str = ""
    neutral_at_ion: LogData | None = None
    missing: list[str] = field(default_factory=list)
    rmsd_to_neutral: float | None = None
    neutral_energy: float = float("nan")

    @property
    def complete(self) -> bool:
        return not self.missing

    def energies(self) -> dict[str, float]:
        if not self.complete:
            return {}
        e0 = self.neutral_energy
        lam1 = self.ion_at_neutral_energy - self.ion_opt.energy
        lam2 = self.neutral_at_ion.energy - e0
        return {"lambda1": lam1, "lambda2": lam2, "lambda": lam1 + lam2}


@dataclass
class MoleculeResult:
    label: str
    formula: str
    logs: list[LogData]
    neutral: LogData | None
    carriers: dict[str, Carrier]
    roles: dict[str, str]           # file name -> human-readable role
    warnings: list[str]


def analyse_molecule(logs: list[LogData]) -> MoleculeResult:
    label, form = molecule_label(logs), formula(logs[0].z)
    roles = {log.name: "Unassigned" for log in logs}
    warnings: list[str] = []

    neutrals = [l for l in logs if l.charge == 0]
    opt_by_charge = {q: [l for l in logs if l.charge == q and l.is_opt] for q in (1, -1)}

    # Reference geometry G0: neutral optimization if present, else where ion optimizations start.
    neutral_opt = next((l for l in neutrals if l.is_opt), None)
    if neutral_opt is not None:
        g0_ref, g0_xyz = neutral_opt, neutral_opt.last_xyz
    else:
        ion_opt = next((o for q in (1, -1) for o in opt_by_charge[q]), None)
        g0_ref, g0_xyz = (ion_opt, ion_opt.first_xyz) if ion_opt else (None, None)

    neutral = None
    if neutral_opt is not None:
        neutral = neutral_opt
        roles[neutral.name] = "Neutral optimization (E0, G0)"
    elif g0_ref is not None:
        neutral = next((l for l in neutrals if not l.is_opt
                        and same_geometry(l, l.first_xyz, g0_ref, g0_xyz)), None)
        if neutral:
            roles[neutral.name] = "Neutral at G0 (E0)"
    if neutral is None:
        warnings.append("No neutral calculation at the neutral geometry (E0) found.")

    carriers: dict[str, Carrier] = {}
    for q, kind, ion_name in ((1, "hole", "Cation"), (-1, "electron", "Anion")):
        car = Carrier(kind=kind)
        opts = opt_by_charge[q]
        if len(opts) > 1:
            warnings.append(f"{len(opts)} {ion_name.lower()} optimizations found; using {opts[0].name}.")
        car.ion_opt = opts[0] if opts else None
        if neutral is not None:
            car.neutral_energy = neutral.energy
        if car.ion_opt is None:
            car.missing.append(f"{ion_name} optimization")
        else:
            roles[car.ion_opt.name] = f"{ion_name} optimization (E{'+' if q > 0 else '-'} at G{'+' if q > 0 else '-'})"
            if g0_xyz is not None:
                car.rmsd_to_neutral = rmsd(car.ion_opt.last_xyz, g0_xyz)
            # ion at neutral geometry: explicit single point, else first step of the optimization
            sp = next((l for l in logs if l.charge == q and not l.is_opt and g0_xyz is not None
                       and same_geometry(l, l.first_xyz, g0_ref, g0_xyz)), None)
            if sp is not None:
                car.ion_at_neutral, car.ion_at_neutral_energy = sp, sp.energy
                car.ion_at_neutral_source = sp.name
                roles[sp.name] = f"{ion_name} at neutral geometry (E{'+' if q > 0 else '-'} at G0)"
            elif g0_xyz is not None and same_geometry(car.ion_opt, car.ion_opt.first_xyz, g0_ref, g0_xyz):
                car.ion_at_neutral_energy = car.ion_opt.first_energy
                car.ion_at_neutral_source = f"{car.ion_opt.name} (first optimization step)"
                warnings.append(f"{ion_name} at neutral geometry taken from the first step of {car.ion_opt.name}.")
            else:
                car.missing.append(f"{ion_name} single point at neutral geometry")
            # neutral at ion geometry
            sp = next((l for l in neutrals if not l.is_opt
                       and same_geometry(l, l.first_xyz, car.ion_opt, car.ion_opt.last_xyz)), None)
            if sp is not None:
                car.neutral_at_ion = sp
                roles[sp.name] = f"Neutral at {ion_name.lower()} geometry (E0 at G{'+' if q > 0 else '-'})"
            else:
                car.missing.append(f"Neutral single point at {ion_name.lower()} geometry")
            # redundant ion single points at the ion's own geometry
            for l in logs:
                if (l.charge == q and not l.is_opt and roles[l.name] == "Unassigned"
                        and same_geometry(l, l.first_xyz, car.ion_opt, car.ion_opt.last_xyz)):
                    roles[l.name] = f"{ion_name} at its own geometry (duplicate of optimization)"
        if neutral is None:
            car.missing.append("Neutral at neutral geometry (E0)")
        carriers[kind] = car

    # label leftovers that still sit at the neutral geometry
    if g0_xyz is not None:
        for l in logs:
            if roles[l.name] != "Unassigned" or l.is_opt or not same_geometry(l, l.first_xyz, g0_ref, g0_xyz):
                continue
            if l.charge == 0:
                roles[l.name] = "Neutral at G0 (duplicate of neutral optimization)"
            elif l.charge in (1, -1):
                ion = "Cation" if l.charge > 0 else "Anion"
                roles[l.name] = f"{ion} at neutral geometry (no {ion.lower()} optimization found)"

    # quality checks
    for l in logs:
        if not l.terminated:
            warnings.append(f"{l.name}: no 'Normal termination'.")
        if l.is_opt and not l.converged:
            warnings.append(f"{l.name}: optimization has no 'Stationary point found'.")
        if l.mult == 2 and l.s2 > 0.8:
            warnings.append(f"{l.name}: spin contamination, <S**2> = {l.s2:.3f}.")
    if len({l.method_key for l in logs}) > 1:
        warnings.append("Different levels of theory/solvent across files: "
                        + "; ".join(sorted({f"[{l.method_key}]" for l in logs})) + ". Results are not comparable.")
    return MoleculeResult(label, form, logs, neutral, carriers, roles, warnings)


def analyse(logs: list[LogData]) -> list[MoleculeResult]:
    results = [analyse_molecule(g) for g in group_by_molecule(logs)]
    return sorted(results, key=lambda r: r.label)
