# ReorgLab

Reorganization energies of hole- and electron-transport materials from Gaussian log files.

Upload your optimization and single-point logs together. Each file is recognised from its **contents**
(charge, multiplicity, job type, geometry), not its file name, and λh and λe are computed with the
four-point (Nelsen) method.

```
λh = [E+(G0) − E+(G+)] + [E0(G+) − E0(G0)]
λe = [E−(G0) − E−(G−)] + [E0(G−) − E0(G0)]
```

## Features
- Automatic recognition and grouping of logs by molecule (`.log`, `.out` or a `.zip`)
- Calculation modes: holes & electrons, holes only, electrons only
- Summary and detailed views, units in eV, meV or kcal/mol, CSV export
- Checks for failed jobs, unconverged optimizations, spin contamination and mixed levels of theory
- Light and dark themes

## Required files per molecule
Neutral optimization, cation optimization, anion optimization, and single points of the ion at the
neutral geometry and of the neutral at each ion geometry. If the ion single point at the neutral geometry
is absent, the first step of the ion optimization is used. Missing pieces are reported per molecule.

## Run
```
pip install -r requirements.txt
python -m streamlit run app.py
```

`reorg_core.py` can also be used on its own (e.g. in a notebook): `parse_log(name, text)` and `analyse(logs)`.

## Notes
Energies are the last `SCF Done` value of each log. All files for a molecule should share one level of
theory (functional, basis set, solvent model).

## Author
Sania Ismaeel
