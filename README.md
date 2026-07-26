# PUxH
### Polarizable Split-Charge Equilibration for Accurate Modeling of Non-Bonded Interactions
doi.org/

## Installation

This repository provides a patch for installing **PUxH** in LAMMPS, together with example bulk structures.

The patch was developed for:

```text
LAMMPS version: [LAMMPS version 23 Jun 2022 - Update 1]
```

Use a clean copy of the compatible LAMMPS source code.

## Apply the Patch

Copy `puxh.patch` to the top-level LAMMPS directory, then run:

```bash
cd lammps/src
patch --dry-run -p1 < ../puxh.patch
patch -p1 < ../puxh.patch
```

The patch will modify existing source files and create additional files for PUxH.

If the dry run reports failed hunks, reversed patches, or existing files, check that:

## Example Structures

Example bulk structures are provided for:

```text
examples/
├── water/
├── ammonia/
└── benzene/
```

These structures are intended as starting configurations.
