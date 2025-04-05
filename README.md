# tcal_pyscf

A Python tool for calculating electronic transfer integrals between molecular dimers using PySCF.

## Overview

tcal_pyscf calculates transfer integrals for HOMO, LUMO, NHOMO, and NLUMO orbitals at B3LYP/6-31G(d,p) level. It provides functionality equivalent to Gaussian16's IOp(3/33=4,5/33=3) settings as a free alternative.

This project was inspired by and builds upon [tcal](https://github.com/matsui-lab-yamagata/tcal). While tcal requires Gaussian for quantum chemical calculations, tcal_pyscf implements the same methodology using PySCF, making it a completely open-source solution accessible to everyone.

## Requirements

- Python 3.8+
- PySCF
- NumPy

## Installation

Clone the repository:
```bash
git clone https://github.com/youthesame/tcal_pyscf.git
cd tcal_pyscf
```

Install required packages:
```bash
pip install pyscf numpy
```

## Usage

### Command Line Interface

Basic transfer integral calculation:
```bash
python -m src.tcal path/to/dimer.xyz
```

Available options:

| Option        | Description                                           |
|---------------|-------------------------------------------------------|
| `-a`, `--apta`  | Perform atomic pair transfer analysis for HOMO        |
| `-l`, `--lumo`  | Perform atomic pair transfer analysis for LUMO        |
| `-o`, `--output` | Save results to CSV file                            |

Examples:

Calculate transfer integrals only:
```bash
python -m src.tcal example/anthracene/anthracene.xyz
```

Perform HOMO APTA analysis and save to CSV:
```bash
python -m src.tcal example/anthracene/anthracene.xyz -a -o
```

Perform LUMO APTA analysis:
```bash
python -m src.tcal example/anthracene/anthracene.xyz -l
```

### Input Format

The XYZ file must contain a molecular dimer where:
- First line: Number of atoms
- Second line: Comment (optional)
- Remaining lines: Atomic coordinates in XYZ format (element, x, y, z)
- First half of atoms belong to the first monomer, second half to the second monomer

Example (anthracene.xyz):
```
48
Anthracene dimer geometry
C       3.202980    1.265029    4.418408
H       2.850050    2.121173    4.628957
...
```

### Output Formats

#### Basic Transfer Integral Output
```
--------------------
 Transfer Integrals
--------------------
NLUMO      -9.517       meV
 LUMO      38.445       meV
 HOMO     -42.802       meV
NHOMO       2.596       meV
```

#### Atomic Pair Transfer Analysis (APTA)
When `-a` or `-l` options are used, the program performs atomic pair transfer analysis, showing:

1. Full matrix of atomic contributions
2. Largest contributing atomic pairs
3. Element pair contributions summary

Example:
```
---------------
 Largest Pairs
---------------
rank    pair            transfer (meV)  ratio (%)
1       4C - 37C                 9.001      -21.0
2       1C - 40C                 9.001      -21.0
...

---------------
 Element Pairs
---------------
C-C       -41.617
C-H        -1.186
```

#### CSV Output
When `-o` is specified, results are saved to CSV files with naming format:
`input_filename_apta_ORBITAL.csv`

## Python Module Usage

```python
from src.tcal import Tcal

# Initialize calculator
calculator = Tcal("path/to/dimer.xyz")

# Calculate transfer integrals
results = calculator.run_transfer_integrals()
print(results)  # {'HOMO': -42.802, 'LUMO': 38.445, ...}

# Perform atomic pair transfer analysis
apta_results = calculator.run_atomic_pair_transfer_analysis("HOMO")
```

## Theory

The transfer integral calculation follows the methodology described in:

$$t = \frac{\braket{A|F|B} - \frac{1}{2} (\epsilon_{A}+\epsilon_{B})\braket{A|B}}{1 - \braket{A|B}^2}$$

where $\braket{A|F|B}$ is the Fock matrix element between molecular orbitals of monomers A and B, $\epsilon_A$ and $\epsilon_B$ are the orbital energies, and $\braket{A|B}$ is the overlap integral.

For more details on interatomic transfer integrals and their applications, please refer to the references.

## References

1. [Koki Ozawa et al., Statistical analysis of interatomic transfer integrals for exploring high-mobility organic semiconductors, *Sci. Technol. Adv. Mater.* **2024**, *25*, 2354652.](https://doi.org/10.1080/14686996.2024.2354652)

## License

MIT License