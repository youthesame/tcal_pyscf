import argparse
import csv
import re
import warnings
from datetime import datetime
from pathlib import Path
from time import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pyscf import dft, gto

warnings.filterwarnings("ignore", message="Since PySCF-2.3, B3LYP .* functional in Gaussian.*")


def main():
    """Execute tcal for command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="XYZ file name", type=str)
    parser.add_argument("-a", "--apta", help="perform atomic pair transfer analysis", action="store_true")
    parser.add_argument("-l", "--lumo", help="perform atomic pair transfer analysis of LUMO", action="store_true")
    parser.add_argument(
        "-o", "--output", action="store_true", help="output csv file on the result of apta and transfer integrals"
    )
    args = parser.parse_args()

    print("-----------------------------")
    print(" tcal-pyscf 1.0 (2025.04.05) ")
    print("-----------------------------")
    print(f"\nInput File Name: {args.file}")
    print_timestamp()
    before = time()
    print()

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: File {args.file} does not exist")
        exit(1)

    try:
        calculator = Tcal(file_path)

        # Determine which analysis to run
        if args.apta:
            analyze_orbital = "HOMO"
            run_apta = True
        elif args.lumo:
            analyze_orbital = "LUMO"
            run_apta = True
        else:
            run_apta = False

        if run_apta:
            # Execute atomic pair transfer analysis
            apta = calculator.run_atomic_pair_transfer_analysis(analyze_orbital)

            # Output CSV file
            if args.output:
                base_path = Path(args.file).with_suffix("")
                apta_filepath = f"{base_path}_apta_{analyze_orbital}.csv"
                output_csv(apta_filepath, apta)
                print(f"\nAPTA results saved to: {apta_filepath}")
        else:
            # Default is to calculate only transfer integrals
            results = calculator.run_transfer_integrals()

            # Output CSV file if requested
            if args.output:
                base_path = Path(args.file).with_suffix("")
                ti_filepath = f"{base_path}_transfer_integrals.csv"
                # Convert dict to list for CSV output
                ti_data = [["Orbital", "Value (meV)"]]
                for orbital in ["NLUMO", "LUMO", "HOMO", "NHOMO"]:
                    if orbital in results:
                        ti_data.append([orbital, f"{results[orbital]:.3f}"])
                output_csv(ti_filepath, ti_data)
                print(f"\nTransfer integral results saved to: {ti_filepath}")

            print()
            print_timestamp()
            after = time()
            print(f"Elapsed Time: {(after - before) * 1000:.0f} ms")
            exit(0)

        print()
    except Exception:
        import traceback

        print(traceback.format_exc().strip())

    print_timestamp()
    after = time()
    print(f"Elapsed Time: {(after - before) * 1000:.0f} ms")


def print_timestamp():
    """Print timestamp."""
    month = {
        1: "Jan",
        2: "Feb",
        3: "Mar",
        4: "Apr",
        5: "May",
        6: "Jun",
        7: "Jul",
        8: "Aug",
        9: "Sep",
        10: "Oct",
        11: "Nov",
        12: "Dec",
    }
    dt_now = datetime.now()
    print(f"Timestamp: {dt_now.strftime('%a')} {month[dt_now.month]} {dt_now.strftime('%d %H:%M:%S %Y')}")


def output_csv(file_name, array):
    """Output csv file of array.

    Parameters
    ----------
    file_name : str
        File name including extension.
    array : array_like
        Array to create csv file.
    """
    with open(file_name, "w", encoding="UTF-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(array)


class Tcal:
    """
    Calculate transfer integrals between molecular orbitals using PySCF.

    This class provides methods to read XYZ files, set up PySCF calculations,
    and compute transfer integrals between frontier molecular orbitals.

    Attributes
    ----------
    EV : float
        Conversion factor from atomic units to meV
    """

    # Conversion factor from atomic units to meV
    EV: float = 4.35974417e-18 / 1.60217653e-19 * 1000.0

    def __init__(self, file_path: Union[str, Path]):
        """
        Initialize the calculator with an XYZ file path.

        Parameters
        ----------
        file_path : Union[str, Path]
            Path to the XYZ file containing the dimer structure
        """
        self.file_path = Path(file_path)
        self.atoms1: List[List[Any]] = []
        self.atoms2: List[List[Any]] = []
        self.mf1: Optional[dft.RKS] = None
        self.mf2: Optional[dft.RKS] = None
        self.dimer: Optional[dft.RKS] = None
        self.monomer_nbasis: int = 0
        self.dimer_nbasis: int = 0
        self.mf1_occ: int = 0
        self.mf2_occ: int = 0
        self.ovlp: Optional[np.ndarray] = None
        self.fock: Optional[np.ndarray] = None

    @staticmethod
    def print_matrix(matrix):
        """Print matrix.

        Parameters
        ----------
        matrix : array_like
        """
        for i, row in enumerate(matrix):
            for j, cell in enumerate(row[:-1]):
                if i == 0 or j == 0:
                    print(f"{cell:^9}", end="\t")
                else:
                    print(f"{cell:>9}", end="\t")
            if i == 0:
                print(f"{row[-1]:^9}")
            else:
                print(f"{row[-1]:>9}")

    def xyz_to_pyscf(self) -> Tuple[List[List[Any]], List[List[Any]]]:
        """
        Read XYZ file and convert to PySCF input format for two monomers.

        Returns
        -------
        Tuple[List[List[Any]], List[List[Any]]]
            Two lists containing atomic coordinates for each monomer
            in the format required by PySCF
        """
        with open(self.file_path, "r") as f:
            lines = f.readlines()

        natoms = int(lines[0])
        atoms = []
        for line in lines[2 : 2 + natoms]:
            symbol, x, y, z = line.split()
            atoms.append([symbol, float(x), float(y), float(z)])

        # Split atoms into two monomers
        monomer1 = atoms[: len(atoms) // 2]
        monomer2 = atoms[len(atoms) // 2 :]

        atoms1 = [[atom[0]] + atom[1:] for atom in monomer1]
        atoms2 = [[atom[0]] + atom[1:] for atom in monomer2]

        return atoms1, atoms2

    def create_mf(self, atoms: List[List[Any]]) -> dft.RKS:
        """
        Create and configure a PySCF mean-field object for DFT calculations.

        Parameters
        ----------
        atoms : List[List[Any]]
            Atomic coordinates in PySCF format

        Returns
        -------
        dft.RKS
            Configured restricted Kohn-Sham DFT object
        """
        mf = dft.RKS(gto.M(atom=atoms, basis="6-31g(d,p)", symmetry=False, cart=True))
        mf.xc = "b3lyp"
        mf.conv_tol = 1e-7
        mf.max_cycle = 128

        return mf

    def cal_transfer_integral(self, bra: np.ndarray, overlap: np.ndarray, fock: np.ndarray, ket: np.ndarray) -> float:
        """
        Calculate transfer integral between two molecular orbitals.

        Parameters
        ----------
        bra : np.ndarray
            First molecular orbital vector
        overlap : np.ndarray
            Overlap matrix
        fock : np.ndarray
            Fock matrix
        ket : np.ndarray
            Second molecular orbital vector

        Returns
        -------
        float
            Transfer integral value in meV
        """
        s11 = bra @ overlap @ bra
        s22 = ket @ overlap @ ket
        s12 = bra @ overlap @ ket
        f11 = bra @ fock @ bra
        f22 = ket @ fock @ ket
        f12 = bra @ fock @ ket

        if abs(s11 - 1) > 1e-2:
            print(f"WARNING! Self overlap is not unity: S11 = {s11}")
        if abs(s22 - 1) > 1e-2:
            print(f"WARNING! Self overlap is not unity: S22 = {s22}")

        transfer = (f12 - 0.5 * (f11 + f22) * s12) / (1 - s12**2)
        return transfer * self.EV

    def cal_atomic_pair_transfer_analysis(self, analyze_orbital: str = "HOMO") -> np.ndarray:
        """
        Calculate atomic pair transfer analysis for a specific orbital.

        Parameters
        ----------
        analyze_orbital : str
            Orbital to analyze (HOMO or LUMO)

        Returns
        -------
        np.ndarray
            Matrix of atomic pair transfer values
        """
        self.atom_index = []
        self.atom_symbol = []
        for ao_label in self.dimer.mol.ao_labels():
            atom_idx, atom_symbol, _ = ao_label.split()
            self.atom_index.append(int(atom_idx))
            self.atom_symbol.append(atom_symbol)

        # Initialize extended MO vectors in dimer basis
        mo1_ext = np.zeros(self.dimer_nbasis)
        mo2_ext = np.zeros(self.dimer_nbasis)

        # Fill in the monomer MO coefficients based on the orbital to analyze
        if analyze_orbital.upper() == "LUMO":
            mo1_ext[: self.monomer_nbasis] = self.mf1.mo_coeff[:, self.mf1_occ]
            mo2_ext[self.monomer_nbasis :] = self.mf2.mo_coeff[:, self.mf2_occ]
        else:  # Default to HOMO
            mo1_ext[: self.monomer_nbasis] = self.mf1.mo_coeff[:, self.mf1_occ - 1]
            mo2_ext[self.monomer_nbasis :] = self.mf2.mo_coeff[:, self.mf2_occ - 1]

        # Calculate Fock and overlap terms
        f11 = mo1_ext @ self.fock @ mo1_ext
        f22 = mo2_ext @ self.fock @ mo2_ext
        s12 = mo1_ext @ self.ovlp @ mo2_ext

        # Calculate atomic pair transfers
        atom_num = max(self.atom_index) + 1
        a_transfer = np.zeros((atom_num, atom_num))

        for i in range(self.dimer_nbasis):
            for j in range(self.dimer_nbasis):
                orb_transfer = (
                    mo1_ext[i] * mo2_ext[j] * (self.fock[i][j] - 0.5 * (f11 + f22) * self.ovlp[i][j]) / (1 - s12**2)
                )
                a_transfer[self.atom_index[i]][self.atom_index[j]] += orb_transfer

        return a_transfer

    def setup(self) -> None:
        """
        Set up the calculation by reading the XYZ file and creating mean-field objects.
        """
        # Read XYZ file
        self.atoms1, self.atoms2 = self.xyz_to_pyscf()

        # Create mean-field objects
        self.mf1 = self.create_mf(self.atoms1)
        self.mf2 = self.create_mf(self.atoms2)
        self.dimer = self.create_mf(self.atoms1 + self.atoms2)

        # Get necessary parameters
        self.monomer_nbasis = self.mf1.mol.nao
        self.dimer_nbasis = self.dimer.mol.nao
        self.mf1_occ = self.mf1.mol.nelectron // 2
        self.mf2_occ = self.mf2.mol.nelectron // 2

    def run_scf(self) -> None:
        """
        Run self-consistent field calculations for monomers and dimer.
        """
        if self.mf1 is None or self.mf2 is None or self.dimer is None:
            raise ValueError("Setup must be called before running SCF calculations")

        # Run SCF calculations
        self.mf1.kernel()
        self.mf2.kernel()
        self.dimer.kernel()

        # Get overlap and Fock matrices
        self.ovlp = self.dimer.get_ovlp()
        self.fock = self.dimer.get_fock()

    def calculate_transfer_integrals(self) -> Dict[str, float]:
        """
        Calculate transfer integrals for frontier orbitals.

        Returns
        -------
        Dict[str, float]
            Dictionary mapping orbital names to transfer integral values in meV
        """
        results = {}

        # Calculate transfer integrals for frontier orbitals
        for orb_idx in range(1, -3, -1):
            # Initialize extended MO vectors in dimer basis
            mo1_ext = np.zeros(self.dimer_nbasis)
            mo2_ext = np.zeros(self.dimer_nbasis)

            # Fill in the monomer MO coefficients
            mo1_ext[: self.monomer_nbasis] = self.mf1.mo_coeff[:, self.mf1_occ + orb_idx]
            mo2_ext[self.monomer_nbasis :] = self.mf2.mo_coeff[:, self.mf2_occ + orb_idx]

            # Calculate transfer integral
            transfer = self.cal_transfer_integral(mo1_ext, self.ovlp, self.fock, mo2_ext)

            # Store results with orbital labels
            if orb_idx == 1:
                results["NLUMO"] = transfer
            elif orb_idx == 0:
                results["LUMO"] = transfer
            elif orb_idx == -1:
                results["HOMO"] = transfer
            elif orb_idx == -2:
                results["NHOMO"] = transfer

        return results

    def print_results(self, results: Dict[str, float], message: str = "Transfer Integrals") -> None:
        """
        Print the transfer integral results.

        Parameters
        ----------
        results : Dict[str, float]
            Dictionary mapping orbital names to transfer integral values in meV
        message : str
            Header message for output
        """
        print()
        print("-" * (len(message) + 2))
        print(f" {message} ")
        print("-" * (len(message) + 2))

        # Print ordered results
        for orbital in ["NLUMO", "LUMO", "HOMO", "NHOMO"]:
            if orbital in results:
                label = f" {orbital}" if orbital in ["LUMO", "HOMO"] else orbital
                print(f"{label}\t{results[orbital]:9.3f}\tmeV")

    def print_apta(self, a_transfer: np.ndarray, message: str = "Atomic Pair Transfer Analysis") -> List[List[str]]:
        """
        Print atomic pair transfer analysis results.

        Parameters
        ----------
        a_transfer : np.ndarray
            Matrix of atomic pair transfer values
        message : str
            Header message for the output

        Returns
        -------
        List[List[str]]
            Table of APTA results for potential CSV output
        """
        n_atoms = max(self.atom_index) + 1
        n_atoms1 = len(self.atoms1)

        # Create list of atom labels
        labels = [""] * n_atoms
        for i, idx in enumerate(self.atom_index):
            labels[idx] = self.atom_symbol[i]

        # Calculate sums
        col_sum = np.sum(a_transfer, axis=1)
        row_sum = np.sum(a_transfer, axis=0)
        total_sum = np.sum(a_transfer)

        # Print header
        print()
        print("-" * (len(message) + 2))
        print(f" {message} ")
        print("-" * (len(message) + 2))

        # Create table for printing
        apta = []
        tmp_list = ["atom"]
        for i in range(n_atoms1, n_atoms):
            tmp_list.append(f"{i + 1}{labels[i]}")
        tmp_list.append("sum")
        apta.append(tmp_list)

        for i in range(n_atoms1):
            tmp_list = []
            tmp_list.append(f"{i + 1}{labels[i]}")
            for j in range(n_atoms1, n_atoms):
                tmp_list.append(f"{a_transfer[i][j] * Tcal.EV:.3f}")
            tmp_list.append(f"{col_sum[i] * Tcal.EV:.3f}")
            apta.append(tmp_list)

        tmp_list = ["sum"]
        for j in range(n_atoms1, n_atoms):
            tmp_list.append(f"{row_sum[j] * Tcal.EV:.3f}")
        tmp_list.append(f"{total_sum * Tcal.EV:.3f}")
        apta.append(tmp_list)

        self.print_matrix(apta)

        return apta

    def run_transfer_integrals(self) -> Dict[str, float]:
        """
        Run the complete transfer integral calculation workflow.

        Returns
        -------
        Dict[str, float]
            Dictionary mapping orbital names to transfer integral values in meV
        """
        # Setup and run calculations
        self.setup()
        self.run_scf()
        results = self.calculate_transfer_integrals()

        # Print results
        self.print_results(results)

        return results

    def run_atomic_pair_transfer_analysis(self, analyze_orbital: str = "HOMO") -> List[List[str]]:
        """
        Run the complete atomic pair transfer analysis workflow.

        Parameters
        ----------
        analyze_orbital : str
            Orbital to analyze (HOMO or LUMO)

        Returns
        -------
        List[List[str]]
            Table of APTA results
        """
        self.setup()
        self.run_scf()

        results = self.calculate_transfer_integrals()
        self.print_results(results)

        apta = self.cal_atomic_pair_transfer_analysis(analyze_orbital)
        apta_table = self.print_apta(apta)

        pair_analysis = PairAnalysis(apta_table)
        pair_analysis.print_largest_pairs()
        pair_analysis.print_element_pairs()

        return apta_table


class PairAnalysis:
    """Analyze atomic pair transfer integrals."""

    def __init__(self, apta):
        """Inits PairAnalysisClass.

        Parameters
        ----------
        apta : list
            List of atomic pair transfer integrals including labels.
        """
        self._labels = []
        self._a_transfer = []

        for row in apta[1:-1]:
            symbol = re.sub("[0-9]+", "", row[0])
            self._labels.append(symbol)
            self._a_transfer.append(row[1:-1])

        label = apta[0][1:-1]
        label = [re.sub("[0-9]+", "", x) for x in label]

        self._labels.extend(label)
        self._a_transfer = np.array(self._a_transfer, dtype=np.float64)
        self.n_atoms1 = self._a_transfer.shape[0]
        self.n_atoms2 = self._a_transfer.shape[1]

    def print_largest_pairs(self):
        """Print largest pairs."""
        transfer = np.sum(self._a_transfer)
        a_transfer_flat = self._a_transfer.flatten()
        sorted_index = np.argsort(a_transfer_flat)
        print()
        print("---------------")
        print(" Largest Pairs ")
        print("---------------")
        print(f"rank\tpair{' ' * 9}\ttransfer (meV)\tratio (%)")

        rank_list = np.arange(1, len(sorted_index) + 1)
        if len(sorted_index) <= 20:
            print_index = sorted_index
            ranks = rank_list
        else:
            print_index = np.hstack([sorted_index[:10], sorted_index[-10:]])
            ranks = np.hstack([rank_list[:10], rank_list[-10:]])

        for i, a_i in enumerate(reversed(print_index)):
            row_i = a_i // self.n_atoms2
            col_i = a_i % self.n_atoms2
            pair = f"{row_i + 1}{self._labels[row_i]}" + " - " + f"{col_i + self.n_atoms2 + 1}{self._labels[col_i]}"
            ratio = np.divide(a_transfer_flat[a_i], transfer, out=np.array(0.0), where=(transfer != 0)) * 100
            print(f"{ranks[i]:<4}\t{pair:<13}\t{a_transfer_flat[a_i]:>14}\t{ratio:>9.1f}")

    def print_element_pairs(self):
        """Print element pairs."""
        element_pair = {
            "H-I": 0.0,
            "C-C": 0.0,
            "C-H": 0.0,
            "C-S": 0.0,
            "C-Se": 0.0,
            "C-I": 0.0,
            "S-S": 0.0,
            "Se-Se": 0.0,
            "N-S": 0.0,
            "I-S": 0.0,
            "I-I": 0.0,
        }
        keys = element_pair.keys()

        for i in range(self.n_atoms1):
            sym1 = self._labels[i]
            for j in range(self.n_atoms2):
                sym2 = self._labels[self.n_atoms1 + j]
                key = "-".join(sorted([sym1, sym2]))
                if key in keys:
                    element_pair[key] += self._a_transfer[i][j]

        print()
        print("---------------")
        print(" Element Pairs ")
        print("---------------")
        for k, value in element_pair.items():
            if value != 0:
                print(f"{k:<5}\t{value:>9.3f}")


if __name__ == "__main__":
    main()
