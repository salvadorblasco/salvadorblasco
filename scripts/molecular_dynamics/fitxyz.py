#!/usr/bin/env python3
"""
fitxyz.py - Utility to manipulate XYZ molecular geometry files.

Features:
- Load, view, transform XYZ files
- Align vectors, center at COM or midpoint
- Compute inertia tensor and principal axes
- Visualize with VMD
- Interactive CLI + batch mode

Author: Salvador Blasco <salvador.blasco@uv.es>
License: MIT
"""

from __future__ import annotations

import cmd
import io
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence, Tuple, List, Optional
from dataclasses import dataclass

import periodictable as pt
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation


FArray = NDArray[np.float64]
IArray = NDArray[np.int64]


@dataclass
class XYZdata:
    """Container for atomic symbols and 3D coordinates."""
    elements: List[pt.core.Element]
    coordinates: FArray  # Shape: (N, 3)

    def __post_init__(self) -> None:
        """Validate input shapes."""
        if self.coordinates.shape[1:] != (3,):
            raise ValueError(f"coordinates must be (N, 3), got {self.coordinates.shape}")
        if len(self.elements) != self.coordinates.shape[0]:
            raise ValueError("Number of elements must match number of coordinate rows")

    @property
    def natoms(self) -> int:
        """Number of atoms."""
        return len(self.elements)

    @property
    def masses(self) -> FArray:
        """Atomic masses as 1D array."""
        return np.array([elm.mass for elm in self.elements], dtype=np.float64)

    @property
    def symbols(self) -> List[str]:
        """List of atomic symbols."""
        return [elm.symbol for elm in self.elements]

    # ============================= I/O =============================

    @classmethod
    def from_file(cls, filename: str | Path) -> "XYZdata":
        """Load XYZ data from file."""
        path = Path(filename)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        with path.open('r') as f:
            lines = f.readlines()

        if len(lines) < 2:
            raise ValueError("XYZ file must have at least 2 lines")

        try:
            n_atoms = int(lines[0].strip())
        except ValueError:
            raise ValueError("First line must be integer (number of atoms)")

        if len(lines) < n_atoms + 2:
            raise ValueError(f"Expected {n_atoms} atom lines, got {len(lines) - 2}")

        elements: List[pt.core.Element] = []
        coords: List[List[float]] = []

        for line in lines[2:2 + n_atoms]:
            parts = line.split()
            if len(parts) < 4:
                continue  # skip malformed
            symbol = parts[0].capitalize()
            try:
                elements.append(pt.elements.symbol(symbol))
            except ValueError:
                raise ValueError(f"Unknown element symbol: {symbol}")
            try:
                x, y, z = map(float, parts[1:4])
            except ValueError:
                raise ValueError(f"Invalid coordinates in line: {line.strip()}")
            coords.append([x, y, z])

        return cls(elements, np.array(coords, dtype=np.float64))

    def to_string(self) -> str:
        """Return XYZ format as string."""
        buffer = io.StringIO()
        buffer.write(f"{self.natoms}\n")
        buffer.write("Transformed molecule\n")
        for sym, (x, y, z) in zip(self.symbols, self.coordinates):
            buffer.write(f"{sym:>3}  {x:12.6f} {y:12.6f} {z:12.6f}\n")
        return buffer.getvalue()

    def write_file(self, filename: str | Path) -> None:
        """Write XYZ data to file."""
        Path(filename).write_text(self.to_string())

    # ========================= Geometry =========================

    def center_of_mass(self) -> FArray:
        """Calculate center of mass."""
        if self.natoms == 0:
            return np.zeros(3)
        return np.average(self.coordinates, axis=0, weights=self.masses)

    def translate(self, vector: Sequence[float]) -> None:
        """Translate all atoms by vector."""
        v = np.asarray(vector, dtype=np.float64)
        if v.shape != (3,):
            raise ValueError("Translation vector must be 3D")
        self.coordinates -= v

    def center_at_origin(self, center: Optional[Sequence[float]] = None) -> None:
        """Move specified center to origin."""
        if center is None:
            center = self.center_of_mass()
        self.translate(center)

    def inertia_tensor(self) -> FArray:
        """Compute inertia tensor about COM."""
        com = self.center_of_mass()
        r = self.coordinates - com
        r2 = np.sum(r * r, axis=1)

        masses = self.masses
        diag = np.einsum('i,i->', masses, r2)
        I = np.eye(3) * diag - np.einsum('ni,nj,n->ij', r, r, masses)
        return I

    @staticmethod
    def diagonalize_inertia(I: FArray) -> Tuple[FArray, FArray]:
        """Diagonalize inertia tensor → eigenvalues, eigenvectors."""
        eigvals, eigvecs = np.linalg.eig(I)
        idx = np.argsort(eigvals)
        return eigvals[idx], eigvecs[:, idx]

    def principal_axes(self) -> Tuple[FArray, FArray]:
        """Return sorted eigenvalues and normalized eigenvectors."""
        return self.diagonalize_inertia(self.inertia_tensor())

    # ========================= Alignment =========================

    def align_vector(self, v_old: Sequence[float], v_new: Sequence[float],
                     center: Optional[Sequence[float]] = None) -> FArray:
        """
        Rotate molecule so that v_old aligns with v_new.

        Parameters
        ----------
        v_old : array-like, shape (3,)
            Vector in current frame.
        v_new : array-like, shape (3,)
            Target direction.
        center : array-like, shape (3,), optional
            Rotation center (default: origin).

        Returns
        -------
        rotation_matrix : np.ndarray, shape (3,3)
            Applied rotation matrix.
        """
        v_old = np.asarray(v_old, dtype=np.float64).ravel()
        v_new = np.asarray(v_new, dtype=np.float64).ravel()

        if v_old.shape != (3,) or v_new.shape != (3,):
            raise ValueError("Vectors must be 3D")

        norm_old = np.linalg.norm(v_old)
        norm_new = np.linalg.norm(v_new)

        if norm_old == 0 or norm_new == 0:
            raise ValueError("Zero vector cannot be aligned")

        u_old = v_old / norm_old
        u_new = v_new / norm_new

        if np.allclose(u_old, u_new):
            return np.eye(3)
        if np.allclose(u_old, -u_new):
            # 180° rotation
            perp = np.array([1, 0, 0]) if abs(u_old[0]) < 0.9 else np.array([0, 0, 1])
            axis = np.cross(u_old, perp)
            axis /= np.linalg.norm(axis)
            R = Rotation.from_rotvec(np.pi * axis).as_matrix()
        else:
            R, *_ = Rotation.align_vectors([u_new], [u_old])
            R = R.as_matrix()

        # Translate to center
        if center is not None:
            c = np.asarray(center, dtype=np.float64)
            self.translate(c)

        # Apply rotation: coords @ R.T → since R maps old→new basis
        self.coordinates = self.coordinates @ R.T

        # Translate back
        if center is not None:
            self.translate(-c)

        return R

    # ========================= Utils =========================

    def list_atoms(self, file: Optional[io.TextIOWrapper] = None) -> None:
        """Print atom list."""
        out = sys.stdout if file is None else file
        for i, (sym, (x, y, z)) in enumerate(zip(self.symbols, self.coordinates)):
            print(f"{i:4d}  {sym:>3}  {x:12.6f} {y:12.6f} {z:12.6f}", file=out)

    def __repr__(self) -> str:
        return f"XYZdata(natoms={self.natoms}, elements={self.symbols[:5]}{'...' if self.natoms > 5 else ''})"


# ============================= CLI =============================

class MyCLI(cmd.Cmd):
    intro = "FITXYZ - Molecular XYZ manipulator. Type 'help' for commands."
    prompt = "(fitxyz) "

    def __init__(self, xyz: Optional[XYZdata] = None):
        super().__init__()
        self.xyz: XYZdata = xyz or XYZdata([], np.zeros((0, 3)))

    # --------------------- Helpers ---------------------

    def _parse_vector_ref(self, ref: str) -> FArray:
        """Parse vector reference: x, y, z, Imax, 1 2, [1.0, 0, 0]"""
        tokens = shlex.split(ref)
        if len(tokens) == 1:
            t = tokens[0].upper()
            if t in ("X", "Y", "Z"):
                vec = np.zeros(3)
                vec["XYZ".index(t)] = 1.0
                return vec
            if t in ("IMAX", "IMID", "IMIN"):
                I = self.xyz.inertia_tensor()
                evals, evecs = self.xyz.diagonalize_inertia(I)
                idx = {"IMIN": 0, "IMID": 1, "IMAX": 2}[t]
                return evecs[:, idx]
        elif len(tokens) == 2:
            i, j = int(tokens[0]), int(tokens[1])
            if not (0 <= i < self.xyz.natoms and 0 <= j < self.xyz.natoms):
                raise IndexError("Atom index out of range")
            return self.xyz.coordinates[j] - self.xyz.coordinates[i]
        elif len(tokens) == 3:
            return np.array([float(x) for x in tokens])
        else:
            raise ValueError(f"Invalid vector spec: {ref}")

    # --------------------- Commands ---------------------

    def do_load(self, arg: str) -> None:
        """load <file.xyz> - Load XYZ file"""
        if not arg:
            print("Usage: load <filename.xyz>")
            return
        try:
            self.xyz = XYZdata.from_file(arg)
            print(f"Loaded {self.xyz.natoms} atoms from {arg}")
        except Exception as e:
            print(f"Error loading file: {e}")

    def do_list(self, arg: str) -> None:
        """list - Show all atoms"""
        if self.xyz.natoms == 0:
            print("No molecule loaded.")
            return
        self.xyz.list_atoms()

    def do_lsxyz(self, arg: str) -> None:
        """lsxyz - List .xyz files in current directory"""
        xyz_files = sorted(p for p in Path('.').glob('*.xyz'))
        if xyz_files:
            print("XYZ files:", "  ".join(p.name for p in xyz_files))
        else:
            print("No .xyz files found.")

    def do_com(self, arg: str) -> None:
        """com - Print center of mass"""
        com = self.xyz.center_of_mass()
        print(f"COM:  {com[0]:12.6f} {com[1]:12.6f} {com[2]:12.6f}")

    def do_center_at(self, arg: str) -> None:
        """center_at <com | midpoint [i j k...]> - Center molecule"""
        parts = shlex.split(arg)
        if not parts:
            print("Usage: center_at com | center_at midpoint 0 1 2")
            return

        if parts[0] == "com":
            center = self.xyz.center_of_mass()
            label = "COM"
        elif parts[0] == "midpoint":
            if len(parts) < 2:
                print("Specify at least one atom index")
                return
            indices = []
            for s in parts[1:]:
                try:
                    idx = int(s)
                    if not (0 <= idx < self.xyz.natoms):
                        raise IndexError
                    indices.append(idx)
                except:
                    print(f"Invalid atom index: {s}")
                    return
            center = np.mean(self.xyz.coordinates[indices], axis=0)
            label = f"midpoint of atoms {', '.join(map(str, indices))}"
        else:
            print("Unknown option. Use 'com' or 'midpoint ...'")
            return

        print(f"Centering at {label}: {center[0]:.6f} {center[1]:.6f} {center[2]:.6f}")
        self.xyz.center_at_origin(center)

    def do_align(self, arg: str) -> None:
        """align <ref1> with <ref2> - Align vector ref1 to ref2"""
        if 'with' not in arg.lower():
            print("Usage: align <vector1> with <vector2>")
            print("  <vector>: x, y, z, Imax, 1 2, [0,0,1]")
            return

        part1, part2 = [s.strip() for s in arg.split('with', 1)]
        if not part1 or not part2:
            print("Both vectors required.")
            return

        try:
            v1 = self._parse_vector_ref(part1)
            v2 = self._parse_vector_ref(part2)
        except Exception as e:
            print(f"Vector parse error: {e}")
            return

        print(f"Aligning [{v1[0]:.3f}, {v1[1]:.3f}, {v1[2]:.3f}] → "
              f"[{v2[0]:.3f}, {v2[1]:.3f}, {v2[2]:.3f}]")

        try:
            R = self.xyz.align_vector(v1, v2)
            print("Rotation matrix:")
            for row in R:
                print("  " + " ".join(f"{x:10.6f}" for x in row))
        except Exception as e:
            print(f"Alignment failed: {e}")

    def do_inertia(self, arg: str) -> None:
        """inertia - Show inertia tensor and principal moments"""
        if arg:
            print("Warning: arguments ignored.")
        I = self.xyz.inertia_tensor()
        evals, evecs = self.xyz.principal_axes()

        print("Inertia tensor (a.u.):")
        for row in I:
            print("  " + " ".join(f"{x:12.4f}" for x in row))
        print("\nPrincipal moments (Imin → Imax):")
        print("  ", " ".join(f"{v:12.4f}" for v in evals))
        print("\nPrincipal axes (columns):")
        for i in range(3):
            vec = evecs[:, i]
            print(f"  {i}: [{vec[0]: .3f}, {vec[1]: .3f}, {vec[2]: .3f}]")

    def do_write(self, arg: str) -> None:
        """write <file.xyz> - Save current geometry"""
        if not arg:
            print("Usage: write <filename.xyz>")
            return
        try:
            self.xyz.write_file(arg)
            print(f"Saved to {arg}")
        except Exception as e:
            print(f"Write failed: {e}")

    def do_save(self, arg: str) -> None:
        """save <file.xyz> - Alias for write"""
        self.do_write(arg)

    def do_vmd(self, arg: str) -> None:
        """vmd - View current structure in VMD"""
        if not shutil.which("vmd"):
            print("VMD not found in PATH")
            return
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False) as f:
            f.write(self.xyz.to_string())
            fname = f.name
        try:
            subprocess.run(['vmd', '-xyz', fname], check=True)
        except subprocess.CalledProcessError:
            print("VMD failed to start")
        finally:
            try:
                os.unlink(fname)
            except:
                pass

    def do_quit(self, arg: str) -> bool:
        """quit - Exit program"""
        print("Goodbye!")
        return True

    def do_exit(self, arg: str) -> bool:
        return self.do_quit(arg)

    def do_EOF(self, arg: str) -> bool:
        print()
        return self.do_quit(arg)

    # Aliases
    do_q = do_quit
    do_w = do_write
    do_s = do_save


# ============================= Main =============================

def main() -> None:
    if len(sys.argv) > 1:
        # Batch mode: load file and enter interactive
        filename = sys.argv[1]
        try:
            mol = XYZdata.from_file(filename)
            cli = MyCLI(mol)
            print(f"Loaded {filename}")
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        cli = MyCLI()

    cli.cmdloop()


if __name__ == "__main__":
    main()
