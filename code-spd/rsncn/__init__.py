"""RSN-CN solvers for geodesic-convex optimization on the SPD manifold."""

from .vanilla_newton import VanillaNewtonOptions, solve_spd_vanilla_newton
from .nystrom_newton import NewtonOptions, solve_spd_nystrom_newton
from .full_cubic_newton import FullCubicOptions, solve_spd_full_cubic
from .nystrom_cubic_newton import SolverOptions, solve_spd_nystrom_cubic
from . import spd, subproblem

__all__ = [
    "VanillaNewtonOptions",
    "solve_spd_vanilla_newton",
    "NewtonOptions",
    "solve_spd_nystrom_newton",
    "FullCubicOptions",
    "solve_spd_full_cubic",
    "SolverOptions",
    "solve_spd_nystrom_cubic",
    "spd",
    "subproblem",
]
