from firedrake import *
from firedrake.petsc import PETSc
from alfi import *
import numpy as np


class TwoDimLidDrivenCavityProblem(NavierStokesProblem):
    def __init__(self, baseN, diagonal=None, regularised=True):
        super().__init__()
        self.baseN = baseN
        if diagonal is None:
            diagonal = "left"
        self.diagonal = diagonal
        self.regularised = regularised

    def mesh(self, distribution_parameters):
        base = RectangleMesh(self.baseN, self.baseN, 2, 2,
                             distribution_parameters=distribution_parameters,
                             diagonal=self.diagonal)
        """
        import matplotlib.pyplot as plt
        triplot(base)
        plt.show()
        """
        return base

    def bcs(self, Z):
        bcs = [DirichletBC(Z.sub(0), self.driver(Z.ufl_domain()), 4),
               DirichletBC(Z.sub(0), Constant((0., 0.)), [1, 2, 3])]
        return bcs

    def has_nullspace(self): return True

    def driver(self, domain):
        (x, y) = SpatialCoordinate(domain)
        if self.regularised:
            driver = as_vector([x*x*(2-x)*(2-x)*(0.25*y*y), 0])
        else:
            driver = as_vector([(0.25*y*y), 0])
        return driver

    def char_length(self): return 2.0

    def relaxation_direction(self): return "0+:1-"


if __name__ == "__main__":

    from alfi.solver import *


    diagonal = "left"
    baseN    = 4

    for k in range(3, 10):
        for nref in range(1, 6):

            problem = TwoDimLidDrivenCavityProblem(baseN, diagonal)
            stokes  = ScottVogeliusSolver(problem,
                                      # problem
                                      hierarchy="bary",
                                      nref=nref,
                                      k=k,                      # discretization order
                                      gamma=0,
                                      solver_type="almg",

                                      # stabilization
                                      stabilisation_type=None,
                                      supg_method="burman",
                                      supg_magic=-1, # not used with burman
                                      stabilisation_weight=5e-3,
                                      restriction=True,

                                      # relaxation
                                      smoothing=2,              # smoothing steps
                                      patch="star",
                                      patch_composition="additive",

                                      use_mkl=False,
                                      rebalance_vertices=False,
                                      hierarchy_callback=None,
                                      high_accuracy=True,
                                      # nref_vis=1, # not used (ever)
                                    )

            stokes.solve(0)
