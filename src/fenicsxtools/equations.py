"""equations.py

General templates for conservation laws.
"""

import dolfinx
import ufl

class ConservationLaw:
    """Generalized conservation law.

    dU/dt + div F(U) = S(x, t, U)

    Attributes:
        U (ufl.core.expr.Expr): The conserved quantity.
        flux (callable): The flux function F(U).
        F (ufl.core.expr.Expr): The vector-valued flux.
        J (ufl.core.expr.Expr): The vector-valued flux Jacobian.
        S (ufl.core.expr.Expr): A scalar-valued source expression.
    """

    def __init__(self, U: dolfinx.fem.Function, F: ufl.core.expr.Expr,
        S: ufl.core.expr.Expr = None):
        """Constructor.

        Arguments:
            U (dolfinx.fem.Function): The conserved quantity. This reference is
                important for correctly constructing the solution method.
            F (ufl.core.expr.Expr): The vector-valued flux.
            S (ufl.core.expr.Expr, optional): A scalar-valued source expression.
                Default is None, which results in no forcing.
        """
        self.U = U
        self.F = F # Includes computation of flux Jacobian
        self.S = S

    @property
    def U(self) -> ufl.core.expr.Expr:
        """ufl.core.expr.Expr: The conserved quantity."""
        return self._U

    @U.setter
    def U(self, value: ufl.core.expr.Expr):
        self._U = value

    @property
    def F(self) -> ufl.core.expr.Expr:
        """ufl.core.expr.Expr: The vector-valued flux."""
        return self._F

    @F.setter
    def F(self, value):
        self._F = value
        # Compute J symbolically
        # UFL requires a Variable wrapper for differentiation
        Uvar = ufl.variable(self._U)
        # Nested replace-differentiate-replace
        self._J = ufl.replace(
            ufl.diff(
                ufl.replace(
                    self._F,
                    {self._U: Uvar}
                ),
                Uvar),
            {Uvar: self._U}
        )

    @property
    def J(self) -> ufl.core.expr.Expr:
        """ufl.core.expr.Expr: The vector-valued flux Jacobian."""
        return self._J

    @property
    def S(self) -> ufl.core.expr.Expr:
        """ufl.core.expr.Expr: A scalar-valued source expression."""
        return self._S

    @S.setter
    def S(self, value: ufl.core.expr.Expr):
        self._S = value

    def __str__(self):
        """Represent the equation as a string."""
        U = ufl.formatting.ufl2unicode.ufl2unicode(self._U)
        F = ufl.formatting.ufl2unicode.ufl2unicode(ufl.algorithms.ad.expand_derivatives(self._F))
        S = ufl.formatting.ufl2unicode.ufl2unicode(ufl.algorithms.ad.expand_derivatives(self._S)) if self._S is not None else '0'
        return f'dU/dt + div({F}) = {S}'

def get_advection(domain: dolfinx.mesh.Mesh,
    U: dolfinx.fem.Function, v: float | tuple) -> ConservationLaw:
    """Get constant- and homogeneous-coefficient scalar advection problem.

    dU/dt + div(U v) = 0

    Arguments:
        domain (dolfinx.mesh.Mesh): The domain of the problem.
        U (dolfinx.fem.Function): The solution variable. This reference is important for
            correctly constructing the solution method.
        v (Union[real, tuple]): The advection speed. Must be consistent with the
            dimension of the domain.

    Returns:
        ConservationLaw: The conservation law.
    """
    vv = dolfinx.fem.Constant(domain, v)
    return ConservationLaw(
        U=U,
        F=U * vv
    )

def get_advection_diffusion(domain: dolfinx.mesh.Mesh, U: dolfinx.fem.Function,
    v: float | tuple, d: float) -> ConservationLaw:
    """Get a constant- and homogeneneous-coefficient scalar advection-diffusion
    problem.

    dU/dt + div(U v - d grad U) = 0

    Arguments:
        domain (dolfinx.mesh.Mesh): The domain of the problem.
        v (Union[real, tuple]): The advection speed. Must be consistent with the
            dimension of the domain.
        d (real): The isotropic diffusion rate.

    Returns:
        ConservationLaw: The conservation law.
    """
    vv = dolfinx.fem.Constant(domain, v)
    return ConservationLaw(
        U=U,
        F=U * vv - d * ufl.grad(U)
    )

def get_depth_averaged_advection_diffusion(domain: dolfinx.mesh.Mesh,
    iota: dolfinx.fem.Function, h: dolfinx.fem.Function, v: dolfinx.fem.Function,
    D: ufl.core.expr.Expr) -> ConservationLaw:
    """Get a depth-averaged advection diffusion problem, with potentially space-
    and time-varying, anisotropic constants.

    diota/dt + div(iota v - D grad c) = 0

    Arguments:
        domain (dolfinx.mesh.Mesh): The domain of the problem.
        iota (dolfinx.fem.Function): The conserved height-scaled concentration. This
            reference is important for correctly constructing the solution
            method.
        h (dolfinx.fem.Function): The water column height function. This may be updated
            during solution to represent a time-varying quantity, but care
            must be taken to correctly update operators and residuals.
        v (dolfinx.fem.Function): The current velocity function. This may be updated
            during solution to represent a time-varying quantity, but care
            must be taken to correctly update operators and residuals.
        D (ufl.core.expr.Expr): The diffusion tensor. This may be state-dependent.

    Returns:
        ConservationLaw: The conservation law.
    """
    c = iota / h # Concentration
    return ConservationLaw(
        U=iota,
        F=iota * v - ufl.dot(D, ufl.grad(c))
    )
