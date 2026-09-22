"""equations.py

General templates for conservation laws.
"""

import dolfinx
import ufl

class Flux:
    """A lightweight wrapper for a flux component of a conservation law.

    Attributes:
        expression (ufl.core.expr.Expr): The flux expression.
        jacobian_expression (ufl.core.expr.Expr): The flux Jacobian dF/dU.
        is_stiff (bool): Whether the term should be treated implicitly in
            implicit-explicit (IMEX) formulations.
        is_hyperbolic (bool): Whether the term is purely hyperbolic, or has
            higher derivatives that lead to a parabolic conservation law. This
            is used by the LDG formulation to determine whether an upwinded or
            alternating trace is used.
    """

    def __init__(self, U: dolfinx.fem.Function, expression: ufl.core.expr.Expr,
        is_stiff: bool, is_hyperbolic: bool):
        """Constructor.

        Arguments:
            U (dolfinx.fem.Function): The conserved quantity.
            expression (ufl.core.expr.Expr): The flux expression.
            is_stiff (bool): Whether the term should be treated implicitly in
                implicit-explicit (IMEX) formulations.
            is_hyperbolic (bool): Whether the term is purely hyperbolic, or has
                higher derivatives that lead to a parabolic conservation law.
                This is used by the LDG formulation to determine whether an
                upwinded or alternating trace is used.
        """
        self._U = U
        self._expression = expression
        Uvar = ufl.variable(U)
        self._jacobian_expression = ufl.replace(
            ufl.diff(
                ufl.replace(
                    F_non_stiff,
                    {U: Uvar}
                ),
                Uvar),
            {Uvar: U}
        )
        self._is_stiff = is_stiff
        self._is_hyperbolic = is_hyperbolic

    @property
    def U(self) -> dolfinx.fem.Function:
        return self._U

    @property
    def expression(self) -> ufl.core.expr.Expr:
        """ufl.core.expr.Expr: The flux expression."""
        return self._expression

    def jacobian_expression(self) -> ufl.core.expr.Expr:
        "ufl.core.expr.Expr: The flux Jacobian dF/dU."
        return self._jacobian_expression

    @property
    def is_stiff(self) -> bool:
        """bool: Whether the term should be treated implicitly in
            implicit-explicit (IMEX) formulations."""
        return self._is_stiff

    @property
    def is_hyperbolic(self) -> bool:
        """bool: Whether the term is purely hyperbolic, or has
            higher derivatives that lead to a parabolic conservation law.
            This is used by the LDG formulation to determine whether an
            upwinded or alternating trace is used."""
        return self._is_hyperbolic

class Source:
    """A lightweight wrapper for a source component of a conservation law.

    Attributes:
        expression (ufl.core.expr.Expr): The flux expression.
        is_stiff (bool): Whether the term should be treated implicitly in
            implicit-explicit (IMEX) formulations.
    """

    def __init__(self, expression: ufl.core.expr.Expr, is_stiff: bool):
        """Constructor.

        Arguments:
            expression (ufl.core.expr.Expr): The flux expression.
            is_stiff (bool): Whether the term should be treated implicitly in
                implicit-explicit (IMEX) formulations.
        """
        self._expression = expression
        self._is_stiff = is_stiff

    @property
    def expression(self) -> ufl.core.expr.Expr:
        """ufl.core.expr.Expr: The source expression."""
        return self._expression

    @property
    def is_stiff(self) -> bool:
        """bool: Whether the term should be treated implicitly in
            implicit-explicit (IMEX) formulations."""
        return self._is_stiff

class ConservationLaw:
    """Generalized conservation law.

    dU/dt + div F(U) = S(x, t, U)

    Attributes:
        U (ufl.core.expr.Expr): The conserved quantity.
        fluxes (ufl.core.expr.Expr, optional): A list of vector-valued fluxes.
            Default is None.
        sources (ufl.core.expr.Expr, optional): A list of scalar-valued sources.
            Default is None.
    """

    def __init__(self, U: dolfinx.fem.Function,
        fluxes: list[Flux] | None = None,
        sources: list[Source] | None = None):
        """Constructor.

        Arguments:
            U (dolfinx.fem.Function): The conserved quantity. This reference is
                important for correctly constructing the solution method.
            fluxes (ufl.core.expr.Expr, optional): A list of vector-valued
                fluxes. Default is None.
            sources (ufl.core.expr.Expr, optional): A list of scalar-valued
                sources. Default is None.
        """
        self._U = U
        self._fluxes = fluxes if fluxes is not None else []
        self._sources = sources if fluxes is not None else []

    @property
    def U(self) -> ufl.core.expr.Expr:
        """ufl.core.expr.Expr: The conserved quantity."""
        return self._U

    @property
    def fluxes(self) -> ufl.core.expr.Expr:
        """ufl.core.expr.Expr: A list of vector-valued fluxes. May be empty."""
        return self._fluxes

    @property
    def sources(self) -> ufl.core.expr.Expr:
        """ufl.core.expr.Expr: A list of scalar-valued sources. May be empty."""
        return self._sources

def get_constant_advection(domain: dolfinx.mesh.Mesh,
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
        fluxes=[
            Flux(
                U=U,
                expression=U * vv,
                is_stiff=False,
                is_hyperbolic=True
            )
        ]
    )

def get_constant_advection_diffusion(domain: dolfinx.mesh.Mesh,
    U: dolfinx.fem.Function, v: float | tuple, d: float) -> ConservationLaw:
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
        fluxes=[
            Flux(
                U=U,
                expression=U * vv,
                is_stiff=False,
                is_hyperbolic=True
            ),
            Flux(
                U=U,
                expression=-d * ufl.grad(U),
                is_stiff=True,
                is_hyperbolic=False
            )
        ]
    )

def get_advection(domain: dolfinx.mesh.Mesh, U: dolfinx.fem.Function,
    v: dolfinx.fem.Function) -> ConservationLaw:
    """Get an advection problem, with potentially space- and time-varying
    velocity.

    du/dt + div(u v) = 0

    Arguments:
        domain (dolfinx.mesh.Mesh): The domain of the problem.
        u (dolfinx.fem.Function): The conserved quantity. This reference is
            important for correctly constructing the solution method.
        v (dolfinx.fem.Function): The advection velocity function. This may be
            updated during solution to represent a time-varying quantity, but
            care must be taken to correctly update operators and residuals.
    """
    return ConservationLaw(
        U=U,
        fluxes=[
            Flux(
                U=U,
                expression=U * v,
                is_stiff=False,
                is_hyperbolic=True
            )
        ]
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
        fluxes=[
            Flux(
                U=iota,
                expression=iota * v,
                is_stiff=False,
                is_hyperbolic=True
            ),
            Flux(
                U=iota,
                expression=-ufl.dot(D, ufl.grad(c)),
                is_stiff=True,
                is_hyperbolic=False
            )
        ]
    )
