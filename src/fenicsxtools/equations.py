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
        F (ufl.core.expr.Expr): The total vector-valued flux. May be None.
        F_stiff (ufl.core.expr.Expr): The stiff vector-valued flux. May be None.
        F_non_stiff (ufl.core.expr.Expr): The non-stiff vector-valued flux. May
            be None.
        J (ufl.core.expr.Expr): The total vector-valued flux Jacobian. May be
            None.
        J_stiff (ufl.core.expr.Expr): The stiff vector-valued flux Jacobian. May
            be None.
        J_non_stiff (ufl.core.expr.Expr): The non-stiff vector-valued flux
            Jacobian. May be None.
        S (ufl.core.expr.Expr): The total scalar-valued source. May be None.
        S_stiff (ufl.core.expr.Expr): The stiff scalar-valued source. May be
            None.
        S_non_stiff (ufl.core.expr.Expr): The non-stiff scalar-valued source.
            May be None.
    """

    def __init__(self, U: dolfinx.fem.Function, F_stiff: ufl.core.expr.Expr = None,
        F_non_stiff: ufl.core.expr.Expr = None, S_stiff: ufl.core.expr.Expr = None,
        S_non_stiff: ufl.core.expr.Expr = None):
        """Constructor.

        Arguments:
            U (dolfinx.fem.Function): The conserved quantity. This reference is
                important for correctly constructing the solution method.
            F_stiff (ufl.core.expr.Expr, optional): The stiff vector-valued
                flux. Default is None.
            F_non_stiff (ufl.core.expr.Expr, optional): The non-stiff vector-
                valued flux. Default is None.
            S_stiff (ufl.core.expr.Expr, optional): The stiff scalar-valued
                source. Default is None.
            S_non_stiff (ufl.core.expr.Expr, optional): The non-stiff scalar-
                valued source. Default is None.
        """
        self._U = U
        if F_stiff is not None and F_non_stiff is not None:
            self._F = F_stiff + F_non_stiff
        elif F_stiff is not None:
            self._F = F_stiff
        elif F_non_stiff is not None:
            self._F = F_non_stiff
        else:
            self._F = None
        self._F_stiff = F_stiff
        self._F_non_stiff = F_non_stiff
        if S_stiff is not None and S_non_stiff is not None:
            self._S = S_stiff + S_non_stiff
        elif S_stiff is not None:
            self._S = S_stiff
        elif S_non_stiff is not None:
            self._S = S_non_stiff
        else:
            self._S = None
        self._S_stiff = S_stiff
        self._S_non_stiff = S_non_stiff

        # Compute J symbolically
        # UFL requires a Variable wrapper for differentiation
        Uvar = ufl.variable(U)
        # Nested replace-differentiate-replace
        if F_stiff is not None and F_non_stiff is not None:
            self._J_stiff = ufl.replace(
                ufl.diff(
                    ufl.replace(
                        F_stiff,
                        {U: Uvar}
                    ),
                    Uvar),
                {Uvar: U}
            )
            self._J_non_stiff = ufl.replace(
                ufl.diff(
                    ufl.replace(
                        F_non_stiff,
                        {U: Uvar}
                    ),
                    Uvar),
                {Uvar: U}
            )
            self._J = self._J_stiff + self._J_non_stiff
        elif F_stiff is not None:
            self._J_stiff = ufl.replace(
                ufl.diff(
                    ufl.replace(
                        F_stiff,
                        {U: Uvar}
                    ),
                    Uvar),
                {Uvar: U}
            )
            self._J_non_stiff = None
            self._J = self._J_stiff
        elif F_non_stiff is not None:
            self._J_stiff = None
            self._J_non_stiff = ufl.replace(
                ufl.diff(
                    ufl.replace(
                        F_non_stiff,
                        {U: Uvar}
                    ),
                    Uvar),
                {Uvar: U}
            )
            self._J = self._J_non_stiff
        else:
            self._J_stiff = None
            self._J_non_stiff = None
            self._J = None

    @property
    def U(self) -> ufl.core.expr.Expr:
        """ufl.core.expr.Expr: The conserved quantity."""
        return self._U

    @property
    def F(self) -> ufl.core.expr.Expr:
        """ufl.core.expr.Expr: The total vector-valued flux."""
        return self._F

    @property
    def F_stiff(self) -> ufl.core.expr.Expr:
        """ufl.core.expr.Expr: The stiff vector-valued flux."""
        return self._F_stiff

    @property
    def F_non_stiff(self) -> ufl.core.expr.Expr:
        """ufl.core.expr.Expr: The non-stiff vector-valued flux."""
        return self._F_non_stiff

    @property
    def J(self) -> ufl.core.expr.Expr:
        """ufl.core.expr.Expr: The total vector-valued flux Jacobian."""
        return self._J

    @property
    def J_stiff(self) -> ufl.core.expr.Expr:
        """ufl.core.expr.Expr: The stiff vector-valued flux Jacobian."""
        return self._J_stiff

    @property
    def J_non_stiff(self) -> ufl.core.expr.Expr:
        """ufl.core.expr.Expr: The non-stiff vector-valued flux Jacobian."""
        return self._J_non_stiff

    @property
    def S(self) -> ufl.core.expr.Expr:
        """ufl.core.expr.Expr: The total scalar-valued source."""
        return self._S

    @property
    def S_stiff(self) -> ufl.core.expr.Expr:
        """ufl.core.expr.Expr: The stiff scalar-valued source."""
        return self._S_stiff

    @property
    def S_non_stiff(self) -> ufl.core.expr.Expr:
        """ufl.core.expr.Expr: The non-stiff scalar-valued source."""
        return self._S_non_stiff

    def __str__(self):
        """Represent the equation as a string."""
        U = ufl.formatting.ufl2unicode.ufl2unicode(self._U)
        F = ufl.formatting.ufl2unicode.ufl2unicode(ufl.algorithms.ad.expand_derivatives(self._F)) if self._F is not None else '0'
        S = ufl.formatting.ufl2unicode.ufl2unicode(ufl.algorithms.ad.expand_derivatives(self._S)) if self._S is not None else '0'
        return f'd{U}/dt + div({F}) = {S}'

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
        F_non_stiff=U * vv
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
        F_stiff= -d * ufl.grad(U),
        F_non_stiff=U * vv
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
        F_non_stiff=U * v
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
        F_stiff= -ufl.dot(D, ufl.grad(c)),
        F_non_stiff=iota * v
    )
