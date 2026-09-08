"""equations.py

General templates for conservation laws.
"""

import dolfinx
import ufl

class HyperbolicConservationLaw:
    """Generalized scalar hyperbolic conservation law.

    dU/dt + div F(U) = S(x, t, U)

    Attributes:
        U (ufl.Expr): The conserved quantity. This reference is important for
            correctly constructing the solution method.
        F (ufl.Expr): A vector-valued hyperbolic flux, dependent on the state U.
        S (ufl.Expr): A scalar-valued source expression, dependent on the
            position x, time t, and state U.
    """

    def __init__(self, U: ufl.Function, F: ufl.Expr, S: ufl.Expr = None):
        """Constructor.

        Arguments:
            U (ufl.Function): The conserved quantity. This reference is
                important for correctly constructing the solution method.
            F (ufl.Expr): A vector-valued hyperbolic flux, dependent on the
                state U.
            S (ufl.Expr, optional): A scalar-valued source expression, dependent
                on the position x, time t, and state U. Default is None, which
                results in no forcing.
        """
        self.F = F
        self.S = S

    @property
    def U(self) -> ufl.expr:
        """ufl.Expr: The scalar-valued conserved variable."""
        return self._U

    @U.setter
    def U(self, value: ufl.Expr):
        self._U = value

    @property
    def F(self) -> ufl.Expr:
        """ufl.Expr: The vector-valued hyperbolic flux."""
        return self._F

    @F.setter
    def F(self, value: ufl.Expr):
        self._F = value

    @property
    def S(self) -> ufl.Expr:
        """ufl.Expr: The scalar-valued source term."""
        return self._S

    @S.setter
    def S(self, value: ufl.Expr):
        self._S = value

class ParabolicConservationLaw:
    """Generalized scalar parabolic conservation law.

    dU/dt + div F(U, grad U) = S(x, t, U)

    Attributes:
        U (ufl.Function): The conserved quantity. This reference is important
            for correctly constructing the solution method.
        F (ufl.Expr): A vector-valued parabolic flux, dependent on the state U
            and gradient grad U.
        S (ufl.Expr): A scalar-valued source expression, dependent on the
            position x, time t, and state U.
        grad_of (ufl.Expr): The expression whose gradient is taken. This is
            necessary for the LDG formulation. Often, this is just U. For
            depth-averaged transoport, however, it would be U / h.
    """

    def __init__(self, U: ufl.Function, F: ufl.Expr, grad_of: ufl.Expr = None,
        S: ufl.Expr = None):
        """Constructor.

        Arguments:
            U (ufl.Function): The conserved quantity. This reference is
                important for correctly constructing the solution method.
            F (ufl.Expr): A vector-valued hyperbolic flux, dependent on the
                state U and gradient grad U.
            grad_of (ufl.Expr, optional): The expression whose gradient is
                taken. This is necessary for formulations like Local
                Discontinuous Galerkin. Often, this is just U. For
                depth-averaged transoport, however, it would be U / h. Default
                is U.
            S (ufl.Expr, optional): A scalar-valued source expression, dependent
                on the position x, time t, and state U. Default is None, which
                results in no forcing.
        """
        self.U = U
        self.F = F
        if grad_of is not None:
            self.grad_of = grad_of
        else:
            self.grad_of = U
        self.S = S

    @property
    def U(self) -> ufl.expr:
        """ufl.Expr: The scalar-valued conserved variable."""
        return self._U

    @U.setter
    def U(self, value: ufl.Expr):
        self._U = value

    @property
    def F(self) -> ufl.Expr:
        """ufl.Expr: The vector-valued parbolic flux."""
        return self._F

    @F.setter
    def F(self, value: ufl.Expr):
        self._F = value

    @property
    def S(self) -> ufl.Expr:
        """ufl.Expr: The scalar-valued source term."""
        return self._S

    @S.setter
    def S(self, value: ufl.Expr):
        self._S = value

    @property
    def grad_of(self) -> ufl.Expr:
        """ufl.Expr: The expression whose gradient is an auxiliary variable."""
        return self._grad_of

    @grad_of.setter
    def grad_of(self, value: ufl.Expr):
        self._grad_of = value

def get_advection(domain: dolfinx.mesh.Mesh,
    U: ufl.Function, v: Union[real, tuple]) -> HyperbolicConservationLaw:
    """Get constant- and homogeneous-coefficient scalar advection problem.

    dU/dt + div(U v) = 0

    Arguments:
        domain (dolfinx.mesh.Mesh): The domain of the problem.
        U (ufl.Function): The solution variable. This reference is important for
            correctly constructing the solution method.
        v (Union[real, tuple]): The advection speed. Must be consistent with the
            dimension of the domain.

    Returns:
        HyperbolicConservationLaw: The conservation law.
    """
    return HyperbolicConservationLaw(
        U=U,
        F=U * ufl.Constant(domain, v)
    )

def get_advection_diffusion(domain: dolfinx.mesh.Mesh, U: ufl.Function,
    v: Union[real, tuple], d: real) -> ParabolicConservationLaw:
    """Get a constant- and homogeneneous-coefficient scalar advection-diffusion
    problem.

    dU/dt + div(U v - d grad U) = 0

    Arguments:
        domain (dolfinx.mesh.Mesh): The domain of the problem.
        v (Union[real, tuple]): The advection speed. Must be consistent with the
            dimension of the domain.
        d (real): The isotropic diffusion rate.

    Returns:
        ParabolicConservationLaw: The conservation law.
    """
    return ParabolicConservationLaw(
        U=U,
        F=U * ufl.Constant(domain, v) - d * ufl.grad(U)
    )

def get_depth_averaged_advection_diffusion(domain: dolfinx.mesh.Mesh,
    iota: ufl.Function, h: ufl.Function, v: ufl.Function,
    D: ufl.Expr) -> ParabolicConservationLaw:
    """Get a depth-averaged advection diffusion problem, with potentially space-
    and time-varying, anisotropic constants.

    diota/dt + div(iota v - D grad c = 0

    Arguments:
        domain (dolfinx.mesh.Mesh): The domain of the problem.
        iota (ufl.Function): The conserved height-scaled concentration. This
            reference is important for correctly constructing the solution
            method.
        h (ufl.Function): The water column height function. This may be updated
            during solution to represent a time-varying quantity, but care
            must be taken to correctly update operators and residuals.
        v (ufl.Function): The current velocity function. This may be updated
            during solution to represent a time-varying quantity, but care
            must be taken to correctly update operators and residuals.
        D (ufl.Expr): The diffusion tensor. This may be state-dependent.

    Returns:
        ParabolicConservationLaw: The conservation law.
    """
    c = iota / h # Concentration
    return ParabolicConservationLaw(
        U=iota,
        F=iota * v + ufl.inner(D, ufl.grad(c)),
        grad_of=c
    )
