"""equations.py

General templates for conservation laws.
"""

import ufl

class HyperbolicConservationLaw:
    """Generalized scalar hyperbolic conservation law.

    dU/dt + div F(U) = S(x, t, U)

    Attributes:
        F (ufl.Expr): A vector-valued hyperbolic flux, dependent on the state U.
        S (ufl.Expr): A scalar-valued source expression, dependent on the
            position x, time t, and state U.
    """

    @property
    def F(self) -> ufl.Expr:
        """ufl.Expr: The vector-valued hyperbolic flux."""
        return self._F

    @F.setter
    def F(self, value: ufl.Expr):
        self._F = value

    @property
    def F(self) -> ufl.Expr:
        """ufl.Expr: The scalar-valued source term."""
        return self._S

    @S.setter
    def S(self, value: ufl.Expr):
        self._S = value

class ParabolicConservationLaw:
    """Generalized scalar hyperbolic conservation law.

    dU/dt + div F(U, grad U) = S(x, t, U)

    Attributes:
        F (ufl.Expr): A vector-valued parabolic flux, dependent on the state U
            and gradient grad U.
        S (ufl.Expr): A scalar-valued source expression, dependent on the
            position x, time t, and state U.
    """

    @property
    def F(self) -> ufl.Expr:
        """ufl.Expr: The vector-valued parbolic flux."""
        return self._F

    @F.setter
    def F(self, value: ufl.Expr):
        self._F = value

    @property
    def F(self) -> ufl.Expr:
        """ufl.Expr: The scalar-valued source term."""
        return self._S

    @S.setter
    def S(self, value: ufl.Expr):
        self._S = value
