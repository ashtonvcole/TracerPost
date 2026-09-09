"""formulations.py

Different FEM formulation options for conservation laws.
"""

from . import equations
import dolfinx
import ufl

class DifferentialFormulation:
    """Generalized simple FEM formulation.

    This formulation relies on the following general semi-discrete weak form
    that comes from conservation laws.

    Find q in Uh such that, for all phi in Vh,

    mass_form(dq/dt; phi) = residual(q; phi)

    Attributes:
        q (dolfinx.fem.Function): The solution variable.
        equation (equations.ConservationLaw): The conservation law being solved.
        mass_form (ufl.Form): The weak form of the time derivative.
        residual_form (ufl.Form): The weak form of the residual.
        dresidual_form_dq (ufl.Form): The weak form for the semi-discrete
            Jacobian dR/dq. Not to be confused with the flux Jacobian.
    """

    @property
    def q(self) -> dolfinx.fem.Function:
        """dolfinx.fem.Function: The solution variable."""
        return self._q

    @q.setter
    def q(self, value):
        self._q = value
        self._dq = ufl.TrialFunction(value.function_space)

    @property
    def equation(self) -> equations.ConservationLaw:
        """equations.ConservationLaw: The conservation law being solved."""
        return self._equation

    @equation.setter
    def equation(self, value: equations.ConservationLaw):
        self._equation = value

    @property
    def mass_form(self) -> ufl.Form:
        """ufl.Form: The weak form of the time derivative."""
        return self._mass_form

    @mass_form.setter
    def mass_form(self, value: ufl.Form):
        self._mass_form = value

    @property
    def residual_form(self) -> ufl.Form:
        """ufl.Form: The weak form of the residual."""
        return self._residual_form

    @residual_form.setter
    def residual_form(self, value: ufl.Form):
        self._residual_form = value
        # Update form Jacobian
        self._dresidual_form_dq = ufl.derivative(
            value,
            self.q,
            self._dq
        )

    @property
    def dresidual_form_dq(self) -> ufl.Form:
        """ufl.Form: The weak form for the semi-discrete Jacobian dR/dq. Not to
        be confused with the flux Jacobian.
        """
        return self._dresidual_form_dq

class DifferentialAlgebraicFormulation:
    """Generalized differential-algebraic FEM formulation.

    This formulation relies on the following general semi-discrete weak form
    that comes from conservation laws.

    Find q in Uh such that, for all phi in Vh and psi in Wh,

    mass_form_q(dq/dt; phi) = residual_q(q, r; phi)
    mass_form_r(r; psi) = residual_r(q, r; psi)

    Attributes:
        q (dolfinx.fem.Function): The solution variable.
        r (dolfinx.fem.Function): The algebraic variable.
        equation (equations.ConservationLaw): The conservation law being solved.
        mass_form_q (ufl.Form): The weak form of the time derivative of q.
        mass_form_r (ufl.Form): The weak form of the mass form of r.
        residual_form_q (ufl.Form): The semi-discrete residual of q.
        residual_form_r (ufl.Form): The semi-discrete residual of r.
        dresidual_q_form (ufl.Form): The weak form for the semi-discrete
            Jacobian dR_q/dq + dR_q/dr. Not to be confused with the flux
            Jacobian.
        dresidual_r_form (ufl.Form): The weak form for the semi-discrete
            Jacobian dR_r/dq + dR_r/dr. Not to be confused with the flux
            Jacobian.
    """

    @property
    def q(self) -> dolfinx.fem.Function:
        """dolfinx.fem.Function: The solution variable."""
        return self._q

    @q.setter
    def q(self, value):
        self._q = value
        self._dq = ufl.TrialFunction(value.function_space)

    @property
    def r(self) -> dolfinx.fem.Function:
        """dolfinx.fem.Function: The algebraic variable."""
        return self._r

    @r.setter
    def r(self, value):
        self._r = value
        self._dr = ufl.TrialFunction(value.function_space)

    @property
    def equation(self) -> equations.ConservationLaw:
        """equations.ConservationLaw: The conservation law being solved."""
        return self._equation

    @equation.setter
    def equation(self, value: equations.ConservationLaw):
        self._equation = value

    @property
    def mass_form_q(self) -> ufl.Form:
        """ufl.Form: The weak form of the time derivative of q."""
        return self._mass_form_q

    @mass_form_q.setter
    def mass_form_q(self, value: ufl.Form):
        self._mass_form_q = value

    @property
    def mass_form_r(self) -> ufl.Form:
        """ufl.Form: The weak form of the mass form of r."""
        return self._mass_form_r

    @mass_form_r.setter
    def mass_form_r(self, value: ufl.Form):
        self._mass_form_r = value

    @property
    def residual_form_q(self) -> ufl.Form:
        """ufl.Form: The weak form of the residual of q."""
        return self._residual_form_q

    @residual_form_q.setter
    def residual_form_q(self, value: ufl.Form):
        self._residual_form_q = value
        # Update q form Jacobians
        self._dresidual_form_q_dq = ufl.derivative(
            value,
            self.q,
            self._dq
        )
        self._dresidual_form_q_dr = ufl.derivative(
            value,
            self.r,
            self._dr
        )

    @property
    def residual_form_r(self) -> ufl.Form:
        """ufl.Form: The weak form of the residual of r."""
        return self._residual_form_r

    @residual_form_r.setter
    def residual_form_r(self, value: ufl.Form):
        self._residual_form_r = value
        # Update r form Jacobians
        self._dresidual_form_r_dq = ufl.derivative(
            value,
            self.q,
            self._dq
        )
        self._dresidual_form_r_dr = ufl.derivative(
            value,
            self.r,
            self._dr
        )

    @property
    def dresidual_form_q_dq(self) -> ufl.Form:
        """ufl.Form: The weak form for the semi-discrete Jacobian dR_q/dq. Not
        to be confused with the flux Jacobian.
        """
        return self._dresidual_form_q_dq

    @property
    def dresidual_form_q_dr(self) -> ufl.Form:
        """ufl.Form: The weak form for the semi-discrete Jacobian dR_q/dr. Not
        to be confused with the flux Jacobian.
        """
        return self._dresidual_form_q_dr

    @property
    def dresidual_form_r_dq(self) -> ufl.Form:
        """ufl.Form: The weak form for the semi-discrete Jacobian dR_r/dq. Not
        to be confused with the flux Jacobian.
        """
        return self._dresidual_form_r_dq

    @property
    def dresidual_form_r_dr(self) -> ufl.Form:
        """ufl.Form: The weak form for the semi-discrete Jacobian dR_r/dr. Not
        to be confused with the flux Jacobian.
        """
        return self._dresidual_form_r_dr

class CGFormulation(DifferentialFormulation):
    """Continuous Galerkin formulation for a conservation law.

    Given an arbitrary conservation law,

    dU/dt + div F = S

    this constructs the simple CG weak form.

    dq/dt * phi * dx = F . grad(phi) * dx - F . n * phi * ds + S * phi * dx

    It inherits from the DifferentialFormulation class.
    """

    def __init__(self, equation: equations.ConservationLaw):
        # Preliminaries
        space = equation.U.function_space
        domain = equation.U.function_space.mesh
        phi = ufl.TestFunction(space)
        n = ufl.FacetNormal(domain)

        # Set solution variable and equation
        self.q = equation.U
        self.equation = equation

        # Construct M(dq/dt; phi) using trial function
        self.mass_form = self._dq * phi * ufl.dx

        # Construct R(q; phi) from flux and source terms
        self.residual_form = ufl.dot(self.equation.F, ufl.grad(phi)) * ufl.dx
        # Add exterior boundary contribution
        self.residual_form += -ufl.conditional(
            ufl.dot(self.equation.J, n) > 0.0,
            ufl.dot(self.equation.F, n),
            dolfinx.fem.Constant(domain, 0.0)
        ) * phi * ufl.ds
        # Add source
        if self.equation.S is not None:
            self.residual_form += self.equation.S * phi * ufl.dx

class DGFormulation(DifferentialFormulation):
    """Discontinuous Galerkin formulation for a conservation law.

    Given an arbitrary conservation law,

    dU/dt + div F = S

    this constructs the simple DG weak form.

    dq/dt * phi * dx = F . grad(phi) * dx - trace(F . n) * jump(phi) * dS -
    F . n * phi * ds + S * phi * dx

    It inherits from the DifferentialFormulation class.
    """

    def __init__(self, equation: equations.ConservationLaw, trace_function):
        # Preliminaries
        space = equation.U.function_space
        domain = equation.U.function_space.mesh
        phi = ufl.TestFunction(space)
        n = ufl.FacetNormal(domain)

        # Set solution variable and equation
        self.q = equation.U
        self.equation = equation

        # Construct M(dq/dt; phi) using trial function
        self.mass_form = self._dq * phi * ufl.dx

        # Construct R(q; phi) from flux and source terms
        self.residual_form = ufl.dot(self.equation.F, ufl.grad(phi)) * ufl.dx
        # Add interior face contributions
        self.residual_form += -trace_function(
            self.equation.F,
            self.equation.U,
            self.equation.J,
            n
        ) * ufl.jump(phi) * ufl.dS
        # Add exterior boundary contribution
        self.residual_form += -ufl.conditional(
            ufl.dot(self.equation.J, n) > 0.0,
            ufl.dot(self.equation.F, n),
            dolfinx.fem.Constant(domain, 0.0)
        ) * phi * ufl.ds
        # Add source
        if self.equation.S is not None:
            self.residual_form += self.equation.S * phi * ufl.dx

class LDGFormulation(DifferentialAlgebraicFormulation):
    """Local Discontinuous Galerkin formulation for a conservation law.

    Given an arbitrary parabolic conservation law,

    dU/dt + div F(U, grad U) = S

    this constructs a DG weak form with an auxiliary variable for a gradient.

    dq/dt * phi * dx = F(q, r) . grad(phi) * dx - trace(F(q, r) . n) *
        jump(phi) * dS - F(q, r) . n * phi * ds + S * phi * dx
    r . psi * dx = - grad_of * div(psi) * dx + trace(grad_of) * jump(psi) . n *
        dS + grad_of * psi . n * ds

    It inherits from the DifferentialAlgebraicFormulation class.
    """

    def __init__(self, equation: equations.ConservationLaw,
        grad_of: ufl.core.expr.Expr, space_auxiliary: dolfinx.fem.FunctionSpace,
        trace_function_q, trace_function_r):
        # Preliminaries
        space = equation.U.function_space
        domain = equation.U.function_space.mesh
        phi = ufl.TestFunction(space)
        psi = ufl.TestFunction(space_auxiliary)
        n = ufl.FacetNormal(domain)

        # Set solution and algebraic variables
        self.q = equation.U
        self.equation = equation
        self.r = dolfinx.fem.Function(space_auxiliary)

        # Replace gradient in flux with auxiliary variable for LDG
        self.equation.F = ufl.replace(
            self.equation.F,
            {ufl.grad(grad_of): self.r}
        )

        # Construct q mass
        self.mass_form_q = self._dq * phi * ufl.dx

        # Construct q residual
        self.residual_form_q = ufl.dot(self.equation.F, ufl.grad(phi)) * ufl.dx
        # Add interior face contributions
        self.residual_form_q += -trace_function_q(
            self.equation.F,
            self.equation.U,
            self.equation.J,
            n
        ) * ufl.jump(phi) * ufl.dS
        # Add exterior boundary contribution
        self.residual_form_q += -ufl.conditional(
            ufl.dot(self.equation.J, n) > 0.0,
            ufl.dot(self.equation.F, n),
            dolfinx.fem.Constant(domain, 0.0)
        ) * phi * ufl.ds
        # Add source
        if equation.S is not None:
            self.residual_form_q += self.equation.S * phi * ufl.dx

        # Construct r mass
        self.mass_form_r = ufl.dot(self._dr, psi) * ufl.dx

        # Construct r residual
        self.residual_form_r = -grad_of * ufl.div(psi) * ufl.dx
        # Add interior face contributions
        self.residual_form_r += trace_function_r(
            grad_of,
            self.equation.U,
            self.equation.J,
            n
        ) * ufl.jump(psi, n) * ufl.dS
        # Add exterior boundary conditions
        # @TODO fix this?
        self.residual_form_r += grad_of * ufl.dot(psi, n) * ufl.ds

# SUPG, SIPG, LDG
