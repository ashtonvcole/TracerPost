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
    def q(self, value)
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
    """Continuous Galerkin formulation for a conservation law."""
    def __init__(self, equation: equations.ConservationLaw,
        domain: dolfinx.mesh.Mesh,
        space: dolfinx.fem.FunctionSpace):
        # Preliminaries
        phi = ufl.TestFunction(space)
        n = ufl.FacetNormal(domain)

        # Set solution variable
        self.q = equation.U

        # Construct M(dq/dt; phi) using trial function
        self.mass_form = self._dq * phi * ufl.dx

        # Construct R(q; phi) from flux and source terms
        self.residual_form = ufl.dot(equation.F, ufl.grad(phi)) * ufl.dx
        # Add exterior boundary contribution
        self.residual_form += -ufl.conditional(
            ufl.dot(equation.J, n) > 0.0,
            ufl.dot(equation.F, n),
            dolfinx.fem.Constant(domain, 0.0)
        ) * phi * ufl.ds
        if equation.S is not None:
            self.residual_form += equation.S * phi * ufl.dx

class DGFormulation(DifferentialFormulation):
    """Discontinuous Galerkin formulation for a conservation law."""
    def __init__(self, equation: equations.ConservationLaw,
        domain: dolfinx.mesh.Mesh,
        space: dolfinx.fem.FunctionSpace, trace_function):
        # Preliminaries
        phi = ufl.TestFunction(space)
        n = ufl.FacetNormal(domain)

        # Set solution variable
        self.q = equation.U

        # Construct M(dq/dt; phi) using trial function
        self.mass_form = self._dq * phi * ufl.dx

        # Construct R(q; phi) from flux and source terms
        self.residual_form = ufl.dot(equation.F, ufl.grad(phi)) * ufl.dx
        # Add interior face contributions
        self.residual_form += -trace_function(
            equation.F,
            equation.U,
            equation.J,
            n
        ) * ufl.jump(phi) * ufl.dS
        # Add exterior boundary contribution
        self.residual_form += -ufl.conditional(
            ufl.dot(equation.J, n) > 0.0,
            ufl.dot(equation.F, n),
            dolfinx.fem.Constant(domain, 0.0)
        ) * phi * ufl.ds
        if equation.S is not None:
            self.residual_form += equation.S * phi * ufl.dx

# SUPG, SIPG, LDG
