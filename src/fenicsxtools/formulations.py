"""formulations.py

Different FEM formulation options for conservation laws.
"""

from . import equations
import dolfinx
import ufl

class Formulation:
    """Generalized FEM formulation.

    This formulation relies on the following general semi-discrete weak form
    that comes from conservation laws.

    Find q in Uh such that, for all phi in Vh,

    mass_form(dq/dt; phi) = residual(q; phi)

    Attributes:
        equation (equations.ConservationLaw): The conservation law being solved.
        mass_form (ufl.Form): The weak form of the time derivative.
        residual_form (ufl.Form): The semi-discrete residual.
        dresidual_form (ufl.Form): The weak form for the semi-discrete Jacobian
            dR/dq. Not to be confused with the flux Jacobian.
    """

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

    @property
    def dresidual_form(self) -> ufl.Form:
        """ufl.Form: The weak form for the semi-discrete Jacobian dR/dq. Not to
        be confused with the flux Jacobian.
        """
        return self._dresidual_form

    @dresidual_form.setter
    def dresidual_form(self, value: ufl.Form):
        self._dresidual_form = value

class CGFormulation(Formulation):
    """Continuous Galerkin formulation for a conservation law."""
    def __init__(self, equation: equations.ConservationLaw,
        domain: dolfinx.mesh.Mesh,
        space: dolfinx.fem.FunctionSpace):
        dq_dt = ufl.TrialFunction(space) # For mass form
        dq = ufl.TrialFunction(space) # For residual Jacobian form
        phi = ufl.TestFunction(space)
        n = ufl.FacetNormal(domain)

        # Construct M(dq/dt; phi)
        self.mass_form = dq_dt * phi * ufl.dx

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

        # Construct dR/dq(q; phi)
        self.dresidual_form = ufl.derivative(
            self.residual_form,
            equation.U,
            dq
        )

class DGFormulation(Formulation):
    def __init__(self, equation: equations.ConservationLaw,
        domain: dolfinx.mesh.Mesh,
        space: dolfinx.fem.FunctionSpace, trace_function):
        dq_dt = ufl.TrialFunction(space)
        phi = ufl.TestFunction(space)
        n = ufl.FacetNormal(domain)

        # Construct M(dq/dt; phi)
        self.mass_form = dq_dt * phi * ufl.dx

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

        # Construct dR/dq(q; phi)
        self.dresidual_form = ufl.derivative(
            self.residual_form,
            equation.U,
            phi
        )

# SUPG, SIPG, LDG
