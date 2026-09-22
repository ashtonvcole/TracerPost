"""formulations.py

Different FEM formulation options for conservation laws.
"""

from types import NoneType

from . import equations
import dolfinx
import ufl

class SemiDiscreteEquation:
    """Abstraction for semidiscrete equations found in FEM formulations.

    B(xi; phi) = R_stiff(q, r, ...; phi) + R_nonstiff(q, r, ...; phi)

    Attributes:
        variable (dolfinx.fem.function): The solution variable associated with
            this equation.
        is_differential (bool): Whether this equation is a differential or
            algebraic constraint. E.g., if variable = r and is_differential =
            True, then xi is dr/dt.
        bilinear_form (ufl.Form): The left-hand side of the equation, bilinear
            in both the trial and test functions.
        stiff_residual_form (ufl.Form): A stiff component of the residual,
            marked for implicit treatment in an IMEX solver. This should be
            linear in the test function. May be None.
        non_stiff_residual_form (ufl.Form): A non-stiff component of the
            residual, marked for explicit treatment in an IMEX solver. This
            should be linear in the test function. May be None.
    """
    def __init__(self, variable: dolfinx.fem.Function, is_differential: bool,
        bilinear_form: ufl.Form,
        stiff_residual_form: ufl.Form | None = None,
        non_stiff_residual_form: ufl.Form | None = None,
        is_bilinear_form_constant: bool = False):
        """Constructor.

        Arguments:
            variable (dolfinx.fem.function): The solution variable associated
                with this equation.
            is_differential (bool): Whether this equation is a differential or
                algebraic constraint. E.g., if variable = r and is_differential
                = True, then xi is dr/dt.
            bilinear_form (ufl.Form): The left-hand side of the equation,
                bilinear in both the trial and test functions.
            stiff_residual_form (ufl.Form, optional): A stiff component of the
                residual, marked for implicit treatment in an IMEX solver. This
                should be linear in the test function. Default is None.
            non_stiff_residual_form (ufl.Form, optional): A non-stiff component
                of the residual, marked for explicit treatment in an IMEX
                solver. This should be linear in the test function. Default is
                None.
            is_bilinear_form_constant (bool, optional): Whether the bilinear
                form is constant in time. This impacts whether matrices need to
                be re-formed at every time step, which is an expensive
                operation. Default is False.
        """
        self._variable = variable
        self._is_differential = is_differential
        self._bilinear_form = bilinear_form
        self._stiff_residual_form = stiff_residual_form
        self._non_stiff_residual_form = non_stiff_residual_form
        self._is_bilinear_form_constant = is_bilinear_form_constant

    @property
    def variable(self) -> dolfinx.fem.Function:
        """dolfinx.fem.function: The solution variable associated with this
        equation.
        """
        return self._variable

    @property
    def is_differential(self) -> bool:
        """bool: Whether this equation is a differential or algebraic
        constraint.
        """
        return self._is_differential

    @property
    def bilinear_form(self) -> ufl.Form:
        """ufl.Form: The left-hand side of the equation, bilinear in both the trial
        and test functions.
        """
        return self._bilinear_form

    @property
    def stiff_residual_form(self):
        return self._stiff_residual_form

    @property
    def non_stiff_residual_form(self):
        return self._non_stiff_residual_form

class SemiDiscreteSystem:
    """Abstraction for  a semi-discrete system of equations found in FEM
    formulations.

    Attributes:
        equations (list[SemiDiscreteEquation]): A list of differential and
            algebraic equations.
        lifts (list[SemiDiscreteEquation]): A list of intermediate variable
            lifts, provided in dependency order. E.g., if the equation for c
            is dependent on a and b, b on a and c, and a on c, then the order
            is a, b.
    """
    def __init__(self, equations: list[SemiDiscreteEquation],
        lifts: list[SemiDiscreteEquation] | None = None):
        """Constructor.

        Arguments:
            equations (list[SemiDiscreteEquation]): A list of differential and
                algebraic equations.
            lifts (list[SemiDiscreteEquation], optional): A list of intermediate
                variable lifts, provided in dependency order. E.g., if the
                equation for c is dependent on a and b, b on a and c, and a on
                c, then the order is a, b. Default is None.
        """
        self._equations = equations
        self._lifts = lifts

    @property
    def sdequations(self) -> list[SemiDiscreteEquation]:
        return self._equations

    def lifts(self) -> list[SemiDiscreteEquation] | None:
        return self._lifts

    @classmethod
    def using_CG(cls, equation: equations.ConservationLaw
        ) -> 'SemiDiscreteSystem':
        """Create a semi-discrete system of equations using the Continuous
        Galerkin method.

        Given an arbitrary conservation law,

        dU/dt + div F = S

        this constructs the simple CG weak form.

        dq/dt * phi * dx = F . grad(phi) * dx - F . n * phi * ds + S * phi * dx

        Arguments:
            equation (equations.ConservationLaw): The conservation law being
                solved for.
        """
        # Preliminaries
        space = equation.U.function_space
        domain = equation.U.function_space.mesh
        xi = ufl.TrialFunction(space)
        phi = ufl.TestFunction(space)
        n = ufl.FacetNormal(domain)

        # Construct mass form
        bilinear_form = xi * phi * ufl.dx

        stiff_residual_form = None
        non_stiff_residual_form = None

        # Flux contributions
        for flux in equation.fluxes:
            if flux.is_stiff:
                # Volume integral
                stiff_residual_form += ufl.dot(flux.expression,
                    ufl.grad(phi)) * ufl.dx

                # Exterior boundary contribution
                stiff_residual_form += -ufl.conditional(
                    ufl.dot(flux.jacobian_expression, n) > 0.0,
                    ufl.dot(flux.expression, n),
                    dolfinx.fem.Constant(domain, 0.0)
                ) * phi * ufl.ds
            else:
                # Volume integral
                non_stiff_residual_form += ufl.dot(flux.expression,
                    ufl.grad(phi)) * ufl.dx

                # Exterior boundary contribution
                non_stiff_residual_form += -ufl.conditional(
                    ufl.dot(flux.jacobian_expression, n) > 0.0,
                    ufl.dot(flux.expression, n),
                    dolfinx.fem.Constant(domain, 0.0)
                ) * phi * ufl.ds

        # Source contributions
        for source in equation.sources:
            if source.is_stiff:
                stiff_residual_form += source.expression * phi * ufl.dx
            else:
                non_stiff_residual_form += source.expression * phi * ufl.dx

        sd_equation = SemiDiscreteEquation(
            variable=equation.U,
            is_differential=True,
            bilinear_form=bilinear_form,
            stiff_residual_form=stiff_residual_form,
            non_stiff_residual_form=non_stiff_residual_form,
            is_bilinear_form_constant=True
        )
        return cls([sd_equation], None)

    @classmethod
    def using_DG(cls, equation: equations.ConservationLaw,
        trace_function) -> 'SemiDiscreteSystem':
        """Create a semi-discrete system of equations using the Discontinuous
        Galerkin method.

        Given an arbitrary hyperbolic conservation law,

        dU/dt + div F = S

        this constructs the DG weak form.

        dq/dt * phi * dx = F . grad(phi) * dx - trace(F . n) * jump(phi) * dS -
        F . n * phi * ds + S * phi * dx

        Arguments:
            equation (equations.ConservationLaw): The conservation law being
                solved for. This must be hyperbolic.
            trace_function (callable): The flux solver being used at element
                interfaces to resolve discontinuities. The function signature
                should be trace_function(F, U, J, n).
        """
        # Preliminaries
        space = equation.U.function_space
        domain = equation.U.function_space.mesh
        xi = ufl.TrialFunction(space)
        phi = ufl.TestFunction(space)
        n = ufl.FacetNormal(domain)

        # Construct mass form
        bilinear_form = xi * phi * ufl.dx

        # Construct stiff residual form
        stiff_residual_form = None
        non_stiff_residual_form = None
        jacobians = None

        # Flux contributions
        for flux in equation.fluxes:
            if flux.is_stiff:
                # Volume integral
                stiff_residual_form += ufl.dot(flux.expression,
                    ufl.grad(phi)) * ufl.dx

                # Interior boundary contributions
                # Accumulate jacobians for now
                jacobians += flux.jacobian_expression

                # Exterior boundary contribution
                stiff_residual_form += -ufl.conditional(
                    ufl.dot(equation.J_stiff, n) > 0.0,
                    ufl.dot(equation.F_stiff, n),
                    dolfinx.fem.Constant(domain, 0.0)
                ) * phi * ufl.ds
            else:
                # Volume integral
                non_stiff_residual_form += ufl.dot(flux.expression,
                    ufl.grad(phi)) * ufl.dx

                # Interior boundary contributions
                # Accumulate jacobians for now
                jacobians += flux.jacobian_expression

                # Exterior boundary contribution
                non_stiff_residual_form += -ufl.conditional(
                    ufl.dot(equation.J_stiff, n) > 0.0,
                    ufl.dot(equation.F_stiff, n),
                    dolfinx.fem.Constant(domain, 0.0)
                ) * phi * ufl.ds

        # Interior boundary contributions
        for flux in equation.fluxes:
            if flux.is_stiff:
                stiff_residual_form += -trace_function(
                    flux.expression,
                    equation.U,
                    jacobians,
                    n
                ) * ufl.jump(phi) * ufl.dS
            else:
                non_stiff_residual_form += -trace_function(
                    flux.expression,
                    equation.U,
                    jacobians,
                    n
                ) * ufl.jump(phi) * ufl.dS

        # Source contributions
        for source in equation.sources:
            if source.is_stiff:
                stiff_residual_form += source.expression * phi * ufl.dx
            else:
                non_stiff_residual_form += source.expression * phi * ufl.dx

        sd_equation = SemiDiscreteEquation(
            variable=equation.U,
            is_differential=True,
            bilinear_form=bilinear_form,
            stiff_residual_form=stiff_residual_form,
            non_stiff_residual_form=non_stiff_residual_form,
            is_bilinear_form_constant=True
        )
        return cls([sd_equation], None)

    @classmethod
    def using_LDG(cls, equation: equations.ConservationLaw,
        grad_of: ufl.core.expr.Expr,
        space_auxiliary: dolfinx.fem.FunctionSpace,
        trace_function_upwind, trace_function_downwind) -> 'SemiDiscreteSystem':
        """Create a semi-discrete system of equations using the Local Discontinuous
        Galerkin method.

        Given an arbitrary parabolic conservation law,

        dU/dt + div F(U, grad U) = S

        this constructs a DG weak form with an auxiliary variable for a gradient.

        dq/dt * phi * dx = F(q, r) . grad(phi) * dx - trace(F(q, r) . n) *
            jump(phi) * dS - F(q, r) . n * phi * ds + S * phi * dx
        r . psi * dx = - grad_of * div(psi) * dx + trace(grad_of) * jump(psi) . n *
            dS + grad_of * psi . n * ds

        Arguments:
            equation (equations.ConservationLaw): The conservation law being
                solved for.
            grad_of (ufl.core.expr.Expr): The variable whose derivative is lifted
                using an auxiliary variable.
            space_auxiliary (dolfinx.fem.FunctionSpace): The function space used to
                weakly enforce the gradient term.
            trace_function_upwind (callable): The upwind flux solver being used at
                element interfaces to resolve discontinuities. The function
                signature should be trace_function(F, U, J, n).
            trace_function_downwind (callable): The downwind flux solver being used at
                element interfaces to resolve discontinuities. The function
                signature should be trace_function(F, U, J, n).
        """
        # Preliminaries
        space = equation.U.function_space
        domain = equation.U.function_space.mesh
        xi = ufl.TrialFunction(space)
        phi = ufl.TestFunction(space)
        chi = ufl.TrialFunction(space_auxiliary)
        psi = ufl.TestFunction(space_auxiliary)
        g = dolfinx.fem.Function(space_auxiliary) # Auxiliary lifting variable
        n = ufl.FacetNormal(domain)

        # Replace gradient in flux/source with auxiliary variable
        # Leave the ConservationLaw unchanged
        # Pass the result to local variables
        F_stiff = None
        F_non_stiff = None
        J_stiff = None
        J_non_stiff = None
        S_stiff = None
        S_non_stiff = None
        if equation.F_stiff is not None:
            F_stiff = ufl.replace(
                equation.F_stiff,
                {ufl.grad(grad_of): g}
            )
            J_stiff = ufl.replace(
                equation.J_stiff,
                {ufl.grad(grad_of): g}
            )
        if equation.F_non_stiff is not None:
            F_non_stiff = ufl.replace(
                equation.F_non_stiff,
                {ufl.grad(grad_of): g}
            )
            J_non_stiff = ufl.replace(
                equation.J_non_stiff,
                {ufl.grad(grad_of): g}
            )
        if equation.S_stiff is not None:
            S_stiff = ufl.replace(
                equation.S_stiff,
                {ufl.grad(grad_of): g}
            )
        if equation.S_non_stiff is not None:
            S_non_stiff = ufl.replace(
                equation.S_non_stiff,
                {ufl.grad(grad_of): g}
            )

        # Begin with differential equation
        # Construct mass form
        bilinear_form = xi * phi * ufl.dx

        # Construct stiff residual form
        stiff_residual_form = None
        if F_stiff is not None:
            # Volume integral
            stiff_residual_form = ufl.dot(F_stiff,
                ufl.grad(phi)) * ufl.dx

            # Interior boundary contribution
            stiff_residual_form += -trace_function_upwind(
                F_stiff,
                equation.U,
                J_stiff,
                n
            ) * ufl.jump(phi) * ufl.dS

            # Exterior boundary contribution
            stiff_residual_form += -ufl.conditional(
                ufl.dot(J_stiff, n) > 0.0,
                ufl.dot(F_stiff, n),
                dolfinx.fem.Constant(domain, 0.0)
            ) * phi * ufl.ds

        # Stiff source contribution
        # Add to existing or set from None
        if S_stiff is not None:
            if stiff_residual_form is not None:
                stiff_residual_form += S_stiff * phi * ufl.dx
            else:
                stiff_residual_form = S_stiff * phi * ufl.dx

        # Construct non-stiff residual form
        non_stiff_residual_form = None
        if F_non_stiff is not None:
            # Volume integral
            non_stiff_residual_form = ufl.dot(F_non_stiff,
                ufl.grad(phi)) * ufl.dx

            # Interior boundary contribution
            non_stiff_residual_form += -trace_function_upwind(
                F_non_stiff,
                equation.U,
                J_non_stiff,
                n
            ) * ufl.jump(phi) * ufl.dS

            # Exterior boundary contribution
            non_stiff_residual_form += -ufl.conditional(
                ufl.dot(J_non_stiff, n) > 0.0,
                ufl.dot(F_non_stiff, n),
                dolfinx.fem.Constant(domain, 0.0)
            ) * phi * ufl.ds

        # Non-stiff source contribution
        # Add to existing or set from None
        if S_non_stiff is not None:
            if non_stiff_residual_form is not None:
                non_stiff_residual_form += S_non_stiff * phi * ufl.dx
            else:
                non_stiff_residual_form = S_non_stiff * phi * ufl.dx

        differential_equation = SemiDiscreteEquation(
            variable=equation.U,
            is_differential=True,
            bilinear_form=bilinear_form,
            stiff_residual_form=stiff_residual_form,
            non_stiff_residual_form=non_stiff_residual_form,
            is_bilinear_form_constant=True
        )

        # Now create the lift equation
        bilinear_form = ufl.dot(chi, psi) * ufl.dx

        # Construct non-stiff residual only
        # This is what enables efficient computation for explicit solvers
        # If an implicit solver is used, both residuals are solved implicitly
        # Volume integral
        non_stiff_residual_form = -grad_of * ufl.div(psi) * ufl.dx

        # Interior boundary contribution
        if J_stiff is not None:
            non_stiff_residual_form += trace_function_downwind(
                grad_of,
                equation.U,
                equation.J_stiff,
                n
            ) * ufl.jump(psi, n) * ufl.dS
        if J_non_stiff is not None:
            non_stiff_residual_form += trace_function_downwind(
                grad_of,
                equation.U,
                equation.J_non_stiff,
                n
            ) * ufl.jump(psi, n) * ufl.dS

        # Exterior boundary contribution
        non_stiff_residual_form += grad_of * ufl.dot(psi, n) * ufl.ds

        lift_equation = SemiDiscreteEquation(
            variable=g,
            is_differential=False,
            bilinear_form=bilinear_form,
            stiff_residual_form=None,
            non_stiff_residual_form=non_stiff_residual_form,
            is_bilinear_form_constant=True
        )

        return cls([differential_equation], [lift_equation])

# Begin deferred structures
# To be removed

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
