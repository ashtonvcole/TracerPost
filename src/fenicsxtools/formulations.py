"""formulations.py

Different FEM formulation options for conservation laws.
"""

from types import NoneType
from . import equations, bcs
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
        is_bilinear_form_constant (bool): Whether the bilinear form is constant
            in time. This impacts whether matrices need to be re-formed at every
            time step, which is an expensive operation.
        mass_solver_options (dict): PETSc solver options based on the
            structure of the resultant matrix from the function spaces and
            solution method.
    """
    def __init__(self, variable: dolfinx.fem.Function, is_differential: bool,
        bilinear_form: ufl.Form,
        stiff_residual_form: ufl.Form | None = None,
        non_stiff_residual_form: ufl.Form | None = None,
        is_bilinear_form_constant: bool = False,
        mass_solver_options: dict | None = None):
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
            mass_solver_options (dict, optional): PETSc solver options based on
                the structure of the resultant matrix from the function spaces
                and solution method. Default is None.
        """
        self._variable = variable
        self._is_differential = is_differential
        self._bilinear_form = bilinear_form
        self._stiff_residual_form = stiff_residual_form
        self._non_stiff_residual_form = non_stiff_residual_form
        self._is_bilinear_form_constant = is_bilinear_form_constant
        self._mass_solver_options = mass_solver_options or {}

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
    def stiff_residual_form(self) -> ufl.Form | None:
        """ufl.Form: A stiff component of the residual, marked for implicit
        treatment in an IMEX solver. This should be linear in the test function.
        May be None.
        """
        return self._stiff_residual_form

    @property
    def non_stiff_residual_form(self) -> ufl.Form | None:
        """ufl.Form: A non-stiff component of the residual, marked for explicit
        treatment in an IMEX solver. This should be linear in the test function.
        May be None.
        """
        return self._non_stiff_residual_form

    @property
    def is_bilinear_form_constant(self) -> bool:
        """bool: Whether the bilinear form is constant in time. This impacts
        whether matrices need to be re-formed at every time step, which is an
        expensive operation.
        """
        return self._is_bilinear_form_constant

    @property
    def mass_solver_options(self) -> dict:
        """dict: PETSc solver options based on the structure of the resultant
        matrix from the function spaces and solution method.
        """
        return self._mass_solver_options

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
    def __init__(self, sdequations: list[SemiDiscreteEquation],
        lifts: list[SemiDiscreteEquation] | None = None):
        """Constructor.

        Arguments:
            sdequations (list[SemiDiscreteEquation]): A list of differential and
                algebraic equations.
            lifts (list[SemiDiscreteEquation], optional): A list of intermediate
                variable lifts, provided in dependency order. E.g., if the
                equation for c is dependent on a and b, b on a and c, and a on
                c, then the order is a, b. Default is None.
        """
        self._sdequations = sdequations
        self._lifts = lifts

    @property
    def sdequations(self) -> list[SemiDiscreteEquation]:
        return self._sdequations

    @property
    def lifts(self) -> list[SemiDiscreteEquation] | None:
        return self._lifts

    @classmethod
    def using_CG(cls, equation: equations.ConservationLaw,
        boundary_conditions: list[bcs.BoundaryCondition] | None = None,
        ds = None
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
            ds: A partitioned boundary object.
        """
        # Preliminaries
        space = equation.U.function_space
        domain = equation.U.function_space.mesh
        xi = ufl.TrialFunction(space)
        phi = ufl.TestFunction(space)
        n = ufl.FacetNormal(domain)
        boundary_conditions = boundary_conditions or []
        if len(boundary_conditions) > 0 and ds is None:
            raise ValueError('Boundary conditions provided without tagged measure!')
        ds = ds if ds is not None else ufl.ds
        zero = dolfinx.fem.Constant(domain, 0.0)

        # Construct mass form
        bilinear_form = xi * phi * ufl.dx

        # Construct residual forms
        stiff_residual_forms = []
        non_stiff_residual_forms = []

        # Accumulate fluxes for boundary terms
        stiff_fluxes = []
        non_stiff_fluxes = []

        # Test inflow/outflow from net hyperbolic flux
        hyperbolic_fluxes = [
            flux 
            for flux in equation.fluxes 
            if flux.is_hyperbolic
        ]
        Jn_h = ufl.dot(
            sum(flux.jacobian_expression
                for flux in hyperbolic_fluxes),
            n
        ) if len(hyperbolic_fluxes) > 0 else zero

        # Flux contributions
        # Add volume forms and sort boundary terms
        for flux in equation.fluxes:
            residual_form = ufl.dot(flux.expression,
                ufl.grad(phi)) * ufl.dx
            if flux.is_stiff:
                stiff_residual_forms.append(residual_form)
                stiff_fluxes.append(flux.expression)
                stiff_jacobians.append(flux.jacobian_expression)
            else:
                non_stiff_residual_forms.append(residual_form)
                non_stiff_fluxes.append(flux.expression)
                non_stiff_jacobians.append(flux.jacobian_expression)

        # Revisit interior/exterior boundary contributions
        if len(stiff_fluxes) > 0 and len(non_stiff_fluxes) > 0:
            stiff_residual_forms.append(get_CG_boundary_terms(
                F=sum(stiff_fluxes),
                U=equation.U,
                Jn_h=Jn_h,
                n=n,
                phi=phi,
                boundary_conditions=boundary_conditions,
                ds=ds,
                zero=zero,
                apply_flux_conditions=False
            ))
            non_stiff_residual_forms.append(get_CG_boundary_terms(
                F=sum(non_stiff_fluxes),
                U=equation.U,
                Jn_h=Jn_h,
                n=n,
                phi=phi,
                boundary_conditions=boundary_conditions,
                ds=ds,
                zero=zero,
                apply_flux_conditions=True
            ))
        elif len(stiff_fluxes) > 0:
            stiff_residual_forms.append(get_CG_boundary_terms(
                F=sum(stiff_fluxes),
                U=equation.U,
                Jn_h=Jn_h,
                n=n,
                phi=phi,
                boundary_conditions=boundary_conditions,
                ds=ds,
                zero=zero,
                apply_flux_conditions=True
            ))
        elif len(non_stiff_fluxes) > 0:
            non_stiff_residual_forms.append(get_CG_boundary_terms(
                F=sum(non_stiff_fluxes),
                U=equation.U,
                Jn_h=Jn_h,
                n=n,
                phi=phi,
                boundary_conditions=boundary_conditions,
                ds=ds,
                zero=zero,
                apply_flux_conditions=True
            ))

        # Source contributions
        for source in equation.sources:
            residual_form = source.expression * phi * ufl.dx
            if source.is_stiff:
                stiff_residual_forms.append(residual_form)
            else:
                non_stiff_residual_forms.append(residual_form)

        sd_equation = SemiDiscreteEquation(
            variable=equation.U,
            is_differential=True,
            bilinear_form=bilinear_form,
            stiff_residual_form=sum(stiff_residual_forms)
                if len(stiff_residual_forms) != 0
                else None,
            non_stiff_residual_form=sum(non_stiff_residual_forms)
                if len(non_stiff_residual_forms) != 0
                else None,
            is_bilinear_form_constant=True,
            mass_solver_options={
                'ksp_type': 'cg',
                'pc_type': 'jacobi',
                'ksp_rtol': 1e-12
            }
        )
        return cls([sd_equation], None)

    @classmethod
    def using_DG(cls, equation: equations.ConservationLaw,
        trace_function,
        boundary_conditions: list[bcs.BoundaryCondition] | None = None,
        ds = None
        ) -> 'SemiDiscreteSystem':
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
                should be trace_function(F_p, F_m, U_p, U_m, J_p, J_m, n_p).
            boundary_conditions (list[bcs.BoundaryCondition], optional): A list
                of boundary conditions. Default is None.
            ds: A partitioned boundary object.
        """
        # Preliminaries
        space = equation.U.function_space
        domain = equation.U.function_space.mesh
        xi = ufl.TrialFunction(space)
        phi = ufl.TestFunction(space)
        n = ufl.FacetNormal(domain)
        boundary_conditions = boundary_conditions or []
        if len(boundary_conditions) > 0 and ds is None:
            raise ValueError('Boundary conditions provided without tagged measure!')
        ds = ds if ds is not None else ufl.ds
        zero = dolfinx.fem.Constant(domain, 0.0)

        # Construct mass form
        bilinear_form = xi * phi * ufl.dx

        # Construct residual forms
        stiff_residual_forms = []
        non_stiff_residual_forms = []

        # Accumulate fluxes for boundary terms
        stiff_fluxes = []
        stiff_jacobians = []
        non_stiff_fluxes = []
        non_stiff_jacobians = []

        # Test inflow/outflow from net hyperbolic flux
        hyperbolic_fluxes = [
            flux 
            for flux in equation.fluxes 
            if flux.is_hyperbolic
        ]
        Jn_h = ufl.dot(
            sum(flux.jacobian_expression
                for flux in hyperbolic_fluxes),
            n
        ) if len(hyperbolic_fluxes) > 0 else zero

        # Flux contributions
        # Add volume forms and sort boundary terms
        for flux in equation.fluxes:
            residual_form = ufl.dot(flux.expression,
                ufl.grad(phi)) * ufl.dx
            if flux.is_stiff:
                stiff_residual_forms.append(residual_form)
                stiff_fluxes.append(flux.expression)
                stiff_jacobians.append(flux.jacobian_expression)
            else:
                non_stiff_residual_forms.append(residual_form)
                non_stiff_fluxes.append(flux.expression)
                non_stiff_jacobians.append(flux.jacobian_expression)

        # Revisit interior/exterior boundary contributions
        # For consistency, stiff residual gets trace of stiff fluxes
        # Non-stiff gets the difference from the total
        total_boundary_form = get_DG_boundary_terms(
            F=sum(stiff_fluxes + non_stiff_fluxes),
            U=equation.U,
            J=sum(stiff_jacobians + non_stiff_jacobians),
            Jn_h=Jn_h,
            n=n,
            trace_function=trace_function,
            phi=phi,
            boundary_conditions=boundary_conditions,
            ds=ds,
            zero=zero,
            apply_flux_conditions=True
        )
        if len(stiff_fluxes) > 0 and len(non_stiff_fluxes) > 0:
            stiff_boundary_form = get_DG_boundary_terms(
                F=sum(stiff_fluxes),
                U=equation.U,
                J=sum(stiff_jacobians),
                Jn_h=Jn_h,
                n=n,
                trace_function=trace_function,
                phi=phi,
                boundary_conditions=boundary_conditions,
                ds=ds,
                zero=zero,
                apply_flux_conditions=False
            )
            stiff_residual_forms.append(stiff_boundary_form)
            non_stiff_residual_forms.append(total_boundary_form - stiff_boundary_form)
        elif len(stiff_fluxes) > 0:
            stiff_residual_forms.append(total_boundary_form)
        elif len(non_stiff_fluxes) > 0:
            non_stiff_residual_forms.append(total_boundary_form)

        # Source contributions
        for source in equation.sources:
            residual_form = source.expression * phi * ufl.dx
            if source.is_stiff:
                stiff_residual_forms.append(residual_form)
            else:
                non_stiff_residual_forms.append(residual_form)

        sd_equation = SemiDiscreteEquation(
            variable=equation.U,
            is_differential=True,
            bilinear_form=bilinear_form,
            stiff_residual_form=sum(stiff_residual_forms)
                if len(stiff_residual_forms) != 0
                else None,
            non_stiff_residual_form=sum(non_stiff_residual_forms)
                if len(non_stiff_residual_forms) != 0
                else None,
            is_bilinear_form_constant=True,
            mass_solver_options={
                'ksp_type': 'preonly',
                'pc_type': 'bjacobi',
                'sub_ksp_type': 'preonly',
                'sub_pc_type': 'lu'
            }
        )
        return cls([sd_equation], None)

    @classmethod
    def using_LDG(cls, equation: equations.ConservationLaw,
        grad_of: ufl.core.expr.Expr,
        space_auxiliary: dolfinx.fem.FunctionSpace,
        trace_function, beta: float) -> 'SemiDiscreteSystem':
        """Create a semi-discrete system of equations using the Local Discontinuous
        Galerkin method.

        Given an arbitrary parabolic conservation law,

        dU/dt + div F(U, grad U) = S

        this constructs a DG weak form with an auxiliary variable for a gradient.

        dq/dt * phi * dx = F(q, r) . grad(phi) * dx - trace(F(q, r) . n) *
            jump(phi) * dS - F(q, r) . n * phi * ds + S * phi * dx
        r . psi * dx = - grad_of * div(psi) * dx + trace(grad_of) * jump(psi) . n *
            dS + grad_of * psi . n * ds

        Here, trace refers both to hyperbolic flux solvers and parabolic alternation.

        Arguments:
            equation (equations.ConservationLaw): The conservation law being
                solved for.
            grad_of (ufl.core.expr.Expr): The variable whose derivative is lifted
                using an auxiliary variable.
            space_auxiliary (dolfinx.fem.FunctionSpace): The function space used to
                weakly enforce the gradient term.
            trace_function (callable): The flux solver being used at element
                interfaces to resolve discontinuities. The function signature
                should be trace_function(F, U, J, n).
            beta (float): A constant used for LDG flux alternation.
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

        # Begin with differential equation
        # Construct mass form
        bilinear_form = xi * phi * ufl.dx

        # Construct residual forms
        stiff_residual_forms = []
        non_stiff_residual_forms = []

        # Non-alternating fluxes are accumulated
        stiff_hyperbolic_fluxes = []
        stiff_hyperbolic_jacobians = []
        non_stiff_hyperbolic_fluxes = []
        non_stiff_hyperbolic_jacobians = []

        # Flux contributions
        for flux in equation.fluxes:
            # Replace gradient in flux/source with auxiliary variable
            # Leave the ConservationLaw unchanged
            F = ufl.replace(
                flux.expression,
                {ufl.grad(grad_of): g}
            )
            J = ufl.replace(
                flux.jacobian_expression,
                {ufl.grad(grad_of): g}
            )

            # Sanity check
            # Make sure parabolic terms get replaced appropriately
            if not flux.is_hyperbolic:
               assert g in ufl.algorithms.extract_coefficients(F)

            # Volume integral
            residual_form = ufl.dot(F,
                ufl.grad(phi)) * ufl.dx

            # Interior boundary contribution
            if flux.is_hyperbolic:
                # Use normal flux solver
                if flux.is_stiff:
                    stiff_hyperbolic_fluxes.append(F)
                    stiff_hyperbolic_jacobians.append(J)
                else:
                    non_stiff_hyperbolic_fluxes.append(F)
                    non_stiff_hyperbolic_jacobians.append(J)
            else:
                # Use LDG alternation
                residual_form += -(
                    ufl.dot(ufl.avg(F), n('+'))
                    + beta * ufl.jump(F, n)
                ) * ufl.jump(phi) * ufl.dS

            # Exterior boundary contribution
            if flux.is_hyperbolic:
                residual_form += -ufl.conditional(
                    ufl.dot(J, n) > 0.0,
                    ufl.dot(F, n),
                    dolfinx.fem.Constant(domain, 0.0)
                ) * phi * ufl.ds
            else:
                # Neglect parabolic terms on the boundary
                pass

            # Add the individual flux to the correct residual
            if flux.is_stiff:
                stiff_residual_forms.append(residual_form)
            else:
                non_stiff_residual_forms.append(residual_form)

        # Revisit interior boundary contributions
        if len(stiff_hyperbolic_fluxes) > 0:
            F = sum(stiff_hyperbolic_fluxes)
            J = sum(stiff_hyperbolic_jacobians)
            residual_form = -trace_function(
                F,
                equation.U,
                J,
                n
            ) * ufl.jump(phi) * ufl.dS
            stiff_residual_forms.append(residual_form)
        if len(non_stiff_hyperbolic_fluxes) > 0:
            F = sum(non_stiff_hyperbolic_fluxes)
            J = sum(non_stiff_hyperbolic_jacobians)
            residual_form = -trace_function(
                F,
                equation.U,
                J,
                n
            ) * ufl.jump(phi) * ufl.dS
            non_stiff_residual_forms.append(residual_form)

        # Source contributions
        for source in equation.sources:
            # Replace gradient in flux/source with auxiliary variable
            # Leave the ConservationLaw unchanged
            S = ufl.replace(
                source.expression,
                {ufl.grad(grad_of): g}
            )
            residual_form = S * phi * ufl.dx
            if source.is_stiff:
                stiff_residual_forms.append(residual_form)
            else:
                non_stiff_residual_forms.append(residual_form)

        differential_equation = SemiDiscreteEquation(
            variable=equation.U,
            is_differential=True,
            bilinear_form=bilinear_form,
            stiff_residual_form=sum(stiff_residual_forms)
                if len(stiff_residual_forms) != 0
                else None,
            non_stiff_residual_form=sum(non_stiff_residual_forms)
                if len(non_stiff_residual_forms) != 0
                else None,
            is_bilinear_form_constant=True,
            mass_solver_options={
                'ksp_type': 'preonly',
                'pc_type': 'bjacobi',
                'sub_ksp_type': 'preonly',
                'sub_pc_type': 'lu'
            }
        )

        # Now create the lift equation
        # Construct mass form
        bilinear_form = ufl.dot(chi, psi) * ufl.dx

        # Construct non-stiff residual only
        # This is what enables efficient computation for explicit solvers
        # If an implicit solver is used, both residuals are solved implicitly
        # Volume integral
        non_stiff_residual_form = -grad_of * ufl.div(psi) * ufl.dx

        # Interior boundary contribution
        non_stiff_residual_form += (
            ufl.avg(grad_of)
            - beta * ufl.jump(grad_of)
        ) * ufl.jump(psi, n) * ufl.dS

        # Exterior boundary contribution
        non_stiff_residual_form += grad_of * ufl.dot(psi, n) * ufl.ds

        lift_equation = SemiDiscreteEquation(
            variable=g,
            is_differential=False,
            bilinear_form=bilinear_form,
            stiff_residual_form=None,
            non_stiff_residual_form=non_stiff_residual_form,
            is_bilinear_form_constant=True,
            mass_solver_options={
                'ksp_type': 'preonly',
                'pc_type': 'bjacobi',
                'sub_ksp_type': 'preonly',
                'sub_pc_type': 'lu'
            }
        )

        return cls([differential_equation], [lift_equation])

def get_CG_boundary_terms(F, U, Jn_h, n, phi, boundary_conditions, ds, zero, apply_flux_conditions):
    """Get CG boundary integral terms for a flux.
    
    Arguments:
        F: The flux.
        U: The state variable.
        Jn_h: The sum of Jacobians of hyperbolic fluxes dotted with the normal,
            i.e., the characteristic speed.
        n: The normal vector.
        phi: The test function complementary to U.
        boundary_conditions: A list of boundary conditions to apply at inflows.
        ds: A partitioned boundary object.
        zero: A constant zero object.
        apply_flux_conditions: Whether to apply prescribed-flux conditions.
    """
    # Add open boundary contribution at outflows
    residual_form = -ufl.conditional(
        Jn_h > 0.0, # Outflow
        ufl.dot(F, n),
        zero
    ) * phi * ds

    # Add Dirichlet conditions at inflows
    # Only apply flux conditions if requested, to avoid double-counting
    for bc in boundary_conditions:
        if isinstance(bc, bcs.DirichletCondition):
            residual_form += -ufl.conditional(
                Jn_h < 0.0, # Inflow
                ufl.dot(ufl.replace(F, {U: bc.value}), n),
                zero
            ) * phi * ds(bc.tag)
        if isinstance(bc, bcs.FluxCondition) and apply_flux_conditions:
            residual_form += -ufl.conditional(
                Jn_h < 0.0, # Inflow
                ufl.dot(bc.value, n),
                zero
            ) * phi * ds(bc.tag)
    return residual_form

def get_DG_boundary_terms(F, U, J, Jn_h, n, trace_function, phi, boundary_conditions, ds, zero, apply_flux_conditions):
    """Get DG interior and exterior boundary integral terms for a flux.
    
    Arguments:
        F: The flux.
        U: The state variable.
        J: The Jacobian of F. If using UFL differentiation, this must pass
            through expand_derivatives.
        Jn_h: The sum of Jacobians of hyperbolic fluxes dotted with the normal,
            i.e., the characteristic speed.
        n: The normal vector.
        trace_function (callable): The flux solver being used at element
            interfaces to resolve discontinuities.
        phi: The test function complementary to U.
        boundary_conditions: A list of boundary conditions to apply at inflows.
        ds: A partitioned boundary object.
        zero: A constant zero object.
        apply_flux_conditions: Whether to apply prescribed-flux conditions.
    """
    # Add interior contribution
    residual_form = -trace_function(
        F('+'), F('-'), U('+'), U('-'),
        J('+'), J('-'), n('+')
    ) * ufl.jump(phi) * ufl.dS

    # Add open boundary contribution at outflows
    residual_form += -ufl.conditional(
        Jn_h > 0.0, # Outflow
        ufl.dot(F, n),
        zero
    ) * phi * ds

    # Add Dirichlet conditions at inflows
    # Only apply flux conditions if requested, to avoid double-counting
    for bc in boundary_conditions:
        if isinstance(bc, bcs.DirichletCondition):
            residual_form += -ufl.conditional(
                Jn_h < 0.0, # Inflow
                trace_function(
                    F, ufl.replace(F, {U: bc.value}),
                    U, bc.value,
                    J, ufl.replace(J, {U: bc.value}),
                    n
                ),
                zero
            ) * phi * ds(bc.tag)
        if isinstance(bc, bcs.FluxCondition) and apply_flux_conditions:
            residual_form += -ufl.conditional(
                Jn_h < 0.0, # Inflow
                ufl.dot(bc.value, n),
                zero
            ) * phi * ds(bc.tag)
    return residual_form

# SUPG, SIPG, etc?
