"""solvers,py

Turn semidiscrete ODE/DAE systems into PETSc TS solver objects.
"""

import dolfinx
import dolfinx.fem.petsc
from fenicsxtools import formulations
import petsc4py.PETSc

class FormedEquation:
    """Lightweight wrapper for a compiled semi-discrete equation.

    Attributes:
        variable (dolfinx.fem.Function): The equation's state.
        scratch_function (dolfinx.fem.Function): Scratch function for
            holding results of a particular stage.
        use_mass_matrix (bool): Whether a mass matrix was formed.
        use_mass_solver (bool): Whether the mass matrix is inverted by a solver.
        bilinear_form (dolfinx.fem.Form): The compiled bilinear mass form.
        mass_matrix (petsc4py.PETSc.Mat): The mass matrix. May be None if
            use_mass_matrix is False.
        mass_solver (petsc4py.PETSc.KSP): The mass solver. May be Nonw if
            use_mass_solver is False.
        split_residual (bool): Whether the residual is split into separately
            compiled stiff and non-stiff forms.
        stiff_residual_form (dolfinx.fem.Form): The compiled stiff residual
            form. None if the equation has no stiff residual or split_residual
            is False.
        stiff_residual_vector (petsc4py.PETSc.Vec): A scratch vector for the
            stiff residual. May be None if the equation has no stiff residual,
            or split is False.
        non_stiff_residual_form (dolfinx.fem.Form): A compiled non-stiff
            residual form. May be None if the equation has no stiff residual, or
            split is False.
        non_stiff_residual_vector (petsc4py.PETSc.Vec): A scratch vector for the
            non-stiff residual. May be None if the equation has no non-stiff
            residual, or split is False.
        combined_residual_form (dolfinx.fem.Form): A compiled stiff residual
            form. May be None if the equation has no residuals, or split is True.
        combined_residual_vector (petsc4py.PETSc.Vec): A compiled stiff residual
            form. May be None if the equation has no residuals, or split is
            True.
    """

    def __init__(self, sdequation: formulations.SemiDiscreteEquation,
        use_mass_matrix: bool, use_mass_solver: bool,
        split_residual: bool):
        """Constructor.

        Arguments:
            sdequation (formulations.SemiDiscreteEquation): The algebraic or
                differential equation component.
            use_mass_matrix (bool): Whether to form a mass matrix.
            use_mass_solver (bool): Whether the mass matrix needs to be
                inverted.
            split_residual (bool): Whether to compile separate residuals for
                stiff and non-stiff components.
        """
        # Record the sdequation variable
        self._variable = sdequation.variable

        # Set up scratch function for holding stages
        self._scratch_function = dolfinx.fem.Function(
            sdequation.variable.function_space
        )

        # Set up the mass solver
        self._use_mass_matrix = use_mass_matrix
        self._use_mass_solver = use_mass_solver
        self._bilinear_form = None
        self._mass_matrix = None
        self._mass_solver = None

        # Check for non-constant mass form
        # Raise a not implemented error for now
        if not sdequation.is_bilinear_form_constant:
            raise NotImplementedError("Variable mass matrices not supported.")

        # If desired, set up a mass matrix
        if use_mass_matrix:
            self._bilinear_form = dolfinx.fem.form(sdequation.bilinear_form)
            self._mass_matrix = dolfinx.fem.petsc.assemble_matrix(self._bilinear_form)
            self._mass_matrix.assemble()

        # If desired, set up a mass solver
        if use_mass_solver:
            if not use_mass_matrix:
                raise ValueError("Mass matrix construction is required for solve.")
            self._mass_solver = petsc4py.PETSc.KSP().create(
                sdequation.variable.function_space.mesh.comm
            )
            self._mass_solver.setOperators(self._mass_matrix)
            self._mass_solver.setType(petsc4py.PETSc.KSP.Type.PREONLY)
            self._mass_solver.getPC().setType(petsc4py.PETSc.PC.Type.LU)
            self._mass_solver.setFromOptions() # Allows command-line PETSc options to override the above
            self._mass_solver.setUp()

        # If desired, set up a mass linear form
        # Not yet implemented

        # Set up the residual form(s) and vector(s) to hold their value(s)
        self._split_residual = split_residual
        self._stiff_residual_form = None
        self._stiff_residual_vector = None
        self._non_stiff_residual_form = None
        self._non_stiff_residual_vector = None
        self._combined_residual_form = None
        self._combined_residual_vector = None
        if split_residual:
            # Maintain two separate residual forms
            # Combined residual is None, and will trigger errors if used
            if sdequation.stiff_residual_form is not None:
                self._stiff_residual_form = dolfinx.fem.form(
                    sdequation.stiff_residual_form
                )
                self._stiff_residual_vector = dolfinx.fem.petsc.create_vector(
                    sdequation.variable.function_space
                )
            if sdequation.non_stiff_residual_form is not None:
                self._non_stiff_residual_form = dolfinx.fem.form(
                    sdequation.non_stiff_residual_form
                )
                self._non_stiff_residual_vector = dolfinx.fem.petsc.create_vector(
                    sdequation.variable.function_space
                )
        else:
            # Combine into one residual form
            # Split residuals are None, and will trigger errors if used
            if (
                sdequation.stiff_residual_form is not None and
                sdequation.non_stiff_residual_form is not None
            ):
                self._combined_residual_form = dolfinx.fem.form(
                    sdequation.stiff_residual_form +
                    sdequation.non_stiff_residual_form
                )
            elif sdequation.stiff_residual_form is not None:
                self._combined_residual_form = dolfinx.fem.form(
                    sdequation.stiff_residual_form
                )
            elif sdequation.non_stiff_residual_form is not None:
                self._combined_residual_form = dolfinx.fem.form(
                    sdequation.non_stiff_residual_form
                )
            if self._combined_residual_form is not None:
                self._combined_residual_vector = dolfinx.fem.petsc.create_vector(
                    sdequation.variable.function_space
                )

    @property
    def variable(self) -> dolfinx.fem.Function:
        """petsc4py.PETSc.Vec: The equation's PETSc state vector."""
        return self._variable

    @property
    def scratch_function(self) -> dolfinx.fem.Function:
        """dolfinx.fem.Function: Scratch function for holding results of a
        particular stage."""
        return self._scratch_function

    @property
    def use_mass_matrix(self) -> bool:
        """bool: Whether a mass matrix was formed."""
        return self._use_mass_matrix

    @property
    def use_mass_solver(self) -> bool:
        """bool: Whether the mass matrix is inverted by a solver."""
        return self._use_mass_solver

    @property
    def bilinear_form(self) -> dolfinx.fem.Form:
        """dolfinx.fem.Form: The compiled bilinear mass form.

        Raises:
            RuntimeError: If use_mass_matrix is False.
        """
        if not self._use_mass_matrix:
            raise RuntimeError(
                "bilinear_form is unavailable because use_mass_matrix is False."
            )
        return self._bilinear_form

    @property
    def mass_matrix(self) -> petsc4py.PETSc.Mat:
        """petsc4py.PETSc.Mat: The mass matrix.

        Raises:
            RuntimeError: If use_mass_matrix is False.
        """
        if not self._use_mass_matrix:
            raise RuntimeError(
                "mass_matrix is unavailable because use_mass_matrix is False."
            )
        return self._mass_matrix

    @property
    def mass_solver(self) -> petsc4py.PETSc.KSP:
        """petsc4py.PETSc.KSP: The mass solver.

        Raises:
            RuntimeError: If use_mass_solver is False.
        """
        if not self._use_mass_solver:
            raise RuntimeError(
                "mass_solver is unavailable because use_mass_solver is False."
            )
        return self._mass_solver

    @property
    def split_residual(self) -> bool:
        """bool: Whether the residual is split into separately compiled stiff
        and non-stiff forms.
        """
        return self._split_residual

    @property
    def stiff_residual_form(self) -> dolfinx.fem.Form | None:
        """dolfinx.fem.Form | None: The compiled stiff residual form. None if
        the equation has no stiff residual.

        Raises:
            RuntimeError: If split_residual is False.
        """
        if not self._split_residual:
            raise RuntimeError(
                "stiff_residual_form is unavailable because split_residual is "
                "False. Use combined_residual_form."
            )
        return self._stiff_residual_form

    @property
    def stiff_residual_vector(self) -> petsc4py.PETSc.Vec | None:
        """petsc4py.PETSc.Vec | None: A scratch vector for the stiff residual.
        None if the equation has no stiff residual.

        Raises:
            RuntimeError: If split_residual is False.
        """
        if not self._split_residual:
            raise RuntimeError(
                "stiff_residual_vector is unavailable because split_residual "
                "is False. Use combined_residual_vector."
            )
        return self._stiff_residual_vector

    @property
    def non_stiff_residual_form(self) -> dolfinx.fem.Form | None:
        """dolfinx.fem.Form | None: The compiled non-stiff residual form. None
        if the equation has no non-stiff residual.

        Raises:
            RuntimeError: If split_residual is False.
        """
        if not self._split_residual:
            raise RuntimeError(
                "non_stiff_residual_form is unavailable because split_residual "
                "is False. Use combined_residual_form."
            )
        return self._non_stiff_residual_form

    @property
    def non_stiff_residual_vector(self) -> petsc4py.PETSc.Vec | None:
        """petsc4py.PETSc.Vec | None: A scratch vector for the non-stiff
        residual. None if the equation has no non-stiff residual.

        Raises:
            RuntimeError: If split_residual is False.
        """
        if not self._split_residual:
            raise RuntimeError(
                "non_stiff_residual_vector is unavailable because "
                "split_residual is False. Use combined_residual_vector."
            )
        return self._non_stiff_residual_vector

    @property
    def combined_residual_form(self) -> dolfinx.fem.Form | None:
        """dolfinx.fem.Form | None: The compiled sum of the stiff and non-stiff
        residuals. None if the equation has no residuals.

        Raises:
            RuntimeError: If split_residual is True.
        """
        if self._split_residual:
            raise RuntimeError(
                "combined_residual_form is unavailable because split_residual "
                "is True. Use the stiff and non-stiff residual forms."
            )
        return self._combined_residual_form

    @property
    def combined_residual_vector(self) -> petsc4py.PETSc.Vec | None:
        """petsc4py.PETSc.Vec | None: A scratch vector for the combined
        residual. None if the equation has no residuals.

        Raises:
            RuntimeError: If split_residual is True.
        """
        if self._split_residual:
            raise RuntimeError(
                "combined_residual_vector is unavailable because "
                "split_residual is True. Use the stiff and non-stiff residual "
                "vectors."
            )
        return self._combined_residual_vector

    def update_stiff_residual(self):
        """Update the stiff residual scratch vector with present function
        values, if not None.

        Arguments:
            None

        Raises:
            RuntimeError: If split_residual is False.
        """
        if not self._split_residual:
            raise RuntimeError(
                "update_stiff_residual is unavailable because "
                "split_residual is False. Use update_combined_residual."
            )

        if (
            self._stiff_residual_form is not None and
            self._stiff_residual_vector is not None
        ):
            # Flush entries to zero
            with self._stiff_residual_vector.localForm() as residual_local:
                residual_local.set(0.0)

            # Re-assemble with updated function values
            dolfinx.fem.petsc.assemble_vector(
                self._stiff_residual_vector,
                self._stiff_residual_form
            )

            # Pass the updates to ghost entries
            self._stiff_residual_vector.ghostUpdate(
                addv=petsc4py.PETSc.InsertMode.ADD_VALUES,
                mode=petsc4py.PETSc.ScatterMode.REVERSE
            )

    def update_non_stiff_residual(self):
        """Update the non-stiff residual scratch vector with present function
        values, if not None.

        Arguments:
            None

        Raises:
            RuntimeError: If split_residual is False.
        """
        if not self._split_residual:
            raise RuntimeError(
                "update_non_stiff_residual is unavailable because "
                "split_residual is False. Use update_combined_residual."
            )

        if (
            self._non_stiff_residual_form is not None and
            self._non_stiff_residual_vector is not None
        ):
            # Flush entries to zero
            with self._non_stiff_residual_vector.localForm() as residual_local:
                residual_local.set(0.0)

            # Re-assemble with updated function values
            dolfinx.fem.petsc.assemble_vector(
                self._non_stiff_residual_vector,
                self._non_stiff_residual_form
            )

            # Pass the updates to ghost entries
            self._non_stiff_residual_vector.ghostUpdate(
                addv=petsc4py.PETSc.InsertMode.ADD_VALUES,
                mode=petsc4py.PETSc.ScatterMode.REVERSE
            )

    def update_combined_residual(self):
        """Update the combined residual scratch vector with present function
        values, if not None.

        Arguments:
            None

        Raises:
            RuntimeError: If split_residual is True.
        """
        if self._split_residual:
            raise RuntimeError(
                "update_combined_residual is unavailable because "
                "split_residual is True."
            )

        if (
            self._combined_residual_form is not None and
            self._combined_residual_vector is not None
        ):
            # Flush entries to zero
            with self._combined_residual_vector.localForm() as residual_local:
                residual_local.set(0.0)

            # Re-assemble with updated function values
            dolfinx.fem.petsc.assemble_vector(
                self._combined_residual_vector,
                self._combined_residual_form
            )

            # Pass the updates to ghost entries
            self._combined_residual_vector.ghostUpdate(
                addv=petsc4py.PETSc.InsertMode.ADD_VALUES,
                mode=petsc4py.PETSc.ScatterMode.REVERSE
            )

def _get_rhs_explicit(states: list[dolfinx.fem.Function],
    equations: list[FormedEquation],
    lifts: list[FormedEquation]):
    def rhs(ts, t, u, G):
        """Right-hand side G of the general TS ODE.

        F(t, u, du/dt) = G(t, u)

        Arguments:
            ts: A PETSc time stepper object.
            t: The current time.
            u: The PETSc state vector.
            G: The PETSc residual value.

        Returns:
            None: The function sets a new value for G.
        """
        # Update state variables
        dolfinx.fem.petsc.assign(u, states)
        for state in states:
            state.x.scatter_forward()

        # Sequentially solve variable lifts
        for lift in lifts:
            if (
                lift.combined_residual_form is not None and
                lift.combined_residual_vector is not None
            ):
                # Update the residual
                lift.update_combined_residual()
                # Solve for the lift variable
                lift.mass_solver.solve(
                    lift.combined_residual_vector,
                    lift.variable.x.petsc_vec
                )
                lift.variable.x.petsc_vec.ghostUpdate(
                    addv=petsc4py.PETSc.InsertMode.INSERT,
                    mode=petsc4py.PETSc.ScatterMode.FORWARD
                )
            else:
                # No RHS
                pass
        # Solve the RHS of equations
        for equation in equations:
            if (
                equation.combined_residual_form is not None and
                equation.combined_residual_vector is not None
            ):
                # Update residual
                equation.update_combined_residual()
                # Solve for the RHS in a scratch variable
                equation.mass_solver.solve(
                    equation.combined_residual_vector,
                    equation.scratch_function.x.petsc_vec
                )
            else:
                equation.scratch_function.x.array[:] = 0.0
            # Ghost update not needed
            # equation.scratch_function.x.petsc_vec.ghostUpdate(
            #     addv=petsc4py.PETSc.InsertMode.INSERT,
            #     mode=petsc4py.PETSc.ScatterMode.FORWARD
            # )
        # Put the result in G
        dolfinx.fem.petsc.assign(
            [equation.scratch_function for equation in equations],
            G
        )
    return rhs

class TSSolver:
    """Lightweight wrapper for a TS solver.

    Attributes:
        ts (petsc4py.PETSc.TS): The ODE integrator object.
    """

    def __init__(self, ts: petsc4py.PETSc.TS):
        """Constructor.

        Arguments:
            ts (petsc4py.PETSc.TS): The ODE integrator object.
        """
        self._ts = ts

    @property
    def ts(self) -> petsc4py.PETSc.TS:
        """petsc4py.PETSc.TS: The ODE integrator object."""
        return self._ts

    @classmethod
    def from_SemiDiscreteSystem(cls, system: formulations.SemiDiscreteSystem,
        method: str = 'imex') -> 'TSSolver':
        """Construct a TS solver from a semidiscrete system of equations.

        Arguments:
            system (formulations.SemiDiscreteSystem): The system of semidiscrete
                equations.
            method (str, optional): The handling of stiff and non-stiff
                residuals. Accepted values are "explicit", "implicit", and
                "imex", Default is "imex".
        """
        # Need to decide whether to register rhs and lhs
        # Need better handling of communicators
        ts = petsc4py.PETSc.TS().create(
            system.sdequations[0].variable.function_space.mesh.comm
        )

        # Get combined state vector
        # Copy over initial condition
        states = [equation.variable for equation in system.sdequations]
        x = dolfinx.fem.petsc.create_vector(
            [state.function_space for state in states], kind="mpi")
        dolfinx.fem.petsc.assign(states, x)
        ts.setSolution(x)

        # Set up lift equations
        # Always the same
        lifts = [
            FormedEquation(
                sdequation=lift,
                use_mass_matrix=True, # Always
                use_mass_solver=True, # Always
                split_residual=False # Never
            )
            for lift in system.lifts
        ] if system.lifts is not None else []

        # Convert ufl forms into dolfinx/PETSc equations
        if method == 'explicit':
            # Set up differential equations
            # Non-lift algebraic constraints not allowed
            equations = []
            for sdequation in system.sdequations:
                if not sdequation.is_differential:
                    raise ValueError('Explicit solution is restricted to differential equations only.')
                equations.append(FormedEquation(
                    sdequation=sdequation,
                    use_mass_matrix=True, # For solver
                    use_mass_solver=True, # For RHS
                    split_residual=False # Only for IMEX
                ))
            # Register system of equations within ts
            ts.setRHSFunction(
                _get_rhs_explicit(states, equations, lifts),
                x.duplicate()
            )
        elif method == 'implicit':
            # Set up differential equations
            equations = [
                FormedEquation(
                    sdequation=sdequation,
                    use_mass_matrix=True, # For Jacobian
                    use_mass_solver=False, # Not needed
                    split_residual=False # Only for IMEX
                )
                for sdequation in system.sdequations
            ]
            raise NotImplementedError
        elif method == 'imex':
            # Set up differential equations
            equations = [
                FormedEquation(
                    sdequation=sdequation,
                    use_mass_matrix=True, # For Jacobian
                    use_mass_solver=True, # For RHS
                    split_residual=True # Only for IMEX
                )
                for sdequation in system.sdequations
            ]
            raise NotImplementedError
        else:
            raise ValueError
        return cls(ts)
