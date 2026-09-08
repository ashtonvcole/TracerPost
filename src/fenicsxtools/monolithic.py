import dolfinx as dx
import dolfinx.fem.petsc

def monolithic_DG_solver(formulation: DGFormulation, params: dict):
    """Monolithic DG solver for scalar hyperbolic conservation laws.

    Currently restricted to forward Euler.
    """

    # Create LHS
    M = dx.fem.petsc.assemble_matrix(formulation.LHS_form)
    M.assemble()
    M_solver = PETSc.KSP().create(params['comm'])
    M_solver.setOperators(M)
    M_solver.setType(PETSc.KSP.Type.PREONLY)
    M_solver.getPC().setType(PETSc.PC.Type.LU)
    M_solver.setFromOptions() # Allows command-line PETSc options to override the above
    M_solver.setUp()

    # Create RHS
    b = dx.fem.petsc.create_vector(params['V'])
    n_owned = (
        params['V'].dofmap.index_map.size_local
        * params['V'].dofmap.index_map_bs
    )

    # Solution loop
    t = 0.0
    for step in range(params['nt']):
        # Update time
        t = step * params['dt']

        # Update arrays
        with b.local_form as local:
            local.set(0.0)
        dx.fem.assemble_vector(b, RHS_form)

        # Note: in certain cases, LHS would be updated/re-assembled as well

        # Solve for time derivative
        # Same essence for all explicit?
        # Forward Euler update
        # Somewhat tied to time derivative solve
        # Writing
        if step % params['write_every'] == 0:
            pass

    pass
