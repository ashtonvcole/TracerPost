"""tracer_post.py

Run script for passive tracer post-processing of ADCIRC/DGSWEM output. Solves
the depth-averaged advection equation for iota = c h with a DG discretization,
using nodal velocity data from a fort.64 file.

Diffusion is not yet implemented.
"""

import argparse
import os

import adcircxtools as at
import adios4dolfinx as adx
import dolfinx as dx
import dolfinx.fem.petsc
import dolfinx.io
import fenicsxtools as ft
import matplotlib.tri
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
import scipy.spatial


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Advect a passive tracer with ADCIRC/DGSWEM velocities.'
    )

    g = p.add_argument_group('inputs')
    g.add_argument('--mesh', required=True,
        help='Filtered mesh in adios4dolfinx format (.bp), in lon/lat degrees.')
    g.add_argument('--interp-dir', required=True,
        help='Directory containing interp_coordinates.npy, interp_elements.npy,'
             ' and interp_node_map.npy for the interpolation mesh.')
    g.add_argument('--fort64', required=True,
        help='ADCIRC velocity time series (fort.64).')
    g.add_argument('--fort63', default=None,
        help='ADCIRC elevation time series (fort.63). Not needed for '
             'advection only; reserved for diffusion.')

    g = p.add_argument_group('projection')
    g.add_argument('--radius-earth', type=float, default=6378206.4,
        help='Earth radius for CPP [m] (default: %(default)s).')
    g.add_argument('--cpp-lon', type=float, default=-94.5,
        help='CPP reference longitude [deg] (default: %(default)s).')
    g.add_argument('--cpp-lat', type=float, default=29.0,
        help='CPP reference latitude [deg] (default: %(default)s).')

    g = p.add_argument_group('initial condition')
    g.add_argument('--ic-lon', type=float, default=-97.0611,
        help='Gaussian center longitude [deg] (default: %(default)s).')
    g.add_argument('--ic-lat', type=float, default=27.8339,
        help='Gaussian center latitude [deg] (default: %(default)s).')
    g.add_argument('--ic-radius', type=float, default=2000.0,
        help='Gaussian standard deviation [m] (default: %(default)s).')
    g.add_argument('--ic-amplitude', type=float, default=1.0,
        help='Gaussian peak value of iota (default: %(default)s).')

    g = p.add_argument_group('physics')
    g.add_argument('--velocity-scale', type=float, default=1.0,
        help='Multiplier applied to the velocity field (default: %(default)s).')
    g.add_argument('--diffusivity', type=float, default=0.0,
        help='Diffusivity. Not implemented; must be 0.')

    g = p.add_argument_group('time stepping')
    g.add_argument('--t-final', type=float, required=True,
        help='Final simulation time [s].')
    g.add_argument('--dt', type=float, default=1.0,
        help='Time step [s] (default: %(default)s).')
    g.add_argument('--ssp-type', default='rks3',
        help='PETSc SSP Runge-Kutta type (default: %(default)s).')

    g = p.add_argument_group('output')
    g.add_argument('--output', default='tracer.bp',
        help='Output VTX/ADIOS2 path (default: %(default)s).')
    g.add_argument('--write-every', type=int, default=600,
        help='Write a snapshot every N steps (default: %(default)s).')

    args = p.parse_args()
    if args.diffusivity != 0.0:
        p.error('diffusion is not implemented; --diffusivity must be 0')
    return args


class VelocityField:
    """Linear-in-time, linear-in-space interpolation of fort.64 data onto a
    DG vector function. Holds a pair of snapshots (a, b) bracketing the
    current time.
    """

    def __init__(self, W, comm, buffer, triangulation, kd, node_mask, scale):
        self.comm = comm
        self.buffer = buffer  # Only open on rank 0; None elsewhere
        self.triangulation = triangulation
        self.kd = kd
        self.node_mask = node_mask
        self.scale = scale

        self.va = dx.fem.Function(W)
        self.vb = dx.fem.Function(W)
        self.v = dx.fem.Function(W)  # Time-interpolated field

        # Field at time 0 is zero (ADCIRC cold start)
        self.va.x.array[:] = 0.0
        self.ta = 0.0
        self._read_b()
        self._blend(0.0)

    def _read_b(self):
        """Read the next record into vb."""
        # Parse the ASCII record once on rank 0 and broadcast it
        record = None
        if self.comm.rank == 0:
            record = self.buffer.read_step(self.node_mask)
        self.tb, _, dat = self.comm.bcast(record, root=0)
        u_interp = matplotlib.tri.LinearTriInterpolator(
            self.triangulation, dat[:, 0])
        v_interp = matplotlib.tri.LinearTriInterpolator(
            self.triangulation, dat[:, 1])

        def vel(x):
            uv = np.vstack((u_interp(x[0], x[1]), v_interp(x[0], x[1])))
            uv = np.ma.getdata(uv).copy()
            # NaNs appear near the interpolation boundary due to machine
            # precision; fall back to the nearest interpolation node
            is_nan = np.isnan(uv).any(axis=0)
            if is_nan.any():
                _, j = self.kd.query(x[:2, is_nan].T)
                uv[0, is_nan] = dat[j, 0]
                uv[1, is_nan] = dat[j, 1]
            return self.scale * uv

        self.vb.interpolate(vel)
        self.vb.x.scatter_forward()

    def _blend(self, t):
        w = (t - self.ta) / (self.tb - self.ta)
        # Operates on owned and ghost entries alike, so no scatter is needed
        self.v.x.array[:] = w * self.vb.x.array + (1.0 - w) * self.va.x.array

    def update(self, t):
        """Update v to time t. Advances the buffer if t leaves [ta, tb]."""
        while t > self.tb:
            self.va.x.array[:] = self.vb.x.array
            self.ta = self.tb
            self._read_b()
        self._blend(t)


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD

    # Mesh and interpolation structures
    domain = adx.read_mesh(args.mesh, comm)
    interp_coordinates = np.load(os.path.join(args.interp_dir, 'interp_coordinates.npy'))
    interp_elements = np.load(os.path.join(args.interp_dir, 'interp_elements.npy'))
    interp_node_map = np.load(os.path.join(args.interp_dir, 'interp_node_map.npy'))
    interp_node_mask = interp_node_map != -1

    # CPP projection of mesh and interpolation coordinates
    def cpp(coord):
        return at.geometry.cpp(
            coord=coord,
            lon0=args.cpp_lon,
            lat0=args.cpp_lat,
            R=args.radius_earth,
            use_deg=True
        )
    interp_coordinates = cpp(interp_coordinates)
    domain.geometry.x[:, :] = cpp(domain.geometry.x[:, :])

    triangulation = matplotlib.tri.Triangulation(
        interp_coordinates[:, 0], interp_coordinates[:, 1], interp_elements)
    kd = scipy.spatial.cKDTree(interp_coordinates)

    # Initial condition
    x0, y0 = cpp(np.array([[args.ic_lon, args.ic_lat]]))[0, :2]
    def iota0(x):
        return args.ic_amplitude * np.exp(
            -((x[0] - x0) ** 2 + (x[1] - y0) ** 2) / (2 * args.ic_radius ** 2))

    # Function spaces
    V = dx.fem.functionspace(domain, ('Discontinuous Lagrange', 1))
    W = dx.fem.functionspace(domain, ('Discontinuous Lagrange', 1, (domain.geometry.dim,)))
    iota = dx.fem.Function(V, name='iota')
    iota.interpolate(iota0)

    # fort.63 is unused for advection only (iota = c h is conserved)
    # Only rank 0 touches the file
    velocity_buffer = None
    if comm.rank == 0:
        velocity_buffer = at.io.TimeSeriesBuffer(args.fort64)
        velocity_buffer.open()
    try:
        velocity = VelocityField(W, comm, velocity_buffer, triangulation, kd,
            interp_node_mask, args.velocity_scale)

        # Conservation law and semi-discrete DG system
        equation = ft.equations.get_advection(
            domain=domain,
            U=iota,
            v=velocity.v
        )
        system = ft.formulations.SemiDiscreteSystem.using_DG(
            equation=equation,
            trace_function=ft.fluxes.fluxn_upwind_scalar
        )
        solver = ft.solvers.TSSolver.from_SemiDiscreteSystem(system, 'explicit')

        # PETSc time stepper
        opts = PETSc.Options()
        opts['ts_ssp_type'] = args.ssp_type
        solver.ts.setType(PETSc.TS.Type.SSP)
        solver.ts.setTime(0.0)
        solver.ts.setTimeStep(args.dt)
        solver.ts.setMaxTime(args.t_final)
        solver.ts.setExactFinalTime(PETSc.TS.ExactFinalTime.MATCHSTEP)

        with dx.io.VTXWriter(comm, args.output, [iota], engine='BP4') as writer:
            def monitor(ts, step, tt, uu):
                if step % args.write_every != 0:
                    return
                if comm.rank == 0:
                    print(f'writing step {step} at time {tt}')
                dx.fem.petsc.assign(uu, iota)
                iota.x.scatter_forward()
                writer.write(tt)
            solver.ts.setMonitor(monitor)

            def pre_step(ts):
                velocity.update(ts.getTime())
            solver.ts.setPreStep(pre_step)

            solver.ts.setFromOptions()
            solver.ts.solve(iota.x.petsc_vec)
    finally:
        if velocity_buffer is not None:
            velocity_buffer.close()


if __name__ == '__main__':
    main()
