"""tracer_post.py

Run script for passive tracer post-processing of ADCIRC/DGSWEM output. Solves
the depth-averaged advection equation for iota = c h with a DG discretization,
using nodal velocity data from a fort.64 file.

Diffusion is not yet implemented.
"""


import adcircxtools as at
import adios4dolfinx as adx
import argparse
import dolfinx as dx
import dolfinx.cpp.mesh
import dolfinx.fem.petsc
import dolfinx.io
import fenicsxtools as ft
import logging
import matplotlib.tri
from mpi4py import MPI
import numpy as np
import os
from petsc4py import PETSc
import scipy.spatial
import sys


def parse_args() -> argparse.Namespace:
    """Read tracer simulation options using Argparse.

    Returns
        argparse.Namespace: The tracer simulation options.
    """
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
        help='ADCIRC elevation time series (fort.63). Required to output '
             'c = iota / h (h = zeta + DP).')

    g = p.add_argument_group('projection')
    g.add_argument('--radius-earth', type=float, default=6378206.4,
        help='Earth radius for CPP [m] (default: %(default)s).')
    g.add_argument('--cpp-lon', type=float, default=-94.5,
        help='CPP reference longitude [deg] (default: %(default)s).')
    g.add_argument('--cpp-lat', type=float, default=29.0,
        help='CPP reference latitude [deg] (default: %(default)s).')

    g = p.add_argument_group('initial condition')
    g.add_argument('--ic-type', choices=['gaussian', 'constant', 'bp'],
        default='gaussian',
        help='Initial condition for iota (default: %(default)s).')
    g.add_argument('--ic-value', type=float, default=1.0,
        help='Constant value of iota for --ic-type constant '
             '(default: %(default)s).')
    g.add_argument('--ic-bp', default=None,
        help='adios4dolfinx file holding iota for --ic-type bp. Must be a '
             'DG1 function on the mesh (default: the --mesh file).')
    g.add_argument('--ic-name', default='iota',
        help='Name of the function in --ic-bp (default: %(default)s).')
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
    g.add_argument('--output-fields', nargs='+', choices=['iota', 'c'],
        default=['iota'],
        help='Fields to write: iota and/or c = iota / h (default: iota). '
             'c requires --fort63 and bathymetry in --mesh.')
    g.add_argument('--bathymetry-name', default=None,
        help='Name of the continuous P1 bathymetry function (DP) in --mesh '
             '(default: the dolfinx default name, as written by '
             'adx.write_function without renaming).')
    g.add_argument('--h-min', type=float, default=1e-3,
        help='c is set to 0 where h < h-min [m] (default: %(default)s).')
    g.add_argument('--output', default='tracer.bp',
        help='Output VTX/ADIOS2 path (default: %(default)s).')
    g.add_argument('--write-every', type=int, default=3600,
        help='Write a snapshot every N steps (default: %(default)s).')

    args = p.parse_args()
    if args.diffusivity != 0.0:
        p.error('diffusion is not implemented; --diffusivity must be 0')
    if 'c' in args.output_fields and args.fort63 is None:
        p.error('--output-fields c requires --fort63')
    return args


def setup_logging(comm, level=logging.INFO, all_ranks=False):
    log = logging.getLogger('tracer_post')
    log.setLevel(level)
    if all_ranks or comm.rank == 0:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(
            f'%(asctime)s [rank {comm.rank}/{comm.size}] %(levelname)s %(message)s'))
        log.addHandler(h)
    else:
        log.addHandler(logging.NullHandler())
    return log


class NodalTimeSeriesField:
    """Linear-in-time, linear-in-space interpolation of fort.63/64 data onto a
    DG function (scalar or vector). Holds a pair of snapshots (a, b) bracketing
    the current time.
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
        interps = [matplotlib.tri.LinearTriInterpolator(
            self.triangulation, dat[:, k]) for k in range(dat.shape[1])]

        def vel(x):
            uv = np.vstack([f(x[0], x[1]) for f in interps])
            uv = np.ma.getdata(uv).copy()
            # NaNs appear near the interpolation boundary due to machine
            # precision; fall back to the nearest interpolation node
            is_nan = np.isnan(uv).any(axis=0)
            if is_nan.any():
                _, j = self.kd.query(x[:2, is_nan].T)
                uv[:, is_nan] = dat[j, :].T
            uv = self.scale * uv
            return uv if uv.shape[0] > 1 else uv[0]

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
    log = setup_logging(comm=comm, level=logging.INFO, all_ranks=False)
    log.info('Starting Tracer Post salinity solver')

    # Mesh and interpolation structures
    log.info('Reading mesh...')
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
    log.info('Initializing tracer fields...')
    V = dx.fem.functionspace(domain, ('Discontinuous Lagrange', 1))
    W = dx.fem.functionspace(domain, ('Discontinuous Lagrange', 1, (domain.geometry.dim,)))
    iota = dx.fem.Function(V, name='iota')
    if args.ic_type == 'gaussian':
        iota.interpolate(iota0)
    elif args.ic_type == 'constant':
        iota.x.array[:] = args.ic_value
    else:
        # First, check that function space matches
        att = adx.read_attributes(args.ic_bp or args.mesh, comm, 'iota_element')
        el = V.ufl_element()
        expected_family = el.basix_element.family.value
        expected_degree = el.degree
        expected_discontinuous = int(el.discontinuous)
        found_family = int(att['family'][0])
        found_degree = int(att['degree'][0])
        found_discontinuous = int(att['discontinuous'][0])
        if (
            expected_family != found_family
            or expected_degree != found_degree
            or expected_discontinuous != found_discontinuous
        ):
            raise ValueError(
                f'checkpoint element != function space\n'
                f'expected family {expected_family}, found {found_family}\n'
                f'expected degree {expected_degree}, found {found_degree}\n'
                f'expected discontinuous {expected_discontinuous}, found {found_discontinuous}\n'
            )
        adx.read_function(args.ic_bp or args.mesh, iota, name=args.ic_name)
    iota.x.scatter_forward()

    # Output fields: c = iota / h requires h = zeta + DP
    want_c = 'c' in args.output_fields
    if want_c:
        dp_cg = dx.fem.Function(
            dx.fem.functionspace(domain, ('Lagrange', 1)))
        adx.read_function(args.mesh, dp_cg, name=args.bathymetry_name)
        dp = dx.fem.Function(V)
        dp.interpolate(dp_cg)
        c = dx.fem.Function(V, name='c')

    # fort.63 is only needed to convert iota to c (iota = c h is conserved)
    # Only rank 0 touches the files
    velocity_buffer = None
    elevation_buffer = None
    if comm.rank == 0:
        log.info('Loading ADCIRC data buffers...')
        velocity_buffer = at.io.TimeSeriesBuffer(args.fort64)
        velocity_buffer.open()
        if want_c:
            elevation_buffer = at.io.TimeSeriesBuffer(args.fort63)
            elevation_buffer.open()
    try:
        velocity = NodalTimeSeriesField(W, comm, velocity_buffer,
            triangulation, kd, interp_node_mask, args.velocity_scale)
        if want_c:
            elevation = NodalTimeSeriesField(V, comm, elevation_buffer,
                triangulation, kd, interp_node_mask, 1.0)

        # Conservation law and semi-discrete DG system
        log.info('Building system...')
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

        out_fields = [f for f in (iota, c if want_c else None)
                      if f is not None and f.name in args.output_fields]
        with dx.io.VTXWriter(comm, args.output, out_fields,
                             engine='BP4') as writer:
            def monitor(ts, step, tt, uu):
                if step % args.write_every != 0:
                    return
                log.info(f'writing step {step} at time {tt}')
                dx.fem.petsc.assign(uu, iota)
                iota.x.scatter_forward()
                if want_c:
                    elevation.update(tt)
                    # Total depth; ADCIRC dry-node flags are clipped by h_min
                    h = elevation.v.x.array + dp.x.array
                    wet = h >= args.h_min
                    c.x.array[:] = 0.0
                    c.x.array[wet] = iota.x.array[wet] / h[wet]
                    c.x.scatter_forward()
                writer.write(tt)
            solver.ts.setMonitor(monitor)

            def pre_step(ts):
                velocity.update(ts.getTime())
            solver.ts.setPreStep(pre_step)

            solver.ts.setFromOptions()
            log.info('Begin solve...')
            solver.ts.solve(iota.x.petsc_vec)
    finally:
        for b in (velocity_buffer, elevation_buffer):
            if b is not None:
                b.close()


if __name__ == '__main__':
    main()
