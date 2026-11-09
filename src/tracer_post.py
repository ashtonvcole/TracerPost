"""tracer_post.py

Run script for passive tracer post-processing of ADCIRC/DGSWEM output. Solves
the depth-averaged advection equation for iota = c h with a DG discretization,
using nodal velocity data from a fort.64 file.

All run options are read from a TOML configuration file (see
tracer_config.toml for a documented template):

    python tracer_post.py tracer_config.toml

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
from types import SimpleNamespace
import ufl

try:
    import tomllib  # Python >= 3.11
except ModuleNotFoundError:
    import tomli as tomllib


# Configuration schema: {section: {key: (default, type)}}. A default of
# REQUIRED means the key must be present. TOML has no null, so optional keys
# with a default of None are simply omitted from the file.
REQUIRED = object()
SCHEMA = {
    'inputs': {
        'mesh': (REQUIRED, str),
        'bathymetry_name': (None, str),
        'interp_dir': (REQUIRED, str),
        'fort64': (REQUIRED, str),
        'fort63': (REQUIRED, str),
    },
    'projection': {
        'radius_earth': (6378206.4, float),
        'cpp_lon': (-94.5, float),
        'cpp_lat': (29.0, float),
    },
    'initial_condition': {
        'variable': ('iota', str),
        'type': ('gaussian', str),
        'value': (1.0, float),
        'bp': (None, str),
        'name': ('iota', str),
        'lon': (-97.0611, float),
        'lat': (27.8339, float),
        'radius': (2000.0, float),
        'amplitude': (1.0, float),
    },
    'physics': {
        'velocity_scale': (1.0, float),
        'diffusivity': (0.0, float),
    },
    'time_stepping': {
        't_final': (REQUIRED, float),
        'dt': (1.0, float),
        'ssp_type': ('rks3', str),
    },
    'output': {
        'fields': (['iota'], list),
        'h_min': (1e-3, float),
        'path': ('tracer.bp', str),
        'write_every': (3600, int),
    },
}
CHOICES = {
    ('initial_condition', 'variable'): ('iota', 'c'),
    ('initial_condition', 'type'): ('gaussian', 'constant', 'bp'),
}
# Relative paths in these keys are resolved against the config file directory
PATH_KEYS = {
    ('inputs', 'mesh'), ('inputs', 'interp_dir'), ('inputs', 'fort64'),
    ('inputs', 'fort63'), ('initial_condition', 'bp'), ('output', 'path'),
}


def load_config(path: str) -> SimpleNamespace:
    """Read and validate a TOML run configuration.

    Args:
        path: Path to the TOML file.

    Returns:
        SimpleNamespace: Nested namespace, accessed as cfg.section.key, with
            defaults filled in and relative paths resolved.

    Raises:
        ValueError: On unknown sections/keys, missing required keys, wrong
            types, or invalid values.
    """
    with open(path, 'rb') as f:
        raw = tomllib.load(f)
    # Config keys are written with hyphens (e.g. interp-dir); the schema and
    # attribute names use underscores
    raw = {s.replace('-', '_'): ({k.replace('-', '_'): v for k, v in d.items()}
                                 if isinstance(d, dict) else d)
           for s, d in raw.items()}
    base = os.path.dirname(os.path.abspath(path))

    unknown = set(raw) - set(SCHEMA)
    if unknown:
        raise ValueError(f'unknown config section(s): {sorted(unknown)}')

    cfg = {}
    for section, keys in SCHEMA.items():
        given = raw.get(section, {})
        unknown = set(given) - set(keys)
        if unknown:
            raise ValueError(
                f'unknown key(s) in [{section}]: {sorted(unknown)}')
        out = {}
        for key, (default, typ) in keys.items():
            if key not in given:
                if default is REQUIRED:
                    raise ValueError(f'missing required key [{section}] {key}')
                out[key] = default
                continue
            val = given[key]
            # Accept ints where floats are expected, but not bools
            if typ is float and isinstance(val, int) \
                    and not isinstance(val, bool):
                val = float(val)
            if not isinstance(val, typ) or (
                    typ is not list and isinstance(val, bool)):
                raise ValueError(
                    f'[{section}] {key} must be {typ.__name__}, '
                    f'got {type(val).__name__}')
            if (section, key) in PATH_KEYS:
                # Expand ~; absolute paths pass through join unchanged
                val = os.path.normpath(
                    os.path.join(base, os.path.expanduser(val)))
            out[key] = val
        cfg[section] = out

    for (section, key), choices in CHOICES.items():
        if cfg[section][key] not in choices:
            raise ValueError(
                f'[{section}] {key} must be one of {choices}, '
                f'got {cfg[section][key]!r}')
    fields = cfg['output']['fields']
    if not fields or any(f not in ('iota', 'c') for f in fields):
        raise ValueError("[output] fields must be a non-empty list of "
                         "'iota' and/or 'c'")
    if cfg['physics']['diffusivity'] != 0.0:
        raise ValueError('diffusion is not implemented; '
                         '[physics] diffusivity must be 0')

    return SimpleNamespace(**{s: SimpleNamespace(**v) for s, v in cfg.items()})


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
    parser = argparse.ArgumentParser(
        description='Advect a passive tracer with ADCIRC/DGSWEM velocities.')
    parser.add_argument('config', help='TOML run configuration file.')
    cfg = load_config(parser.parse_args().config)

    comm = MPI.COMM_WORLD
    log = setup_logging(comm=comm, level=logging.INFO, all_ranks=False)
    log.info('Starting Tracer Post salinity solver')

    inp, proj, ic = cfg.inputs, cfg.projection, cfg.initial_condition
    phys, ts, out = cfg.physics, cfg.time_stepping, cfg.output

    # Mesh and interpolation structures
    log.info('Reading mesh...')
    domain = adx.read_mesh(inp.mesh, comm)
    interp_coordinates = np.load(os.path.join(inp.interp_dir, 'interp_coordinates.npy'))
    interp_elements = np.load(os.path.join(inp.interp_dir, 'interp_elements.npy'))
    interp_node_map = np.load(os.path.join(inp.interp_dir, 'interp_node_map.npy'))
    interp_node_mask = interp_node_map != -1

    # CPP projection of mesh and interpolation coordinates
    def cpp(coord):
        return at.geometry.cpp(
            coord=coord,
            lon0=proj.cpp_lon,
            lat0=proj.cpp_lat,
            R=proj.radius_earth,
            use_deg=True
        )
    interp_coordinates = cpp(interp_coordinates)
    domain.geometry.x[:, :] = cpp(domain.geometry.x[:, :])

    triangulation = matplotlib.tri.Triangulation(
        interp_coordinates[:, 0], interp_coordinates[:, 1], interp_elements)
    kd = scipy.spatial.cKDTree(interp_coordinates)

    # Initial condition
    x0, y0 = cpp(np.array([[ic.lon, ic.lat]]))[0, :2]
    def iota0(x):
        return ic.amplitude * np.exp(
            -((x[0] - x0) ** 2 + (x[1] - y0) ** 2) / (2 * ic.radius ** 2))

    # Function spaces
    log.info('Initializing tracer fields...')
    V = dx.fem.functionspace(domain, ('Discontinuous Lagrange', 1))
    W = dx.fem.functionspace(domain, ('Discontinuous Lagrange', 1, (domain.geometry.dim,)))
    iota = dx.fem.Function(V, name='iota')
    if ic.type == 'gaussian':
        iota.interpolate(iota0)
    elif ic.type == 'constant':
        iota.x.array[:] = ic.value
    else:
        # First, check that function space matches
        ic_file = ic.bp or inp.mesh
        att = adx.read_attributes(ic_file, comm, f'{ic.name}_element')
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
        adx.read_function(ic_file, iota, name=ic.name)
    iota.x.scatter_forward()

    # Total depth h = zeta + DP is needed for the inflow boundary state
    # (iota = c h) and for the output field c = iota / h
    want_c = 'c' in out.fields
    dp_cg = dx.fem.Function(
        dx.fem.functionspace(domain, ('Lagrange', 1)))
    adx.read_function(inp.mesh, dp_cg, name=inp.bathymetry_name)
    dp = dx.fem.Function(V)
    dp.interpolate(dp_cg)
    if want_c:
        c = dx.fem.Function(V, name='c')

    # fort.63 supplies zeta for the inflow boundary state and for c = iota / h
    # Only rank 0 touches the files
    velocity_buffer = None
    elevation_buffer = None
    if comm.rank == 0:
        log.info('Loading ADCIRC data buffers...')
        velocity_buffer = at.io.TimeSeriesBuffer(inp.fort64)
        velocity_buffer.open()
        elevation_buffer = at.io.TimeSeriesBuffer(inp.fort63)
        elevation_buffer.open()
    try:
        velocity = NodalTimeSeriesField(W, comm, velocity_buffer,
            triangulation, kd, interp_node_mask, phys.velocity_scale)
        elevation = NodalTimeSeriesField(V, comm, elevation_buffer,
            triangulation, kd, interp_node_mask, 1.0)

        # If the initial condition was specified as c, convert: iota = c h
        # with h = max(zeta + DP, 0) evaluated at t = 0
        if ic.variable == 'c':
            h0 = np.maximum(elevation.v.x.array + dp.x.array, 0.0)
            iota.x.array[:] = iota.x.array * h0
            iota.x.scatter_forward()

        # Conservation law and semi-discrete DG system
        log.info('Building system...')
        equation = ft.equations.get_advection(
            domain=domain,
            U=iota,
            v=velocity.v
        )
        system = ft.formulations.SemiDiscreteSystem.using_DG(
            equation=equation,
            trace_function=ft.fluxes.fluxn_upwind_scalar,
            boundary_state=ufl.max_value(elevation.v + dp, 0.0)  # iota_ext = c h, c = 1
        )
        solver = ft.solvers.TSSolver.from_SemiDiscreteSystem(system, 'explicit')

        # PETSc time stepper
        opts = PETSc.Options()
        opts['ts_ssp_type'] = ts.ssp_type
        solver.ts.setType(PETSc.TS.Type.SSP)
        solver.ts.setTime(0.0)
        solver.ts.setTimeStep(ts.dt)
        solver.ts.setMaxTime(ts.t_final)
        solver.ts.setExactFinalTime(PETSc.TS.ExactFinalTime.MATCHSTEP)

        out_fields = [f for f in (iota, c if want_c else None)
                      if f is not None and f.name in out.fields]
        with dx.io.VTXWriter(comm, out.path, out_fields,
                             engine='BP4') as writer:
            def monitor(ts_, step, tt, uu):
                if step % out.write_every != 0:
                    return
                log.info(f'writing step {step} at time {tt}')
                dx.fem.petsc.assign(uu, iota)
                iota.x.scatter_forward()
                if want_c:
                    # Total depth; ADCIRC dry-node flags are clipped by h_min
                    h = elevation.v.x.array + dp.x.array
                    wet = h >= out.h_min
                    c.x.array[:] = 0.0
                    c.x.array[wet] = iota.x.array[wet] / h[wet]
                    c.x.scatter_forward()
                writer.write(tt)
            solver.ts.setMonitor(monitor)

            def pre_step(ts_):
                tt = ts_.getTime()
                elevation.update(tt)
                velocity.update(tt)
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
