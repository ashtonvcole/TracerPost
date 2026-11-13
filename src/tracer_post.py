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
import copy
import dolfinx as dx
import dolfinx.cpp.mesh
import dolfinx.fem.petsc
import dolfinx.io
import fenicsxtools as ft
import json
import jsonschema
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


# Run options are validated against a JSON Schema (draft 2020-12) stored next
# to this script. The validator does not apply defaults, so _apply_defaults
# fills them from the schema first. Optional keys without a default (e.g.
# bathymetry-name) become None in the returned namespace.
SCHEMA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'tracer_config.schema.json')
# Relative paths in these keys are resolved against the config file directory
PATH_KEYS = {
    ('inputs', 'mesh'), ('inputs', 'interp-dir'), ('inputs', 'fort64'),
    ('inputs', 'fort63'), ('initial-condition', 'bp'), ('output', 'path'),
}


def _hyphenate(obj):
    """Return obj with underscores in all dictionary keys changed to hyphens."""
    if isinstance(obj, dict):
        return {k.replace('_', '-'): _hyphenate(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_hyphenate(v) for v in obj]
    return obj


def _apply_defaults(instance: dict, schema: dict):
    """Fill missing keys with schema defaults, in place. Objects missing
    entirely are created empty so that their own defaults apply.
    """
    for key, sub in schema.get('properties', {}).items():
        if key not in instance:
            if 'default' in sub:
                instance[key] = copy.deepcopy(sub['default'])
            elif sub.get('type') == 'object':
                instance[key] = {}
        if isinstance(instance.get(key), dict):
            _apply_defaults(instance[key], sub)


def _to_namespace(instance: dict, schema: dict) -> SimpleNamespace:
    """Convert a validated instance to nested namespaces with underscore
    attribute names. Missing optional keys are None; ints are widened to
    floats where the schema says number.
    """
    values = {}
    for key, sub in schema['properties'].items():
        val = instance.get(key)
        if val is not None:
            kind = sub.get('type')
            if kind == 'object':
                val = _to_namespace(val, sub)
            elif kind == 'array' and sub['items'].get('type') == 'object':
                val = [_to_namespace(v, sub['items']) for v in val]
            elif kind == 'number':
                val = float(val)
            elif kind == 'integer':
                val = int(val)
        values[key.replace('-', '_')] = val
    return SimpleNamespace(**values)


def _find_unimplemented(instance, schema, path=()):
    for key, sub in schema.get('properties', {}).items():
        if key not in instance:
            continue
        here = path + (key,)
        if 'x-unimplemented' in sub:
            yield here, sub['x-unimplemented']
        val = instance[key]
        if isinstance(val, dict):
            yield from _find_unimplemented(val, sub, here)
        elif isinstance(val, list) and isinstance(sub.get('items'), dict):
            for i, item in enumerate(val):
                if isinstance(item, dict):
                    yield from _find_unimplemented(item, sub['items'], here + (i,))


def load_config(path: str) -> SimpleNamespace:
    """Read and validate a TOML run configuration.

    Args:
        path: Path to the TOML file.

    Returns:
        SimpleNamespace: Nested namespace, accessed as cfg.section.key, with
            defaults filled in and relative paths resolved. Boundary
            conditions are a list of namespaces, cfg.boundary_conditions.

    Raises:
        ValueError: If the configuration does not satisfy the schema. All
            violations are reported together.
    """
    with open(SCHEMA_PATH) as f:
        schema = json.load(f)
    with open(path, 'rb') as f:
        raw = tomllib.load(f)
    # Config keys are written with hyphens (e.g. interp-dir); underscores are
    # also accepted
    given = _hyphenate(raw)  # As written, before defaults are filled in
    cfg = copy.deepcopy(given)
    _apply_defaults(cfg, schema)

    errors = sorted(
        jsonschema.Draft202012Validator(schema).iter_errors(cfg),
        key=lambda e: [str(p) for p in e.absolute_path])
    if errors:
        raise ValueError('invalid configuration:\n' + '\n'.join(
            f"  {'/'.join(str(p) for p in e.absolute_path) or '<root>'}: "
            f'{e.message}' for e in errors))

    unimplemented = list(_find_unimplemented(given, schema))
    if unimplemented:
        raise NotImplementedError('not implemented yet:\n' + '\n'.join(
            f"  {'/'.join(str(p) for p in path)}: {why}" for path, why in unimplemented))

    base = os.path.dirname(os.path.abspath(path))
    for section, key in PATH_KEYS:
        # Only paths given in the file are resolved; defaults stay as written
        if key in given.get(section, {}):
            # Expand ~; absolute paths pass through join unchanged
            cfg[section][key] = os.path.normpath(
                os.path.join(base, os.path.expanduser(cfg[section][key])))

    return _to_namespace(cfg, schema)


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

    # Shorthand for TOML sections
    inp = cfg.inputs
    proj = cfg.projection
    icond = cfg.initial_condition
    phys = cfg.physics
    tstep = cfg.time_stepping
    out = cfg.output
    bconds = cfg.boundary_conditions

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
    x0, y0 = cpp(np.array([[icond.lon, icond.lat]]))[0, :2]
    def iota0(x):
        return icond.amplitude * np.exp(
            -((x[0] - x0) ** 2 + (x[1] - y0) ** 2) / (2 * icond.radius ** 2))

    # Function spaces
    log.info('Initializing tracer fields...')
    V = dx.fem.functionspace(domain, ('Discontinuous Lagrange', 1))
    W = dx.fem.functionspace(domain, ('Discontinuous Lagrange', 1, (domain.geometry.dim,)))
    iota = dx.fem.Function(V, name='iota')
    if icond.type == 'gaussian':
        iota.interpolate(iota0)
    elif icond.type == 'constant':
        iota.x.array[:] = icond.value
    else:
        # First, check that function space matches
        ic_file = icond.bp or inp.mesh
        att = adx.read_attributes(ic_file, comm, f'{icond.name}_element')
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
        adx.read_function(ic_file, iota, name=icond.name)
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
        if icond.variable == 'c':
            h0 = np.maximum(elevation.v.x.array + dp.x.array, 0.0)
            iota.x.array[:] = iota.x.array * h0
            iota.x.scatter_forward()

        # Process boundary conditions
        log.info('Processing boundary conditions')
        boundary_conditions = []
        ds = None
        if bconds:
            facet_tags = adx.read_meshtags(inp.mesh, domain, inp.boundary_meshtag_name)
            ds = ufl.Measure('ds', domain=domain, subdomain_data=facet_tags)
            bcm = adx.read_attributes(inp.mesh, comm, inp.boundary_meshtag_map)
            for bcond in bconds:
                marker = (bcond.marker 
                    if bcond.marker is not None
                    else bcm[bcond.name][0]
                )
                if bcond.type == 'dirichlet-c':
                    boundary_conditions.append(
                        ft.bcs.DirichletCondition(
                            tag=marker,
                            value=dx.fem.Constant(domain, bcond.value) * ufl.max_value(
                                elevation.v + dp,
                                dx.fem.Constant(domain, 0.0)
                            )
                        )
                    )
                elif bcond.type == 'dirichlet-iota':
                    boundary_conditions.append(
                        ft.bcs.DirichletCondition(
                            tag=marker,
                            value=dx.fem.Constant(domain, bcond.value)
                        )
                    )
                else:
                    raise ValueError(f'Invalid boundary specification {bcond.type} at marker {bcond.marker}')

        # Conservation law and semi-discrete DG system
        log.info('Building system...')
        equation = ft.equations.get_advection(
            domain=domain,
            U=iota,
            v=velocity.v
        )
        system = ft.formulations.SemiDiscreteSystem.using_DG(
            equation=equation,
            trace_function=ft.fluxes.llf,
            boundary_conditions=boundary_conditions,
            ds=ds
        )
        ts = ft.solvers.ts_from_SemiDiscreteSystem(system, 'explicit')

        # PETSc time stepper
        opts = PETSc.Options()
        opts['ts_ssp_type'] = tstep.ssp_type
        ts.setType(PETSc.TS.Type.SSP)
        ts.setTime(0.0)
        ts.setTimeStep(tstep.dt)
        ts.setMaxTime(tstep.t_final)
        ts.setExactFinalTime(PETSc.TS.ExactFinalTime.MATCHSTEP)

        out_fields = [f for f in (iota, c if want_c else None)
                      if f is not None and f.name in out.fields]
        with dx.io.VTXWriter(comm, out.path, out_fields,
                             engine='BP4') as writer:
            def monitor(ts_, step, tt, uu):
                if step % out.write_every != 0 and not ts_.getConvergedReason():
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
            ts.setMonitor(monitor)

            def pre_step(ts_):
                tt = ts_.getTime()
                elevation.update(tt)
                velocity.update(tt)
            ts.setPreStep(pre_step)

            ts.setFromOptions()

            # Solution loop happens here
            log.info('Begin solve...')
            ts.solve(iota.x.petsc_vec)
    finally:
        for b in (velocity_buffer, elevation_buffer):
            if b is not None:
                b.close()


if __name__ == '__main__':
    main()
