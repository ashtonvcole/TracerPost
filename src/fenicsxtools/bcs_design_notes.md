# Boundary condition framework: design notes

Notes from a design discussion for `bcs.py`. Nothing here has been run, and none
of the classes below were written to `bcs.py` when these notes were saved.
References were cited from memory and are unverified.

## 1. Four independent axes

| Axis | Examples | Owner |
|---|---|---|
| What is prescribed | state `U`, total normal flux, nothing (open) | BC (user-facing) |
| Which operator consumes it | hyperbolic trace, parabolic lift and flux | formulation |
| Which data are used | all vs. only inflow characteristics | eigenstructure at runtime, not a user flag |
| How it is enforced | strong dofs (CG), ghost state or numerical flux (DG/LDG) | formulation |

- "Hard-coded vs. characteristic-dependent" is not a BC property. The user
  supplies a full exterior state, and the hyperbolic machinery discards
  components on outgoing characteristics. Scalar case:
  `conditional(dot(J, n) < 0, U_D, U_in)`.
- Strong (CG) Dirichlet enforcement cannot be characteristic-dependent unless the
  flow field is steady, because the inflow facet set would change with `v`.

## 2. Mathematical content

**Hyperbolic.** With `J.n = R Lambda L` and `w = L U`: `w_b[k] = w_in[k]` where
`lambda_k > 0`, `w_b[k] = w_D[k]` where `lambda_k < 0`, and `U_b = R w_b`. The
boundary flux is a Riemann flux between `U_in` and `U_b`. For 2D SWE the speeds
are `v.n - c`, `v.n`, `v.n + c`: subcritical inflow needs 2 data, subcritical
outflow needs 1.

The current default `conditional(J.n > 0, F.n, 0)` means "inflow flux is zero",
not "inflow state is zero". They coincide only for flux linear in `U`.

**Parabolic (LDG).** BC enters in the lift (`u_hat`) and in `sigma_hat . n`
(Cockburn and Shu 1998):

| BC | `u_hat` (lift boundary term) | `sigma_hat . n` |
|---|---|---|
| Dirichlet `c_D` | `c_D` | `sigma(U_b, r_in) . n` (optional penalty `eta (c_in - c_D)`) |
| Neumann `g` | `c_in` | `g` |
| Current default | `c_in` | `0` |

**Total flux BC.** Replace the entire boundary integrand in the `phi`-equation
with `-f phi ds(tag)`, and use `u_hat = c_in` in the lift. No characteristic logic
is needed. A wall is `TotalFlux(0)`.

**Prescribed `c` vs. `iota`.** Specify the BC on the state `U = iota`, and derive
`u_hat` automatically:

```python
c_hat = ufl.replace(grad_of, {equation.U: U_b})   # grad_of = iota/h -> iota_b/h
```

`c = iota/h` is ill-posed as `h -> 0`, so `h` is passed explicitly.

## 3. Infrastructure decisions

- **One shared `MeshTags`.** `ufl.Measure("ds", subdomain_data=...)` takes a single
  tag set. A collection object should hold the `MeshTags`, build `ds` once, and
  validate that every exterior facet is tagged.
- **Explicit default tag for untagged exterior facets** (the `Open` condition).
  I believe a plain `ds` in UFL also applies to tagged facets, so mixing `ds`
  with `ds(tag)` could double-count (about 80% confident; test it).
- **Time dependence.** `value` is a `Constant` or `Function`. `update(t)` writes
  into it, so forms are symbolic in it and nothing is recompiled. The solver must
  call `update` at every RK/IMEX stage time `t_n + c_i dt`, not only at `t_n`.
- **CG strong Dirichlet.** Use `dolfinx.fem.dirichletbc` on dofs from
  `locate_dofs_topological`. It modifies matrix rows, which interacts with
  `is_bilinear_form_constant`. Keep it as a separate `strong=True` path.
- **`trace_function` signature.** It is currently `(F, U, J, n)` and appears to
  rely on `'+'/'-'` restrictions on `dS` (not checked in detail). A boundary
  variant needs explicit interior and exterior arguments. LLF form:
  `0.5*(dot(F_in, n) + dot(F_b, n)) + 0.5*lam*(U_in - U_b)`.
- **Systems.** `exterior_state` needs `R, Lambda, L`. The current
  `Flux.jacobian_expression` appears scalar-oriented (not verified).

## 4. Review of the SWEMNICS example

The SWEMNICS `BoundaryCondition` is a tagged record. `"Open"` builds a strong
`dirichletbc` (CG only). `"Wall"`, `"OF"` and `"Flux"` are weak and store only a
marker. The weak terms are assumed to be assembled elsewhere.

| SWEMNICS | Meaning (my reading) | This design |
|---|---|---|
| `Open` | prescribed state, strong, CG | `State` with `strong=True` |
| `Wall` | zero normal flux, weak | `TotalFlux(0)` |
| `Flux` | prescribed discharge, weak | `TotalFlux(f)` |
| `OF` | free outflow, weak | `Open` |

Shortcomings:

1. Strong enforcement is CG-only; DG has no facet-interior dofs.
2. The constructor needs `V`, coupling the specification to enforcement.
3. No characteristic awareness (SWE reflections at tidal boundaries).
4. String-type dispatch means every new BC edits the class and the solver.
5. Bugs and risks:
   - `fdim = 1` is hardcoded and should be `domain.topology.dim - 1`.
   - Weak BCs return `[]` and `np.array([])` sentinels.
   - Duplicate facets in `MarkBoundary` are ambiguous in `meshtags`.
   - Untagged exterior facets get no tag.
   - `np.hstack` fails on an empty `boundaries` list.
   - The returned `facet_markers` is in locator order, not facet order.

Adopted: a single shared `MeshTags`, with an overlap check and an explicit
default tag. Possible addition: a `components` argument on `State` for
elevation-only prescription.

```python
def mark_boundary(domain, boundaries, default=0):
    fdim = domain.topology.dim - 1
    domain.topology.create_connectivity(fdim, domain.topology.dim)
    ext = mesh.exterior_facet_indices(domain.topology)      # sorted, unique
    tags = np.full(ext.shape, default, dtype=np.int32)
    claimed = np.zeros(ext.shape, dtype=bool)
    for marker, locator in boundaries:
        f = np.intersect1d(mesh.locate_entities(domain, fdim, locator), ext)
        idx = np.searchsorted(ext, f)
        if claimed[idx].any():
            raise ValueError(f"marker {marker} overlaps an earlier boundary")
        tags[idx], claimed[idx] = marker, True
    return mesh.meshtags(domain, fdim, ext, tags)
```

`exterior_facet_indices` is recalled from the dolfinx API, not run. Check its
parallel behavior against the docs.

## 5. Class sketch

Every hook returns `None` by default, meaning "use the formulation default"
(outflow-only hyperbolic flux, zero diffusive flux).

```python
class BoundaryCondition:
    def __init__(self, tag, update=None):
        self._tag, self._update = tag, update

    tag, update, is_time_varying   # properties

    def prescribed_state(self): return None        # raw U_D, parabolic use
    def exterior_state(self, U_in, J, n): return None   # hyperbolic ghost state
    def boundary_normal_flux(self, F_in, U_in, n): return None  # overrides all
    def lift_trace(self, g_in, g_prescribed): return None       # u_hat
    def diffusive_normal_flux(self, F_diff, n): return None     # sigma_hat . n


class State(BoundaryCondition):
    def __init__(self, tag, value, update=None): ...
    def prescribed_state(self): return self._value
    def exterior_state(self, U_in, J, n):
        # scalar filter only; systems need an eigen-decomposition of J.n
        return ufl.conditional(ufl.dot(J, n) < 0.0, self._value, U_in)
    def lift_trace(self, g_in, g_prescribed): return g_prescribed
    def diffusive_normal_flux(self, F_diff, n): return ufl.dot(F_diff, n)


class TotalFlux(BoundaryCondition):
    """Outward-positive F . n, advective + diffusive. Tracer inflow is negative."""
    def __init__(self, tag, value, update=None): ...
    def boundary_normal_flux(self, F_in, U_in, n): return self._value
    def lift_trace(self, g_in, g_prescribed): return g_in
    # all other hooks: inherited None


def state_from_concentration(c, h): return h * c
```

| Hook | `State` | `TotalFlux` |
|---|---|---|
| `prescribed_state` | `U_D` | `None` |
| `exterior_state` | filtered ghost state | `None` |
| `boundary_normal_flux` | `None` | `f` |
| `lift_trace` | `g_prescribed` | `g_in` |
| `diffusive_normal_flux` | `F_diff(U_D, r_in) . n` | `None` |

**Formulation contract.** `boundary_normal_flux` takes precedence. If it is not
`None`, skip `exterior_state` and `diffusive_normal_flux`, or the flux is
double-counted:

```python
if (f := bc.boundary_normal_flux(F, U, n)) is not None:
    residual += -f * phi * ds(tag)
else:
    ...  # exterior_state -> boundary trace; diffusive_normal_flux -> sigma_hat.n

g_hat = bc.lift_trace(g_in, g_prescribed) or g_in
non_stiff_residual += g_hat * ufl.dot(psi, n) * ds(tag)
```

The formulation computes `g_prescribed` with `ufl.replace` on `grad_of` and
evaluates `F_diff` at `(U_D, r_in)`.

## 6. Open decisions

- **Boundary layer at outflow with `State`.** `State` imposes `u_hat = c_D` in the
  lift even at outflow, producing a diffusive boundary layer. Decide whether the
  diffusive part should be selectable, e.g.
  `State(hyperbolic=..., parabolic=Open)`.
- **`DiffusiveFlux` (Neumann).** Add a separate class, with
  `diffusive_normal_flux` returning `g`, and the hyperbolic part keeping its
  ghost treatment.
- **Systems.** Characteristic filter for SWE.
- **Wiring.** Nothing in `formulations.py` calls the hooks yet.

## 7. Differential tests

1. `Open` reproduces current outputs exactly (regression against `examples/06_bc`).
2. `State(U_b = U_in)` equals `Open` on outflow-only boundaries.
3. Manufactured advection-diffusion solution with `State` at inflow and `Open` at
   outflow; check convergence under `p`-refinement.
4. Global conservation: `d/dt integral U dx = -closed-integral F_hat . n ds` to
   roundoff, for each BC type.
5. `TotalFlux(f)` against `State` with matching Dirichlet data, where the exact
   boundary flux is known.
6. UFL test for `ds` vs. `ds(tag)` double-counting.
7. Two overlapping locators in `mark_boundary` raise `ValueError`; compare the
   SWEMNICS `MarkBoundary` on the same input.

## References (unverified, cited from memory)

- Cockburn and Shu, "The Local Discontinuous Galerkin Method for Time-Dependent
  Convection-Diffusion Systems," *SIAM J. Numer. Anal.* 35(6), 1998.
- Arnold, Brezzi, Cockburn and Marini, "Unified Analysis of Discontinuous Galerkin
  Methods for Elliptic Problems," *SIAM J. Numer. Anal.* 39(5), 2002.
- Hesthaven and Warburton, *Nodal Discontinuous Galerkin Methods*, Springer, 2008.
- LeVeque, *Finite Volume Methods for Hyperbolic Problems*, Cambridge, 2002.
