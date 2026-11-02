class BoundaryCondition(ABC):
    tag: int
    update: Callable[[float], None] | None = None     # None => time-invariant

    # Hooks; each returns a UFL expression or None ("not my concern")
    def exterior_state(self, U_in, J, n): ...         # hyperbolic ghost state
    def boundary_normal_flux(self, F_in, U_in, n): ...  # overrides ALL numerical flux
    def lift_trace(self, g_in, U_b): ...              # û
    def diffusive_normal_flux(self, F_diff_in, n): ... # σ̂·n

class State(BoundaryCondition):      # Dirichlet; value is a ufl expr / Constant / Function
class TotalFlux(BoundaryCondition):  # value = prescribed F·n (incl. diffusive part)
class Open(BoundaryCondition):       # current default: outflow-only hyperbolic, zero diffusive flux



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



from collections.abc import Callable
import ufl


class BoundaryCondition:
    """A wrapper for a boundary condition.

    Every hook returns None by default, meaning "use the formulation default"
    (outflow-only hyperbolic flux, zero diffusive flux).
    """

    def __init__(self, tag: int, update: Callable[[float], None] | None = None):
        self._tag = tag
        self._update = update   # update(t) writes into Constant/Function data

    @property
    def tag(self) -> int:
        return self._tag

    @property
    def update(self) -> Callable | None:
        return self._update

    @property
    def is_time_varying(self) -> bool:
        return self._update is not None

    # Hooks
    def prescribed_state(self):
        """Raw prescribed U, before characteristic filtering (parabolic use)."""
        return None

    def exterior_state(self, U_in, J, n):
        """Hyperbolic ghost state: prescribed data on incoming characteristics,
        interior values elsewhere."""
        return None

    def boundary_normal_flux(self, F_in, U_in, n):
        """Prescribed total normal flux; overrides the whole numerical flux."""
        return None

    def lift_trace(self, g_in, g_prescribed):
        """û in the auxiliary gradient equation. g_prescribed is grad_of with
        U replaced by prescribed_state(), computed by the formulation."""
        return None

    def diffusive_normal_flux(self, F_diff, n):
        """σ̂·n. F_diff is evaluated by the formulation at the prescribed state
        (if any) and the interior auxiliary variable."""
        return None


class State(BoundaryCondition):
    """Dirichlet condition on the conserved state U."""

    def __init__(self, tag, value, update=None):
        super().__init__(tag, update)
        self._value = value

    @property
    def value(self):
        return self._value

    def prescribed_state(self):
        return self._value

    def exterior_state(self, U_in, J, n):
        # Scalar filter: inflow iff the normal wave speed is negative.
        # Systems need an eigen-decomposition of J.n, which is not available yet.
        return ufl.conditional(ufl.dot(J, n) < 0.0, self._value, U_in)

    def lift_trace(self, g_in, g_prescribed):
        return g_prescribed

    def diffusive_normal_flux(self, F_diff, n):
        return ufl.dot(F_diff, n)   # σ̂ = σ(U_D, r_in), Cockburn–Shu


def state_from_concentration(c, h):
    """iota = h c. h is explicit because c is ill-posed as h -> 0."""
    return h * c



class TotalFlux(BoundaryCondition):
    """Prescribed total normal flux F . n (advective + diffusive), outward
    positive. A tracer inflow is therefore a negative value."""

    def __init__(self, tag, value, update=None):
        super().__init__(tag, update)
        self._value = value

    @property
    def value(self):
        return self._value

    def boundary_normal_flux(self, F_in, U_in, n):
        return self._value            # replaces the entire numerical flux

    def lift_trace(self, g_in, g_prescribed):
        return g_in                   # û = c_in (no state is prescribed)

    # prescribed_state, exterior_state, diffusive_normal_flux: inherited None
