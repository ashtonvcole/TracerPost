"""bcs.py

General template for boundary conditions, which are handled differently by
different formulations.
"""

class BoundaryCondition:
    """A wrapper for a boundary condition.
    """

    def __init__(self, tag: int, update: Callable | None = None):
        self._tag = tag
        self._update = update

    @property
    def tag(self) -> int:
        """int: The associated boundary marker."""
        return self._tag

    @property
    def update(self) -> Callable | None:
        """Callable: An update function for a time-varying condition."""
        return self._update

    @property
    def is_time_varying(self) -> bool:
        """bool: Whether the boundary condition needs to be updated."""
        return self._update is not None

    # PUT HOOKS HERE

class DirichletCondition(BoundaryCondition):
    def __init__(self, tag, value, update = None):
        super().__init__(tag, update)
        self._value = value
