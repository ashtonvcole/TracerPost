class ExplicitSemiDiscrete:
    @property
    def operator(self):
        return self._operator

    @property
    def residual(self):
        return self._residual

    @property
    def state(self) -> petsc4py.PETSc.Vec:
        return self._state
