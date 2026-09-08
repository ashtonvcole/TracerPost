from src.fenicsxtools.equations import HyperbolicConservationLaw, ParabolicConservationLaw

class CGFormulation:
    pass

class SUPGFormulation:
    pass

class DGFormulation:
    def __init__(self, equation: HyperbolicConservationLaw, F_trace: ufl.Restricted):
        # Mass form for u
        # Flux form for g
        pass

class LDGFormulation:
    def __init__(self, equation: ParabolicConservationLaw, Fu_trace: ufl.Restricted, Fg_trace: ufl.Restricted):
        # Mass form for u
        # Mass form for g
        # Flux form for u
        # Flux form for g
        pass

class SIPGFormulation:
    pass
