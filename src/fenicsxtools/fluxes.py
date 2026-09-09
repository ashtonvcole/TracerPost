"""fluxes.py

Functions for computing numerical fluxes at element interfaces.
"""

import dolfinx
import ufl

def trace_upwind(F: ufl.core.expr.Expr, U: dolfinx.fem.Function,
    J: ufl.core.expr.Expr, n: ufl.FacetNormal) -> ufl.core.operator.Operator:
    """Upwind function trace.

    The value of the trace at an interior element interface is chosen to be
    either F(+) or F(-) depending on the characteristic speeds. If, from both
    sides, the signal is traveling from + to -, F(+) is used, and vice, versa.
    If the characteristics disagree, then an average is used.
    """

    # Project J from + and - onto outward normal of + to get speeds
    # Positive means that the signal travels from + to - for both
    J_plus_n_plus = ufl.dot(J('+'), n('+'))
    J_minus_n_plus = ufl.dot(J('-'), n('+'))

    # Determine whether the wave is definitively entering or leaving +
    # Again, positive means that the signal travels from + to - for both
    leaving_plus = ufl.And(
        ufl.gt(J_plus_n_plus, 0.0),
        ufl.gt(J_minus_n_plus, 0.0)
    )
    entering_plus = ufl.And(
        ufl.lt(J_plus_n_plus, 0.0),
        ufl.lt(J_minus_n_plus, 0.0)
    )

    # Return the upwind trace from the appropriate source element
    return ufl.conditional(
        leaving_plus, # If + to -, use +
        F('+'),
        ufl.conditional(
            entering_plus, # If - to +, use -
            F('-'),
            ufl.avg(F) # Otherwise, use average
        )
    )

def trace_downwind(F: ufl.core.expr.Expr, U: dolfinx.fem.Function,
    J: ufl.core.expr.Expr, n: ufl.FacetNormal) -> ufl.core.operator.Operator:
    """Downwind function trace.

    The value of the trace at an interior element interface is chosen to be
    either F(+) or F(-) depending on the characteristic speeds. If, from both
    sides, the signal is traveling from + to -, F(+) is used, and vice, versa.
    If the characteristics disagree, then an average is used.
    """

    # Project J from + and - onto outward normal of + to get speeds
    # Positive means that the signal travels from + to - for both
    J_plus_n_plus = ufl.dot(J('+'), n('+'))
    J_minus_n_plus = ufl.dot(J('-'), n('+'))

    # Determine whether the wave is definitively entering or leaving +
    # Again, positive means that the signal travels from + to - for both
    leaving_plus = ufl.And(
        ufl.gt(J_plus_n_plus, 0.0),
        ufl.gt(J_minus_n_plus, 0.0)
    )
    entering_plus = ufl.And(
        ufl.lt(J_plus_n_plus, 0.0),
        ufl.lt(J_minus_n_plus, 0.0)
    )

    # Return the upwind trace from the appropriate source element
    return ufl.conditional(
        leaving_plus, # If + to -, use -
        F('-'),
        ufl.conditional(
            entering_plus, # If - to +, use +
            F('+'),
            ufl.avg(F) # Otherwise, use average
        )
    )

def fluxn_upwind_scalar(F: ufl.core.expr.Expr, U: dolfinx.fem.Function,
    J: ufl.core.expr.Expr, n: ufl.FacetNormal) -> ufl.core.operator.Operator:
    """Upwind flux trace.

    The value of the flux trace at an interior element interface is chosen to be
    either F(+) . n(+) or F(-) . n(+) depending on the characteristic speeds.
    If, from both sides, the signal is traveling from + to -, F(+) is used, and
    vice, versa. If the characteristics disagree, then an average is used.

    Args:
        F (ufl.core.expr.Expr): The unrestricted, vector-valued flux expression.
        U (dolfinx.fem.Function): The unrestricted, scalar-valued state.
        J (ufl.core.expr.Expr): The unrestricted, vector-valued flux Jacobian.
        n (ufl.FacetNormal): The unrestricted normal for element faces.

    Returns:
        ufl.core.expr.Expr: The upwind flux trace.
    """

    # Project J from + and - onto outward normal of + to get speeds
    # Positive means that the signal travels from + to - for both
    J_plus_n_plus = ufl.dot(J('+'), n('+'))
    J_minus_n_plus = ufl.dot(J('-'), n('+'))

    # Determine whether the wave is definitively entering or leaving +
    # Again, positive means that the signal travels from + to - for both
    leaving_plus = ufl.And(
        ufl.gt(J_plus_n_plus, 0.0),
        ufl.gt(J_minus_n_plus, 0.0)
    )
    entering_plus = ufl.And(
        ufl.lt(J_plus_n_plus, 0.0),
        ufl.lt(J_minus_n_plus, 0.0)
    )

    # Return the projected upwind flux from the appropriate source element
    return ufl.dot(ufl.conditional(
        leaving_plus, # If + to -, use +
        F('+'),
        ufl.conditional(
            entering_plus, # If - to +, use -
            F('-'),
            ufl.avg(F) # Otherwise, use average
        )
    ), n('+'))

def fluxn_downwind_scalar(F: ufl.core.expr.Expr, U: dolfinx.fem.Function,
    J: ufl.core.expr.Expr, n: ufl.FacetNormal) -> ufl.core.operator.Operator:
    """Downwind flux trace.

    The value of the flux trace at an interior element interface is chosen to be
    either F(+) . n(+) or F(-) . n(+) depending on the characteristic speeds.
    If, from both sides, the signal is traveling from + to -, F(-) is used, and
    vice, versa. If the characteristics disagree, then an average is used.

    Warning:
        In general, this is not a good choice of flux for hyperbolic laws. It
        has utility for the auxiliary equation of Local Discontinuous Galerkin.

    Args:
        F (ufl.core.expr.Expr): The unrestricted, vector-valued flux expression.
        U (dolfinx.fem.Function): The unrestricted, scalar-valued state.
        J (ufl.core.expr.Expr): The unrestricted, vector-valued flux Jacobian.
        n (ufl.FacetNormal): The unrestricted normal for element faces.

    Returns:
        ufl.core.expr.Expr: The downwind flux trace.
    """

    # Project J from + and - onto outward normal of + to get speeds
    # Positive means that the signal travels from + to - for both
    J_plus_n_plus = ufl.dot(J('+'), n('+'))
    J_minus_n_plus = ufl.dot(J('-'), n('+'))

    # Determine whether the wave is definitively entering or leaving +
    # Again, positive means that the signal travels from + to - for both
    leaving_plus = ufl.And(
        ufl.gt(J_plus_n_plus, 0.0),
        ufl.gt(J_minus_n_plus, 0.0)
    )
    entering_plus = ufl.And(
        ufl.lt(J_plus_n_plus, 0.0),
        ufl.lt(J_minus_n_plus, 0.0)
    )

    # Return the projected upwind flux from the appropriate source element
    return ufl.dot(ufl.conditional(
        leaving_plus, # If + to -, use -
        F('-'),
        ufl.conditional(
            entering_plus, # If - to +, use +
            F('+'),
            ufl.avg(F) # Otherwise, use average
        )
    ), n('+'))

def fluxn_llf_scalar(F: ufl.core.expr.Expr, U: dolfinx.fem.Function,
    J: ufl.core.expr.Expr, n: ufl.FacetNormal) -> ufl.core.operator.Operator:
    """Local Lax-Friedrichs/Rusanov flux trace.

    The value of the flux trace at an interior element interface is chosen to be
    a combination of the average of the flux on each side, adjusted by the jump
    in state.

    0.5 * (F(+) + F(-)) . n(+) - 0.5 * lambda * (U(-) - U(+))

    Lambda here is the maximum-magnitude, i.e. worst-case characteristic speed
    associated with the solution at either end of the interface. Combined with
    the jump, and using +'s outward normal, the effect is that for a higher
    U value in -, the flux becomes more negative. Thus the conserved quantity
    flows into +, diffusing the solution.

    Args:
        F (ufl.core.expr.Expr): The unrestricted, vector-valued flux expression.
        U (dolfinx.fem.Function): The unrestricted, scalar-valued state.
        J (ufl.core.expr.Expr): The unrestricted, vector-valued flux Jacobian.
        n (ufl.FacetNormal): The unrestricted normal for element faces.

    Returns:
        ufl.core.expr.Expr: The LLF/Rusanov flux trace.
    """
    lam = ufl.max_value(
        abs(ufl.dot(J('+'), n('+'))), # Characteristic speed magnitude at +
        abs(ufl.dot(J('+'), n('-'))) # Characteristic speed magnitude at -
    )
    return ufl.dot(ufl.avg(F), n('+')) - lam * ufl.jump(U)

def fluxn_llf_downwind_scalar(F: ufl.core.expr.Expr, U: dolfinx.fem.Function,
    J: ufl.core.expr.Expr, n: ufl.FacetNormal) -> ufl.core.operator.Operator:
    """Downwinded Local Lax-Friedrichs/Rusanov flux trace.

    The value of the flux trace at an interior element interface is chosen to be
    a combination of the average of the flux on each side, adjusted by the jump
    in state.

    0.5 * (F(+) + F(-)) . n(+) + 0.5 * lambda * (U(-) - U(+))

    Lambda here is the maximum-magnitude, i.e. worst-case characteristic speed
    associated with the solution at either end of the interface. Crucually, note
    the change of sign, resulting in anti-diffusion.

    Warning:
        In general, this is not a good choice of flux for hyperbolic laws. It
        has utility for the auxiliary equation of Local Discontinuous Galerkin.

    Args:
        F (ufl.core.expr.Expr): The unrestricted, vector-valued flux expression.
        U (dolfinx.fem.Function): The unrestricted, scalar-valued state.
        J (ufl.core.expr.Expr): The unrestricted, vector-valued flux Jacobian.
        n (ufl.FacetNormal): The unrestricted normal for element faces.

    Returns:
        ufl.core.expr.Expr: The LLF/Rusanov flux trace.
    """
    lam = ufl.max_value(
        abs(ufl.dot(J('+'), n('+'))), # Characteristic speed magnitude at +
        abs(ufl.dot(J('+'), n('-'))) # Characteristic speed magnitude at -
    )
    return ufl.dot(ufl.avg(F), n('+')) + lam * ufl.jump(U)
