export TR_ls_step_computation
"""
Compute the direction used in the one dimensionnal trust region scheme.
Return either the Newton direction or the boundary of the trust region if the
Newton direction cannot be used. Inputs:
  - h, value of h(t).
  - g, value of h'(t).
  - dN, value of the Newton direction.
  - Δn, lower bound of the trust region.
  - Δp, upper bound of the trust region.
"""
function TR_ls_step_computation(h :: Float64, g :: Float64, dN :: Float64,
                                Δn :: Float64, Δp ::Float64)

  if h > 0.0
    if g > 0.0
      d = max(Δn, dN)
    else
      d = min(dN, Δp)
    end
  else
    if g > 0.0
      d = Δn
    else
      d = Δp
    end
  end

  return d
end
