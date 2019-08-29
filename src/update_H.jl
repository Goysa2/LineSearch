export update_H
"""
Update the value of the (approximation) of the second derivative depending on
the approximation we want.
If we want Newton it will compute the exact second derivative.
Secant or Improved Secant will compute approximation of the second derivative.
Inputs:
  - direction, string giving the second derivative to compute (either "Nwt",
  "Sec", "SecA")
  - h, a line model (our function)
  - t, our current point
  - φt, value of φ(t). For more information on φ see the doc for phi_dphi
  - dφt, value of φ'(t).
  - φtprec, previous value of φ(t).
  - dφtprec, previous value of φ'(t).
"""
function update_H(direction :: Symbol, h :: LineModel, t :: Float64,
                  tprec :: Float64, φt :: Float64, dφt :: Float64,
                  φtprec :: Float64, dφtprec :: Float64; kwargs...)

  if direction == :Nwt
    return hess(h, t)
  elseif direction == :Sec
    s = t-tprec
    y = dφt - dφtprec
    seck = y / s

    return seck
  elseif direction == :SecA

    s = t-tprec
    y = dφt - dφtprec

    Γ= 3 * (dφt + dφtprec) * s - 6 * (φt - φtprec)
    if (y * s + Γ) < eps(Float64) * s^2
      yt = y
    else
      yt =  y + Γ / s
    end

    seck = yt / s

    return seck
  end
end
