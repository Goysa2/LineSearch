export pred_ared_computation

"""
This function computes the actual and predicted reduction for one dimensionnal
TR or ARC algorithm. It also returns the ratio bewtween the two.
Inputs:
  - gₖ, the value of the current derivative of the function
  - fₖ, the value of the current value of the function
  - dersec, the value of the current second derivative of the function
  - d, the value of the descent direction
  - ftestTR, the candiate fonction value (h(x + t * d))
  - gtestTR, the candiate derivative value (h'(x + t * d))
  - (opt) seuil, numerical tolerance for the actual reduction. Default value
    is -1e-10.
"""
function pred_ared_computation(gₖ :: Float64, fₖ :: Float64,  dersec :: Float64,
                               d        :: Float64, ftestTR :: Float64,
                               gtestTR  :: Float64; seuil :: Float64 = -1e-10)

  pred = gₖ * d + 0.5 * dersec * d^2
  if pred > seuil
    ared = (gₖ + gtestTR)*d/2
  else
    ared = ftestTR-fₖ
  end

  ratio=ared/pred
  return (pred,ared,ratio)
end
