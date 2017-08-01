export pred_ared_computation
function pred_ared_computation(gₖ::Float64,
                               fₖ::Float64,
                               dersec::Float64,
                               d::Float64,
                               ftestTR::Float64,
                               gtestTR::Float64;
                               seuil::Float64=-1e-10)

  pred = gₖ * d + 0.5 * dersec * d^2

  if pred > seuil
    #print_with_color(:yellow,"!!")
    ared = (gₖ + gtestTR)*d/2
  else
    #print_with_color(:green,"!!")
    ared = ftestTR-fₖ
  end

  ratio=ared/pred

  return (pred,ared,ratio)
end
