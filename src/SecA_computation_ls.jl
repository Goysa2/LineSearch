export SecA_computation_ls
function SecA_computation_ls(t::Float64,
                            tprec :: Float64,
                            φtprec :: Float64,
                            dφtprec :: Float64,
                            d::Float64,
                            φtestTR::Float64,
                            dφtestTR::Float64)

  t = t + d

  dφt = dφtestTR
  φt = φtestTR

  s = t-tprec
  y = dφt - dφtprec

  Γ=3*(dφt+dφtprec)*s-6*(φt-φtprec)
  if (y*s+Γ) < eps(Float64)*s^2
    yt=y
  else
    yt=y+Γ/s
  end

  seck=yt/s

  return (t,φt,dφt,s,y,seck)
end
