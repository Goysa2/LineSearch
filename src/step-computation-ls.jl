export step_computation_ls

function step_computation_ls(direction :: String,
                             h :: LineModel,
                             t :: Float64,
                             tprec :: Float64,
                             φtestTR :: Float64,
                             dφtestTR :: Float64,
                             d :: Float64,
                             φtprec :: Float64,
                             dφtprec :: Float64;
                             kwargs...)

  if direction == "Nwt"
    t = t + d

    φt = φtestTR
    dφt = dφtestTR
    ddφt = hess(h,t)

    return (t,φt,dφt,ddφt)
  end

  if direction == "Sec"
    t = t + d
    dφt = dφtestTR
    φt = φtestTR
    s = t-tprec
    y = dφt - dφtprec
    seck = y/s

    return (t,φt,dφt,seck)
  end

  if direction == "SecA"
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

    return (t,φt,dφt,seck)
  end
end
