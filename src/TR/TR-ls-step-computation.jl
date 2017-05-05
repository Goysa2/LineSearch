export TR_ls_step_computation

function TR_ls_step_computation(h :: Float64,
                                g :: Float64,
                                dN :: Float64,
                                Δn :: Float64,
                                Δp ::Float64)

  if h>0
    if g>0
      d=max(Δn,dN)
    else
      d=min(dN,Δp)
    end
  else
    if g>0
      d=Δn
    else
      d=Δp
    end
  end

  return d

end
