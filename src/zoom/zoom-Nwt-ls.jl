export zoom_nwt_ls
function zoom_nwt_ls(h :: LineModel,
                 h₀ :: Float64,
                 g₀ :: Float64,
                 g :: Array{Float64,1};
                 kwargs...)

  (ti,good_grad,ht,iter,zero,stalled_linesearch) = trouve_intervalleA_ls(h, h₀, g₀, g; direction = "Nwt", kwargs...)

  return (ti,good_grad,ht,iter,zero,stalled_linesearch)

end
