export zoom_Nwt_ls
function zoom_Nwt_ls(h :: LineModel,
                 h₀ :: Float64,
                 g₀ :: Float64,
                 g :: Array{Float64,1};
                 kwargs...)

  (ti,good_grad,ht,iter,zero,stalled_linesearch) = find_intervalA_ls(h, h₀, g₀, g; direction = "Nwt", kwargs...)

  return (ti, ti, good_grad, ht, iter, zero, stalled_linesearch)

end
