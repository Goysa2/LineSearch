export zoom_Cub_ls
function zoom_Cub_ls(h :: LineModel,
                 h₀ :: Float64,
                 g₀ :: Float64,
                 g :: Array{Float64,1};
                 verboseLS :: Bool =false,
                 kwargs...)

  (ti,good_grad,ht,iter,zero,stalled_linesearch, h_f, h_g, h_h) = trouve_intervalleA_ls(h, h₀, g₀, g, direction = "Cub"; kwargs...)

  return (ti,good_grad,ht,iter,zero,stalled_linesearch, h_f, h_g, h_h)

end
