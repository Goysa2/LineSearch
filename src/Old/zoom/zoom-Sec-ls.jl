export zoom_Sec_ls
function zoom_Sec_ls(h :: LineModel,
                     h₀ :: Float64,
                     g₀ :: Float64,
                     g :: Array{Float64,1};
                     kwargs...)

  (ti, ti, good_grad, ht, iter, zero, stalled_linesearch) =
                  find_intervalA_ls(h, h₀, g₀, g; direction = "Sec", kwargs...)

  return (ti, ti, good_grad, ht, iter, zero, stalled_linesearch)

end
