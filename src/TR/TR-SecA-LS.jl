export TR_SecA_ls
function TR_SecA_ls(h :: LineModel,
                   h₀ :: Float64,
                   g₀ :: Float64,
                   g :: Array{Float64,1};
                   kwargs...)

  (t, t_original, good_grad, ht, iter, zero, stalled) =
                  TR_generic_ls(h, h₀, g₀, g, direction = "SecA"; kwargs...)
  return (t, t_original, good_grad, ht, iter, zero, stalled)
end
