export ARC_Sec_ls
function ARC_Sec_ls(h :: LineModel,
                   h₀ :: Float64,
                   g₀ :: Float64,
                   g :: Array{Float64,1};
                   kwargs...)

    (t, t_original, good_grad, ht,iter,zero,stalled_linesearch)=ARC_generic_ls(h,h₀,g₀,g,direction="Sec";kwargs...)

    return (t, t_original, good_grad, ht,iter,zero,stalled_linesearch)  #pourquoi le true et le 0?

end
