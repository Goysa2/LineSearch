export ARC_Sec_ls
function ARC_Sec_ls(h :: AbstractLineFunction2,
                   h₀ :: Float64,
                   g₀ :: Float64,
                   g :: Array{Float64,1};
                   kwargs...)

    (t, t_original, true, ht,iter,zero,stalled_linesearch,h_f,h_g,h_h)=ARC_generic_ls(h,h₀,g₀,g,direction="Sec";kwargs...)

    return (t, t_original, true, ht,iter,zero,stalled_linesearch,h_f,h_g,h_h)  #pourquoi le true et le 0?

end
