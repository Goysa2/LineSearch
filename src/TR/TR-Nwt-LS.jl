export TR_Nwt_ls
function TR_Nwt_ls(h :: AbstractLineFunction2,
                   h₀ :: Float64,
                   g₀ :: Float64,
                   g :: Array{Float64,1};
                   kwargs...)

    (t, t_original, good_grad,ht,iter,zero,stalled,h_f, h_g, h_h)=TR_generic_ls(h,h₀,g₀,g,direction="Nwt";kwargs...)

    return (t, t_original, good_grad, ht, iter,zero,stalled,h_f, h_g, h_h)  #pourquoi le true et le 0?

end
