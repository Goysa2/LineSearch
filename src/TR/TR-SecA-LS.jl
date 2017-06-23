export TR_SecA_ls
function TR_SecA_ls(h :: AbstractLineFunction2,
                   h₀ :: Float64,
                   g₀ :: Float64,
                   g :: Array{Float64,1};
                   kwargs...)

    #println("on est dans le SecA")
    (t, t_original,true,ht,iter,zero,stalled, h_f, h_g, h_h)=TR_generic_ls(h,h₀,g₀,g,direction="SecA";kwargs...)
    return (t, t_original,true, ht, iter,zero,stalled, h_f, h_g, h_h)

end
