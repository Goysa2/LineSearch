export TR_Nwt_ls
function TR_Nwt_ls(h :: AbstractLineFunction2,
                   h₀ :: Float64,
                   g₀ :: Float64,
                   g :: Array{Float64,1};
                   τ₀ :: Float64=1.0e-4,
                   eps1 :: Float64 = 0.1,
                   eps2 :: Float64 = 0.7,
                   red :: Float64 = 0.15,
                   aug :: Float64= 10.0,
                   τ₁ :: Float64=0.9999,
                   maxiter :: Int64=50,
                   verbose :: Bool=false,
                   kwargs...)

    (t,true,ht,iter,zero,stalled,h_f, h_g, h_h)=TR_generic_ls(h,h₀,g₀,g,direction="Nwt")

    return (t,true, ht, iter,zero,stalled,h_f, h_g, h_h)  #pourquoi le true et le 0?

end
