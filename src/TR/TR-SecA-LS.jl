export TR_SecA_ls
function TR_SecA_ls(h :: AbstractLineFunction,
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

    #println("on est dans le SecA")
    (t,true,ht,iter,zero,stalled)=TR_generic_ls(h,h₀,g₀,g,direction="SecA")
    return (t,true, ht, iter,zero,stalled)

end
