export ARC_Sec_ls
function ARC_Sec_ls(h :: AbstractLineFunction,
                   h₀ :: Float64,
                   g₀ :: Float64,
                   g :: Array{Float64,1};
                   eps1 = 0.1,
                   eps2 = 0.7,
                   red = 0.15,
                   aug = 10,
                   α=1.0,
                   τ₀ :: Float64=1.0e-4,
                   τ₁ :: Float64=0.9999,
                   maxiter :: Int64=50,
                   verbose :: Bool=false,
                   kwargs...)

    (t,true, ht,iter,zero)=ARC_generic_ls(h,h₀,g₀,g,direction="Sec")

    return (t,true, ht,iter,zero)  #pourquoi le true et le 0?

end
