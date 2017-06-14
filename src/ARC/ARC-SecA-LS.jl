export ARC_SecA_ls
function ARC_SecA_ls(h :: AbstractLineFunction2,
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

    (t,true, ht,iter,zero,stalled_linesearch,h_f,h_g,h_h)=ARC_generic_ls(h,h₀,g₀,g,direction="SecA")

    return (t,true, ht,iter,zero,stalled_linesearch,h_f,h_g,h_h) #pourquoi le true et le 0?

end
