export zoom_ls
function zoom_ls(h :: AbstractLineFunction2,
                 h₀ :: Float64,
                 g₀ :: Float64,
                 g :: Array{Float64,1};
                 τ₀ :: Float64=1.0e-4,
                 τ₁ :: Float64=0.9999,
                 ϵ :: Float64=1e-5,
                 maxiter :: Int=50,
                 verbose :: Bool=false,
                 kwargs...)

  (ti,good_grad,ht,iter,zero,stalled_linesearch, h_f, h_g, h_h) = trouve_intervalleA_ls(h, h₀, g₀, g; direction = "Biss", kwargs...)

  return (ti,good_grad,ht,iter,zero,stalled_linesearch, h_f, h_g, h_h)

end
