export zoom_Cub_ls2
function zoom_Cub_ls2(h :: AbstractLineFunction,
                 h₀ :: Float64,
                 g₀ :: Float64,
                 t₀ :: Float64,
                 t₁ :: Float64;
                 γ :: Float64=0.8,
                 τ₀ :: Float64=1.0e-4,
                 τ₁ :: Float64=0.9999,
                 ϵ :: Float64=1e-5,
                 maxiter :: Int=50,
                 verbose :: Bool=false,
                 kwargs...)


  (topt,false,ht,iter)=zoom_generic_ls(h,h₀,g₀,t₀,t₁,direction="Cub")
  return (topt,good_grad,ht,iter)
end
