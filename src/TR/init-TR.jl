export init_TR
function init_TR(h :: AbstractLineFunction,
                 h₀ :: Float64,
                 g₀ :: Float64,
                 g :: Array{Float64,1},
                 τ₀ :: Float64,
                 τ₁ :: Float64)

t=1.0
ht = obj(h,t)
gt = grad!(h, t, g)

A=Armijo(t,ht,gt,h₀,g₀,τ₀)
W=Wolfe(gt,g₀,τ₁)

if A && W
  return (t, ht,gt,true,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)
end

if A
  Δp = 1.0  # >=0
  Δn = -1.0  # <=0
  t=1.0
else
  Δp = 1.0  # >=0
  Δn = 0.0  # <=0
  t=0.0
end

# Specialized TR for handling non-negativity constraint on t
# Trust region parameters

ɛa = (τ₁-τ₀)*g₀
ɛb = -(τ₁+τ₀)*g₀

return (t,ht,gt,false,Δp,Δn,ɛa,ɛb)

end
