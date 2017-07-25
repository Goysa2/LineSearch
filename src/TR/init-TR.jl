export init_TR

# This function is designed to output the Necessary parameters required to
# perform a trust region line search

function init_TR(h :: LineModel,
                 h₀ :: Float64,
                 g₀ :: Float64,
                 g :: Array{Float64,1},
                 τ₀ :: Float64,
                 τ₁ :: Float64;
                 Δ :: Float64 = 10.0,
                 kwargs...)

# we start by checking if the step size of 1.0 is admissible

t = 1.0
ht = obj(h, t)
gt = grad!(h, t, g)

A=Armijo(t, ht, gt, h₀, g₀, τ₀)
W=Wolfe(gt, g₀, τ₁)

if A && W
  # if 1.0 is an admissible step size we don't need to go any furter
  return (t, ht, gt, true, 0.0, 0.0, 0.0, 0.0)
end

# if the 1.0 satisfies the Armijo condition we start with a trust region that
# can be slightly negative. Otherwise the trust region is has values that are
# greater or equal to 0.0
if A
  Δp = Δ
  Δn = -1.0
  t = 1.0
else
  Δp = Δ
  Δn = max(0.0, -Δ)
  t = 0.0
end

# Strong wolfe parameters
ɛa = (τ₁-τ₀)*g₀
ɛb = -(τ₁+τ₀)*g₀

return (t, ht, gt, false, Δp, Δn)#, ɛa, ɛb)
# Outputs:
# t :: where we'll start our algorithm
# ht :: h(t) = f(x + t⋆d)
# gt :: g(t) = h'(t) = ∇f(x + t⋆d)d
# true :: If Armijo & Wolfe satisfied, false otherwise
# Δp :: Positive bound of the trust region
# Δn :: Negative bound of the trust region
# εa & εb :: Strong Wolfe bounds
end
