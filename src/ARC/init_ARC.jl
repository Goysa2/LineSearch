export init_ARC

# This function is designed to output all the parameters Necessary to perform
# the ARC line search.

function init_ARC(h :: LineModel,
                  h₀ :: Float64,
                  g₀ :: Float64,
                  g :: Array{Float64,1},
                  τ₀ :: Float64,
                  τ₁ :: Float64)

# we start by checking if the step size of 1.0 is admissible
t = 1.0
ht = obj(h, t)
gt = grad!(h, t, g)


A = Armijo(t, ht, gt, h₀, g₀, τ₀)
W = Wolfe(gt, g₀, τ₁)

if A && W
  # if 1.0 is an admissible step size we don't need to go any furter
  return (t, ht, gt, A, W, (τ₁-τ₀)*g₀, -(τ₁+τ₀)*g₀)
end

# if 1.0 satisfies the Armijo condition then we start our search for an admissible
# step size at 1.0. Otherwise we start at 0.0
if A
  t = 1.0
else
  t = 0.0
end

# Strong wolfe parameters
ɛa = (τ₁ - τ₀) * g₀
ɛb = -(τ₁ + τ₀) * g₀

return (t, ht, gt, A, W)#, ɛa, ɛb)

end
