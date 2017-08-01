# This function is only to explain the hyper parameter of each Line Search.
# It does absolutely nothing.

# Obligatory parameter:
# h :: LineModel is a structure provided in the Optimize package (in the
#      JuliaSmoothOptimizers collection of packages).
#      If f(x) : Rⁿ  → R then h(θ) = ∇f(x + θ * d) where theta is the step size.
# h₀ :: h(0) = f(x)
# g₀ :: g(θ) = h'(θ) => g(0) = h'(0) = ∇f(x)*d. Also called the slope.
# g :: An array containing the value of the gradient of f. This gradient will
#      be updated inside the line search

# Optionnal parameters:
# stp_ls :: TStopping_LS is a type provided in the package. It's designed to
#           make the stopping criterion of all algorithms uniformed.
#           Determine if the computed step is admissible and if we have reach
#           The maximum number of iterations.
# τ₀ and τ₁ :: Armijo and Wolfe conditions parameters
# verboseLS :: If true print the results of the line search iterations by
#              iterations
# check_param :: Made to debug. If true will check if the deafult hyper
#                parameters have been changed.
# debug :: When true will trace each step of the line search
# add_step :: If we want to continue the line search after we have an admissible
#             step size
# check_slope :: Makes sure g₀ ≈ grad(h, 0.0)
# kwargs... :: if other parameters are passed through the line searc


function abstract_line_search(h :: LineModel,
                              h₀ :: Float64,
                              g₀ :: Float64,
                              g :: Array{Float64,1};
                              stp_ls :: TStopping_LS = TStopping_LS(),
                              τ₀ :: Float64 = 0.0001,
                              τ₁ :: Float64 = 0.9999,
                              verboseLS :: Bool=false,
                              check_param :: Bool = false,
                              debug :: Bool = false,
                              add_step :: Bool = false,
                              check_slope :: Bool = false,
                              kwargs...)

  (t, t_original, good_grad, ht, iter, nbW, tired) =
                                (NaN, NaN, NaN, NaN, NaN, NaN, NaN)

  # We can check if the inputs are correct if check_param & check_slope are true
  # Costs extra function evaluations so used for debugging purposes.
  (τ₀ == 1.0e-4) || (check_param && warn("Different linesearch parameters"))
  if check_slope
    (abs(g₀ - grad(h, 0.0)) < 1e-4) || warn("wrong slope")
    verboseLS && @show h₀ obj(h, 0.0) g₀ grad(h,0.0)
  end

  # We don't directly find an admissible step size using h(t) = ∇f(x + t * d).
  # Instead we use φ(t) = h(t) - h(0) - τ₀ * t * h'(0).
  φ(t) = obj(h,t) - h₀ - τ₀*t*g₀
  dφ(t) = grad!(h,t,g) - τ₀*g₀

  # We use start_ls! and stop_ls at the beginning to set up our stopping
  # criterions
  start_ls!(g, stp_ls, τ₀, τ₁, h₀, g₀; kwargs...)
  admissible, tired = stop_ls(stp_ls, dφt, iter; kwargs...)

  # Outputs:
  # t :: the (admissible) step size
  # t_original :: if we do an extra step in the line search t_original is the
  #               step size found before the extra step
  # good_grad :: Bool. If h'(t) = ∇f(x + t * d) has been computed then true
  #              otherwise false
  # ht :: h(t) = f(x + t * d)
  # iter :: number of line search iterations
  # nbW :: Always return a NaN except for Armijo backtracking process.
  # tired :: Bool. True if we have exceeded the maximum number of iterations

  return (t, t_original, good_grad, ht, iter, nbW, tired)
end
