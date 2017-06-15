using LineSearch
using Base.Test

using JuMP, NLPModels, Optimize
using Polynomials
using LSDescentMethods
using OptimizationProblems

for ls in algorithms
  println("algo:",Symbol(ls))
  prob = MathProgNLPModel(eval(:srosenbr)(2), name=string(:srosenbr))

  (x, f, ∇fNorm, iter, optimal, tired, status)=NewtonS(prob; verbose=true,
                                                       linesearch = ls,
                                                       τ₀ = 0.4, τ₁ = 0.6,
                                                       verboseLS = true,
                                                       check_param = true)

end
