using LineSearch
using Base.Test

using JuMP, NLPModels, Optimize
using Polynomials
using LSDescentMethods
using OptimizationProblems

for ls in ls_algorithms
  println("algo: ",Symbol(ls))
  prob = MathProgNLPModel(eval(:srosenbr)(2), name=string(:srosenbr))

  (x, f, ∇fNorm, iter, optimal, tired, status)=Newton(prob; verbose = true,
                                                      linesearch = ls,
                                                      verboseLS = true)
  if norm(x - ones(length(prob.meta.x0)), 2) > 1e-5
    println("ERROR")
  end
end
