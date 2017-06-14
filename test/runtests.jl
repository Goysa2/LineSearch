using LineSearch
using Base.Test

using JuMP, NLPModels, Optimize
using Polynomials
using LSDescentMethods
using OptimizationProblems

for ls in algorithms
  println("algo:",Symbol(ls))
  #prob_init = [:srosenbr]
  prob = MathProgNLPModel(eval(:srosenbr)(2), name=string(:srosenbr))

  (x, f, ∇fNorm, iter, optimal, tired, status)=Newton(prob,verbose=true, linesearch = ls)

end
