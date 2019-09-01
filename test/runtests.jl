@static if VERSION < v"0.7.0-DEV.2005"
    using Base.Test
else
    using Test
end

using State
using NLPModels
using OptimizationProblems
using Stopping
using LineSearch
using LinearAlgebra
using NLPModelsJuMP
using JuMP

# algorithms which reach a solution on that problem
algorithms_good = [:armijo_ls, :TR_Nwt_ls, :TR_Sec_ls, :TR_SecA_ls, :ARC_Nwt_ls]

for algo in algorithms_good
   println(" ")
   println("$algo")
   nlp = MathProgNLPModel(arwhead());
   h = LineModel(nlp, nlp.meta.x0, -grad(nlp,nlp.meta.x0));
   lsatx = LSAtT(0.0, h₀ = obj(h, 0.0), g₀ = grad(h, 0.0),
                      ht = obj(h, 0.0), gt = grad(h, 0.0))
   stop_meta = StoppingMeta(max_iter = 50)
   ls_meta = LS_Function_Meta()
   stop_ls = LS_Stopping(h, (x,y)-> armijo(x, y), lsatx, meta = stop_meta);

   state, optimality = eval(algo)(h, stop_ls, f_meta = ls_meta, verboseLS = true)
   @test optimality == true
   reset!(nlp)
end

# algorithms which cannot reach a solution on that problem
algorithms_bad = [:ARC_Sec_ls, :ARC_SecA_ls]

for algo in algorithms_bad
   println(" ")
   println("$algo")
   nlp = MathProgNLPModel(arwhead());
   h = LineModel(nlp, nlp.meta.x0, -grad(nlp,nlp.meta.x0));
   lsatx = LSAtT(0.0, h₀ = obj(h, 0.0), g₀ = grad(h, 0.0),
                      ht = obj(h, 0.0), gt = grad(h, 0.0))
   stop_ls = LS_Stopping(h, (x,y)-> armijo(x, y), lsatx);
   stop_meta = StoppingMeta(max_iter = 50)

   state, optimality = eval(algo)(h, stop_ls, verboseLS = true)
   @test optimality == false
   reset!(nlp)
end
