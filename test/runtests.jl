using LineSearch
using Base.Test

using JuMP
using NLPModels
using Optimize
using Roots
using ScalarOptimizationProblems

nbSolver=0
nbProb=0

for algo in algorithms
  nbSolver += 1
  println("algo:",Symbol(algo))

  for prob in problem_test
    nbProb+=1
    println("prob:",nbProb%2)
    model=MathProgNLPModel(prob)
    if (nbProb%2)==1
      Ivl1 = 2.7
      Ivl2 = 7.5
    elseif (nbProb%2)==0
      Ivl1 = -float(pi)
      Ivl2 = 2*float(pi)
    end

    h=C1LineFunction(model,[Ivl1],[1.0])
    hh=C2LineFunction(model,[Ivl1],[1.0])
    h₀=obj(h,0.0)
    g₀=grad(h,0.0)
    k₀=hess(hh,0.0)
    inc0=abs(g₀/k₀)
    g=[0.0]
    reset!(model)

    if algo in algo_biss
      (ta,tb, admissible, ht,iter)=trouve_intervalle_ls(h,h₀,g₀,inc0,g)
      if admissible==true
        topt=ta
      else
        (topt, admissible,ht,iter) = algo(hh,h₀,g₀,ta,tb,maxiter=30,verbose=true)
      end
      println("(topt, iter)=",(topt, iter))
      nftot = model.counters.neval_obj + model.counters.neval_grad + model.counters.neval_hprod
      println("Total functions and derivatives: ",nftot, "  iterations: ",iter)
    elseif algo in zoom_methods
      (topt, admissible, ht,iter)=trouve_intervalleA_ls(hh,h₀,g₀,0.0,Ivl2,algo,g)
      println("(topt, iter)=",(topt, iter))
      nftot = model.counters.neval_obj + model.counters.neval_grad + model.counters.neval_hprod
      println("Total functions and derivatives: ",nftot, "  iterations: ",iter)
    else
      (topt, admissible,ht,iter) = algo(hh,h₀,g₀,g,maxiter=30)
      println("(topt, iter)=",(topt, iter))
      nftot = model.counters.neval_obj + model.counters.neval_grad + model.counters.neval_hprod
      println("Total functions and derivatives: ",nftot, "  iterations: ",iter)
    end

  end


end
