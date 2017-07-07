export Biss_ls
function Biss_ls(h :: AbstractLineFunction2,
                 h₀ :: Float64,
                 g₀ :: Float64,
                 g :: Array{Float64,1};
                 τ₀ :: Float64=1.0e-4,
                 τ₁ :: Float64=0.9999,
                 maxiter :: Int=50,
                 verboseLS :: Bool=false,
                 check_param :: Bool = false,
                 check_slope :: Bool = false,
                 add_step :: Bool = true,
                 n_add_step :: Int64 = 0,
                 weak_wolfe :: Bool = false,
                 kwargs...)

    (τ₀ == 1.0e-4) || (check_param && warn("Different linesearch parameters"))
    if check_slope
      (abs(g₀ - grad(h, 0.0)) < 1e-4) || warn("wrong slope")
      verboseLS && @show h₀ obj(h, 0.0) g₀ grad(h,0.0)
    end


    t = 1.0
    ht = obj(h,t)
    gt = grad!(h, t, g)
    if Armijo(t,ht,gt,h₀,g₀,τ₀) && Wolfe(gt,g₀,τ₁)
      return (t, t, true, ht, 0, 0, false, h.f_eval, h.g_eval, h.h_eval)
    end


    (ta, φta, dφta, tb, φtb, dφtb) = trouve_intervalle_ls(h,h₀,g₀,g; kwargs...)
    #println("a la sorti de trouve_intervalle_ls ta=",ta," tb=",tb)
    φ(t) = obj(h,t) - h₀ - τ₀*t*g₀  # fonction et
    dφ(t) = grad!(h,t,g) - τ₀*g₀    # dérivée

    tp=(ta+tb)/2

    iter=0

    # test d'arrêt sur dφ
    ɛa = (τ₁-τ₀)*g₀
    ɛb = -(τ₁+τ₀)*g₀
    if weak_wolfe
      ɛb = Inf
    end

    admissible = false
    t_original = NaN
    tired=iter > maxiter
    verboseLS && @printf("   iter   ta       tb        tp        dφp\n");
    verboseLS && @printf(" %4d %9.2e %9.2e  %9.2e  %9.2e \n", iter,ta,tb,tp,NaN);

    while !(admissible | tired) #admissible: respecte armijo et wolfe, tired: nb d'itérations
      tp=(ta+tb)/2
      dφp=dφ(tp)

      if dφp<=0
        ta=tp
        dφa=dφp
      else
        tb=tp
        dφb=dφp
      end

      iter=iter+1
      admissible = (dφp>=ɛa) & (dφp<=ɛb)
      tired=iter>maxiter

      if admissible && add_step && (n_add_step < 1)
        n_add_step +=1
        admissible = false
      end

      verboseLS && @printf(" %4d %9.2e %9.2e  %9.2e  %9.2e\n", iter,ta,tb,tp,dφp);
    end;

    ht = φ(t) + h₀ + τ₀*t*g₀

    @assert (t > 0.0) && (!isnan(t)) "invalid step"

    return (tp, t_original, true, ht, iter,0,tired, h.f_eval, h.g_eval, h.h_eval)  #pourquoi le true et le 0?
end
