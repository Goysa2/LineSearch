export Biss_SecA_ls
function Biss_SecA_ls(h :: AbstractLineFunction2,
                      h₀ :: Float64,
                      g₀ :: Float64,
                      g :: Array{Float64,1};
                      γ :: Float64=0.8,
                      τ₀ :: Float64=1.0e-4,
                      τ₁ :: Float64=0.9999,
                      maxiter :: Int=50,
                      verboseLS :: Bool=false,
                      check_param :: Bool = false,
                      check_slope :: Bool = false,
                      debug :: Bool = false,
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
      return (t, t, true, ht, 0,0, false, h.f_eval, h.g_eval, h.h_eval)
    end


    (ta, φta, dφta, tb, φtb, dφtb) = trouve_intervalle_ls(h,h₀,g₀,g; kwargs...)

    t=tb
    tp=ta
    tqnp=ta
    iter=0

    φ(t) = obj(h,t) - h₀ - τ₀*t*g₀  # fonction et
    dφ(t) = grad!(h,t,g) - τ₀*g₀    # dérivée

    φt = φtb
    φtm1 = φta
    dφt = dφtb
    dφtm1 = dφta

    dφa = dφta
    dφb = dφtb
    ɛa = (τ₁-τ₀)*g₀
    ɛb = -(τ₁+τ₀)*g₀
    if weak_wolfe
      ɛb = Inf
    end

    verboseLS && println("ɛa = $ɛa ɛb = $ɛb")

    admissible = ((dφt>=ɛa) & (dφt<=ɛb))
    t_original = NaN
    tired=iter > maxiter

    debug && PyPlot.figure(1)
    debug && PyPlot.scatter([t],[φt + h₀ + τ₀*t*g₀])

    verboseLS && @printf("   iter   tp       tqnp        t        φt         dφt\n");
    verboseLS && @printf(" %4d %9.2e %9.2e  %9.2e  %9.2e  %9.2e\n", iter,tp,tqnp,t,φt,dφt);

    while !(admissible | tired) #admissible: respecte armijo et wolfe, tired: nb d'itérations
      s = t-tqnp
      y = dφt-dφtm1

      Γ = 3 * (dφt+dφtm1) * s - 6 * (φt-φtm1)
      if y*s+Γ < eps(Float64)*(s^2)
        yt=y
      else
        yt=y+Γ/s
      end

      dN=-dφt*s/yt

      if ((tp-t)*dN>0) & (dN/(tp-t)<γ)
        tplus = t + dN
        φplus = φ(tplus)
        dφplus= dφ(tplus)
        verboseLS && println("N")
      else
        tplus = (t+tp)/2
        φplus = φ(tplus)
        dφplus = dφ(tplus)
        verboseLS && println("B")
      end

      if t>tp
        if dφplus<0
          tp=t
          tqnp=t
          t=tplus
        else
          tqnp = t
          t = tplus
        end
      else
        if dφplus>0
          tp=t
          tqnp=t
          t=tplus
        else
          tqnp=t
          t=tplus
        end
      end

      φtm1=φt
      dφtm1=dφt
      φt=φplus
      dφt=dφplus

      iter=iter+1
      admissible = (dφt>=ɛa) & (dφt<=ɛb)
      tired=iter>maxiter

      if admissible && add_step && (n_add_step < 1)
        n_add_step +=1
        admissible = false
      end

      debug && PyPlot.figure(1)
      debug && PyPlot.scatter([t],[φt + h₀ + τ₀*t*g₀])

      verboseLS && @printf(" %4d %9.2e %9.2e  %9.2e  %9.2e  %9.2e\n", iter,tp,tqnp,t,φt,dφt);
    end;

    ht = φ(t) + h₀ + τ₀*t*g₀

    t > 0.0 || (verboseLS && println("t = $t"))
    @assert (t > 0.0) && (!isnan(t)) "invalid step"

    return (t, t_original, true, ht, iter,0,tired, h.f_eval, h.g_eval, h.h_eval)  #pourquoi le true et le 0?
end
