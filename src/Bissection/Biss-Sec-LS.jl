export Biss_Sec_ls
function Biss_Sec_ls(h :: LineModel,
                 h₀ :: Float64,
                 g₀ :: Float64,
                 g :: Array{Float64,1};
                 γ :: Float64=0.8,
                 τ₀ :: Float64=1.0e-4,
                 τ₁ :: Float64=0.9999,
                 stp_ls :: TStopping_LS = TStopping_LS(),
                 verboseLS :: Bool=false,
                 debug :: Bool = false,
                 check_param :: Bool = false,
                 add_step :: Bool = true,
                 n_add_step :: Int64 = 0,
                 check_slope :: Bool = false,
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
      return (t, t, true, ht, 0, 0, false)
    end


    (ta, φta, dφta, tb, φtb, dφtb) = find_interval_ls(h,h₀,g₀,g; kwargs...)

    t=ta
    tp=tb
    tqnp=tb
    iter=0

    φ(t) = obj(h,t) - h₀ - τ₀*t*g₀  # fonction et
    dφ(t) = grad!(h,t,g) - τ₀*g₀    # dérivée

    start_ls!(h, g, stp_ls, τ₀, τ₁, h₀, g₀; kwargs...)

    φt = φta
    φtm1 = φtb
    dφt = dφta
    dφtm1 = dφtb

    dφa = dφta
    dφb = dφtb
    admissible, tired = stop_ls(stp_ls, dφt, iter; kwargs...)

    t_original = NaN
    debug && PyPlot.figure(1)
    debug && PyPlot.scatter([t],[φt + h₀ + τ₀*t*g₀])

    verboseLS && @printf("   iter   tp       tqnp        t        φt        dφt\n");
    verboseLS && @printf(" %4d %9.2e %9.2e  %9.2e  %9.2e  %9.2e\n", iter,tp,tqnp,t,φt,dφt);

    while !(admissible | tired) #admissible: respecte armijo et wolfe, tired: nb d'itérations
      s=t-tqnp
      y=dφt-dφtm1

      dN=-dφt*s/y

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
          tqnp=t
          t=tplus
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
      admissible, tired = stop_ls(stp_ls, dφt, iter; kwargs...)

      if admissible && add_step && (n_add_step < 1)
        n_add_step +=1
        admissible = false
      end

      debug && PyPlot.figure(1)
      debug && PyPlot.scatter([t],[φt + h₀ + τ₀*t*g₀])

      verboseLS && @printf(" %4d %9.2e %9.2e  %9.2e  %9.2e  %9.2e\n", iter,tp,tqnp,t,φt,dφt);
    end;

    ht = φ(t) + h₀ + τ₀*t*g₀

    @assert (t > 0.0) && (!isnan(t)) "invalid step"

    return (t, t_original,true, ht, iter,0,tired)  #pourquoi le true et le 0?
end
