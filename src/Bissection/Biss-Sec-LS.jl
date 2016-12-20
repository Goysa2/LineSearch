export Biss_Sec_ls
function Biss_Sec_ls(h :: AbstractLineFunction,
                 h₀ :: Float64,
                 g₀ :: Float64,
                 g :: Array{Float64,1};
                 τ₀ :: Float64=1.0e-4,
                 τ₁ :: Float64=0.9999,
                 maxiter :: Int=50,
                 verbose :: Bool=false)

    t = 1.0
    ht = obj(h,t)
    gt = grad!(h, t, g)
    if Armijo(t,ht,gt,h₀,g₀,τ₀) && Wolfe(gt,g₀,τ₁)
      return (t, true, ht, 0,0)
    end


    (ta,tb)=trouve_intervalle_ls(h,h₀,g₀,g)

    γ=0.8
    t=ta
    tp=tb
    tqnp=tb
    iter=0

    φ(t) = obj(h,t) - h₀ - τ₀*t*g₀  # fonction et
    dφ(t) = grad!(h,t,g) - τ₀*g₀    # dérivée

    φt=φ(t)
    φtm1=φ(tqnp)
    dφt=dφ(t)
    dφtm1=dφ(tqnp)

    dφa=dφ(ta)
    dφb=dφ(tb)
    ɛa = (τ₁-τ₀)*g₀
    ɛb = -(τ₁+τ₀)*g₀

    admissible = false
    tired=iter > maxiter
    verbose && @printf("   iter   tp       tqnp        t        dφt\n");
    verbose && @printf(" %4d %9.2e %9.2e  %9.2e  %9.2e\n", iter,tp,tqnp,t,φt);

    while !(admissible | tired) #admissible: respecte armijo et wolfe, tired: nb d'itérations
      s=t-tqnp
      y=dφt-dφtm1

      dN=-dφt*s/y

      if ((tp-t)*dN>0) & (dN/(tp-t)<γ)
        tplus = t + dN
        φplus = obj(h, tplus)
        dφplus= dφ(tplus)
        verbose && println("N")
      else
        tplus = (t+tp)/2
        φplus = obj(h, tplus)
        dφplus = dφ(tplus)
        verbose && println("B")
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

      admissible = (dφt>=ɛa) & (dφt<=ɛb)
      tired=iter>maxiter

      verbose && @printf(" %4d %9.2e %9.2e  %9.2e  %9.2e\n", iter,tp,tqnp,t,φt);
    end;

    #println("après le while \n")
    #println("ta=",ta," tb=",tb)

    ht = φ(t) + h₀ + τ₀*t*g₀
    #println("on a ht \n")
    return (t,false, ht, iter,0)  #pourquoi le true et le 0?
end
