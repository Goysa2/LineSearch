export Biss_Cub_ls
function Biss_Cub_ls(h :: AbstractLineFunction2,
                     h₀ :: Float64,
                     g₀ :: Float64,
                     g :: Array{Float64,1};
                     γ :: Float64=0.55,
                     τ₀ :: Float64=1.0e-4,
                     τ₁ :: Float64=0.9999,
                     maxiter :: Int=100,
                     verboseLS :: Bool=false,
                     check_param :: Bool = false,
                     debug :: Bool = false,
                     kwargs...)

  (τ₀ == 1.0e-4) || (check_param && warn("Different linesearch parameters"))

 t = 1.0
 ht = obj(h,t)
 gt = grad!(h, t, g)
 if Armijo(t,ht,gt,h₀,g₀,τ₀) && Wolfe(gt,g₀,τ₁)
   return (t, true, ht, 0, 0, false, h.f_eval, h.g_eval, h.h_eval)
 end

 #println("au début de Biss_Cub_ls g₀=",g₀)

 (ta, φta, dφta, tb, φtb, dφtb) = trouve_intervalle_ls(h,h₀,g₀,g)

 #println("on est après le trouve_intervalle_ls")

 γ=0.8
 t=tb
 tp=ta
 tqnp=ta
 iter=0

 φ(t) = obj(h,t) - h₀ - τ₀*t*g₀  # fonction et
 dφ(t) = grad(h,t) - τ₀*g₀    # dérivée

 #println("ta=",ta," tb=",tb)
 #println("dφa=",dφ(ta)," dφb=",dφ(tb))


 iter=0

 φt=φta
 φtm1=φtb
 dφt=dφta
 dφtm1=dφtb

 dφa=dφta
 dφb=dφtb

 ɛa = (τ₁-τ₀)*g₀
 ɛb = -(τ₁+τ₀)*g₀
 admissible = ((dφt>=ɛa) & (dφt<=ɛb))
 tired =  iter > maxiter

 debug && PyPlot.figure(1)
 debug && PyPlot.scatter([t],[φt + h₀ + τ₀*t*g₀])

 verboseLS && @printf(" iter        tqnp        t         dφtm1        dφt        \n")
 verboseLS && @printf(" %7.2e %7.2e  %7.2e  %7.2e  %7.2e\n", iter,tqnp,t,dφtm1,dφt)

 while !(admissible | tired)

   s=t-tqnp
   y=dφt-dφtm1

   α=-s
   z=dφt+dφtm1+3*(φt-φtm1)/α
   discr=z^2-dφt*dφtm1
   denom=dφt+dφtm1+2*z
   if (discr>0) & (abs(denom)>eps(Float64))
     #si on peut on utilise l'interpolation cubique
     w=sqrt(discr)
     dN=-s*(dφt+z+sign(α)*w)/(denom)
   else #on se rabat sur une étape de sécante
     dN=-dφt*s/y
   end

   if ((tp-t)*dN>0) & (dN/(tp-t)<γ)
     tplus = t + dN
     φplus = obj(h, tplus)
     dφplus= dφ(tplus)
     verboseLS && println("N")
   else
     tplus = (t+tp)/2
     φplus = obj(h, tplus)
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
   #println("tp=",tp," t=",t)
   φtm1=φt
   dφtm1=dφt
   φt=φplus
   dφt=dφplus

   iter=iter+1
   admissible = (dφt>=ɛa) & (dφt<=ɛb)
   tired=iter>maxiter

   debug && PyPlot.figure(1)
   debug && PyPlot.scatter([t],[φt + h₀ + τ₀*t*g₀])

   verboseLS && @printf(" %7.2e %7.2e  %7.2e  %7.2e  %7.2e\n", iter,tqnp,t,dφtm1,dφt)
 end

 ht = φ(t) + h₀ + τ₀*t*g₀
 return (t,false,ht,iter,0,tired, h.f_eval, h.g_eval, h.h_eval)
end
