export Biss_Nwt_ls
function Biss_Nwt_ls(h :: AbstractLineFunction2,
                     h₀ :: Float64,
                     g₀ :: Float64,
                     g :: Array{Float64,1};
                     τ₀ :: Float64=1.0e-4,
                     τ₁ :: Float64=0.9999,
                     maxiter :: Int=50,
                     verboseLS :: Bool=false,
                     check_param :: Bool = false,
                     debug :: Bool = false,
                     kwargs...)

  (τ₀ == 1.0e-4) || (check_param && warn("Different linesearch parameters"))

  t = 1.0
  ht = obj(h,t)
  gt = grad!(h, t, g)
  if Armijo(t,ht,gt,h₀,g₀,τ₀) && Wolfe(gt,g₀,τ₁)
    # verboseLS && @printf("   iter   t\n");
    # verboseLS && @printf("%4d %9.2e \n", 0,1.0);
    return (t, true, ht, 0, 0, false, h.f_eval, h.g_eval, h.h_eval)
  end


  (ta, φta, dφta, tb, φtb, dφtb) = trouve_intervalle_ls(h,h₀,g₀,g, verboseLS = verboseLS, debug = debug)
  verboseLS && println("ta = $ta tb = $tb")

   γ=0.8
   t=ta
   tp=tb
   tqnp=tb
   iter=0

   φ(t) = obj(h,t) - h₀ - τ₀*t*g₀  # fonction et
   dφ(t) = grad(h,t) - τ₀*g₀    # dérivée
   ddφ(t) = hess(h,t)

   iter=0

   dφt = dφta
   dφa = dφta
   dφb = dφtb

   ɛa = (τ₁-τ₀)*g₀
   ɛb = -(τ₁+τ₀)*g₀

   verboseLS && println("ϵₐ = $ɛa ϵᵦ = $ɛb")

   admissible = ((dφt>=ɛa) & (dφt<=ɛb))
   tired =  iter>maxiter

   debug && PyPlot.figure(1)
   debug && PyPlot.scatter([t],[φ(t) + h₀ + τ₀*t*g₀])      #costs an additionnal function evaluation

   verboseLS && @printf(" iter     tqnp        t         dφtqnp     dφt        \n")
   verboseLS && @printf(" %7.2e %7.2e  %7.2e  %7.2e  %7.2e\n", iter,tqnp,t,dφa,dφb)

   while !(admissible | tired)

     ddφt=ddφ(t)
     dN=-dφt/ddφt

     if ((tp-t)*dN>0) & (dN/(tp-t)<γ)
       tplus = t + dN
       #hplus = obj(h, tplus)
       dφplus= dφ(tplus)
       verboseLS && println("N")
     else
       tplus = (t+tp)/2
       #hplus = obj(h, tplus)
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

     dφtqnp = dφt

     dφt = dφplus

     iter = iter+1
     admissible = ((dφt>=ɛa) & (dφt<=ɛb))

     debug && PyPlot.figure(1)
     debug && PyPlot.scatter([t],[φ(t) + h₀ + τ₀*t*g₀])    #costs an additionnal function evaluation

     tired =  iter>maxiter

     verboseLS && @printf(" %7.2e %7.2e  %7.2e  %7.2e  %7.2e\n", iter,tqnp,t,dφtqnp,dφt)
   end

   ht = φ(t) + h₀ + τ₀*t*g₀
   return (t,false,ht,iter,0,tired, h.f_eval, h.g_eval, h.h_eval)
end
