export Biss_SecA_ls
function Biss_SecA_ls(h :: AbstractLineFunction,
                 h₀ :: Float64,
                 g₀ :: Float64,
                 ta :: Float64,
                 tb :: Float64;
                 τ₀ :: Float64=1.0e-4,
                 τ₁ :: Float64=0.9999,
                 tol :: Float64=1e-7,
                 maxiter :: Int=50,
                 verbose=true)

 γ=0.8
 t=ta
 tp=tb
 tqnp=tb
 iter=0

 φ(t) = obj(h,t) - h₀ - τ₀*t*g₀  # fonction et
 dφ(t) = grad(h,t) - τ₀*g₀    # dérivée


 iter=0

 φt=φ(t)
 φtm1=φ(tqnp)
 dφt=dφ(t)
 dφtm1=dφ(tqnp)

 dφa=dφ(ta)
 dφb=dφ(tb)

 ɛa = (τ₁-τ₀)*g₀
 ɛb = -(τ₁+τ₀)*g₀
 verbose && println("\n ɛa=",ɛa," ɛb=",ɛb," h(0)=", h₀," g₀=",g₀)

 admissible=false
 tired =  iter > maxiter

 verbose && @printf(" iter        ta        tb         dφa        dφb        \n")
 verbose && @printf(" %7.2e %7.2e  %7.2e  %7.2e  %7.2e\n", iter,ta,tb,dφa,dφb)

 while !(admissible | tired)

   s=t-tqnp
   y=dφt-dφtm1

   Γ=3*(dφt+dφtm1)*s-6*(φt-φtm1)
   if y*s+Γ < eps(Float64)*(s^2)
     yt=y
   else
     yt=y+Γ/s
   end

   dN=-dφt*s/yt

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
   tired = iter > maxiter

   verbose && @printf(" %7.2e %7.2e  %7.2e  %7.2e  %7.2e\n", iter,ta,tb,dφa,dφb)
 end
 println("admissible=",admissible)


 ht = φ(t) + h₀ + τ₀*t*g₀
 return (t,admissible,ht,iter,0)
end
