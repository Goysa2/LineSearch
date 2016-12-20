export Biss_Nwt_ls
function Biss_Nwt_ls(h :: AbstractLineFunction,
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
 dφ(t) = grad(h,t) - τ₀*g₀    # dérivée
 ddφ(t) = hess(h,t)

 iter=0

 dφt=dφ(t)
 dφa=dφ(ta)
 dφb=dφ(tb)

 ɛa = (τ₁-τ₀)*g₀
 ɛb = -(τ₁+τ₀)*g₀

 admissible=false
 tired =  iter>maxiter

 verbose && @printf(" iter        ta        tb         dφa        dφb        \n")
 verbose && @printf(" %7.2e %7.2e  %7.2e  %7.2e  %7.2e\n", iter,tqnp,t,dφa,dφb)

 while !(admissible | tired)

   ddφt=ddφ(t)
   dN=-dφt/ddφt

   if ((tp-t)*dN>0) & (dN/(tp-t)<γ)
     tplus = t + dN
     #hplus = obj(h, tplus)
     dφplus= dφ(tplus)
     verbose && println("N")
   else
     tplus = (t+tp)/2
     #hplus = obj(h, tplus)
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

   dφt=dφplus

   iter=iter+1
   admissible = (dφt>=ɛa) & (dφt<=ɛb)
   tired =  iter>maxiter

   verbose && @printf(" %7.2e %7.2e  %7.2e  %7.2e  %7.2e\n", iter,tqnp,t,dφa,dφb)
 end

 ht = φ(t) + h₀ + τ₀*t*g₀
 return (t,false,ht,iter,0)
end
