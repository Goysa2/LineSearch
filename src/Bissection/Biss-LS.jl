export Biss_ls
function Biss_ls(h :: AbstractLineFunction,
                 h₀ :: Float64,
                 g₀ :: Float64,
                 g :: Array{Float64,1};
                 τ₀ :: Float64=1.0e-4,
                 τ₁ :: Float64=0.9999,
                 bk_max :: Int=10,
                 nbWM :: Int=5,
                 verbose :: Bool=false)

maxiter=nbWM*bk_max

inc0=g[1]

(ta,tb, admissible, ht,iter)=trouve_intervalle_ls(h,h₀,g₀,inc0,g,verbose=false)
if admissible==true
  return (ta, admissible, ht,iter)
end

g=[0.0]

 φ(t) = obj(h,t) - h₀ - τ₀*t*g₀  # fonction et
 dφ(t) = grad(h,t) - τ₀*g₀    # dérivée

 iter=0

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
   tired = iter > maxiter

   verbose && @printf(" %7.2e %7.2e  %7.2e  %7.2e  %7.2e\n", iter,ta,tb,dφa,dφb)
 end
 println("admissible=",admissible)

 t=(ta+tb)/2


 ht = φ(t) + h₀ + τ₀*t*g₀
 return (t,admissible,ht,iter,0)
end
