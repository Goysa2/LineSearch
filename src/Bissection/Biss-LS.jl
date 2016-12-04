function Biss_ls(h :: AbstractLineFunction,
                 h₀ :: Float64,
                 g₀ :: Float64,
                 ta :: Float64,
                 tb :: Float64;
                 τ₀ :: Float64=1.0e-4,
                 τ₁ :: Float64=0.9999,
                 tol :: Float64=1e-7,
                 maxiter :: Int=50,
                 verbose=true)


 φ(t) = obj(h,t) - h₀ - τ₀*t*g₀  # fonction et
 dφ(t) = grad(h,t) - τ₀*g₀    # dérivée

 iter=0

 dφa=dφ(ta)
 dφb=dφ(tb)

 ɛa = (τ₁-τ₀)*min(dφa,dφb)
 ɛb = -(τ₁+τ₀)*min(dφa,dφb)
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
