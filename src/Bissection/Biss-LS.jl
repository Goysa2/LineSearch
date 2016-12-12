export Biss_ls
function Biss_ls(h :: AbstractLineFunction,
                 h₀ :: Float64,
                 g₀ :: Float64,
                 g :: Array{Float64,1};
                 τ₀ :: Float64=1.0e-4,
                 τ₁ :: Float64=0.9999,
                 nftot_max :: Int64=100,
                 verbose :: Bool=false)

inc0=g[1]

(ta,tb, admissible, ht,iter)=trouve_intervalle_ls(h,h₀,g₀,inc0,g)
if admissible==true
  nftot=h.nlp.counters.neval_obj+h.nlp.counters.neval_grad+h.nlp.counters.neval_hprod
  return (ta,true, ht,nftot)
end

g=[0.0]

 φ(t) = obj(h,t) - h₀ - τ₀*t*g₀  # fonction et
 dφ(t) = grad(h,t) - τ₀*g₀    # dérivée

 iter=0

 dφa=dφ(ta)
 dφb=dφ(tb)

 ɛa = (τ₁-τ₀)*g₀
 ɛb = -(τ₁+τ₀)*g₀

 admissible=false
 nftot=h.nlp.counters.neval_obj+h.nlp.counters.neval_grad+h.nlp.counters.neval_hprod
 tired=nftot > nftot_max

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
   nftot=h.nlp.counters.neval_obj+h.nlp.counters.neval_grad+h.nlp.counters.neval_hprod
   tired=nftot > nftot_max

   verbose && @printf(" %7.2e %7.2e  %7.2e  %7.2e  %7.2e\n", iter,ta,tb,dφa,dφb)
 end

 t=(ta+tb)/2


 ht = φ(t) + h₀ + τ₀*t*g₀
 return (t,true,ht,nftot)
end
