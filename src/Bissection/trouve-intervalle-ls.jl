export trouve_intervalle_ls
function trouve_intervalle_ls(h :: AbstractLineFunction,
                              h₀ :: Float64,
                              g₀ :: Float64,
                              inc0 :: Float64,
                              g :: Array{Float64,1};
                              τ₀ :: Float64=1.0e-4,
                              τ₁ :: Float64=0.9999,
                              nftot_max :: Int=100,
                              verbose :: Bool=false)
  g=[0.0]
  t=1.0
  ht = obj(h,t)
  gt = grad!(h, t, g)
  if Armijo(t,ht,gt,h₀,g₀,τ₀) && Wolfe(gt,g₀,τ₁)
      admissible=true
      return (t,0.0, admissible, ht, 0, 0)
  end
  iter=0
  t=0.0
  inc=inc0
  sd=-sign(g₀)
  t₁=t+sd*inc

  φ(t) = obj(h,t) - h₀ - τ₀*t*g₀  # fonction et
  dφ(t) = grad!(h,t,g) - τ₀*g₀    # dérivée

  φt = 0.0          # on sait que φ(0)=0
  dφt = (1.0-τ₀)*g₀ # connu dφ(0)=(1.0-τ₀)*g₀
  φt1=φ(t₁)
  dφt1=dφ(t₁)

  ɛa = (τ₁-τ₀)*g₀
  ɛb = -(τ₁+τ₀)*g₀
  nftot=h.nlp.counters.neval_obj+h.nlp.counters.neval_grad+h.nlp.counters.neval_hprod
  verbose && @printf("iter t        dφt        φt         t1        φt1        dφt1\n")
  verbose && @printf("%4d %7.2e %7.2e  %7.2e  %7.2e  %7.2e  %7.2e \n", iter, t,φt,dφt,t₁,dφt1,φt1)

  while (dφt1*sd<0.0) & (φt1<φt) & (nftot<nftot_max)
    inc=inc*4
    t=t₁
    φt=φt1
    dφt=dφt1
    t₁=t+sd*inc
    φt1=φ(t₁)
    dφt1=dφ(t₁)
    verbose && @printf("%4d %7.2e %7.2e  %7.2e  %7.2e  %7.2e  %7.2e \n", iter, t,φt,dφt,t₁,dφt1,φt1)
    nftot=h.nlp.counters.neval_obj+h.nlp.counters.neval_grad+h.nlp.counters.neval_hprod
    iter=iter+1

  end

  while (dφt1*sd<0.0) & (nftot<nftot_max)
    tₘ=(t₁+t)/2
    φₘ=φ(tₘ)
    dφₘ=dφ(tₘ)
    if φₘ*sd>0
      t₁=tₘ
      φt1=φₘ
      dφt1=dφₘ
    else
      if φₘ<φt
        t=tₘ
        φt=φₘ
        dφt=dφₘ
      else
        t₁=tₘ
        φt1=φₘ
        dφt1=dφₘ
      end
    end
    verbose && @printf("%4d %7.2e %7.2e  %7.2e  %7.2e  %7.2e  %7.2e \n", iter, t,φt,dφt,t₁,dφt1,φt1)
    nftot=h.nlp.counters.neval_obj+h.nlp.counters.neval_grad+h.nlp.counters.neval_hprod
    iter=iter+1
  end

  a=min(t,t₁)
  b=max(t,t₁)

  admissible=false

  return (a,b, admissible, ht, 0, 0)

end
