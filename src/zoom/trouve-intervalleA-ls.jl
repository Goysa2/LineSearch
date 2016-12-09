export trouve_intervalleA_ls
function trouve_intervalleA_ls(h :: AbstractLineFunction,
                               h₀ :: Float64,
                               g₀ :: Float64,
                               g :: Array{Float64,1};
                               methode :: Function=zoom_ls,
                               τ₀ :: Float64=1.0e-4,
                               τ₁ :: Float64=0.9999,
                               t₀ :: Float64=0.0,
                               tmax :: Float64=50.0,
                               max_eval :: Int=100,
                               verbose :: Bool=false)

    t=1.0
    ht = obj(h,t)
    gt = grad!(h, t, g)
    if Armijo(t,ht,gt,h₀,g₀,τ₀) && Wolfe(gt,g₀,τ₁)
        return (t,true,ht,0)
    end

    φ(t) = obj(h,t) - h₀ - τ₀*t*g₀  # fonction et
    dφ(t) = grad!(h,t,g) - τ₀*g₀    # dérivée

    tim1=t₀
    ti=(tim1+tmax)/2

    i=0.0

    φtim1=0.0            #t₀=0.0 => φ(0)=0.0
    dφtim1=(1.0-τ₀)*g₀   #t₀=0.0 => dφ(0)=(1.0-τ₀)*g₀
    φti=φ(ti)
    dφti=dφ(ti)

    ɛa = (τ₁-τ₀)*g₀
    ɛb = -(τ₁+τ₀)*g₀

    φtim1=0.0 #t₀=0.0 => φ(0)=0.0
    dφtim1=(1.0-τ₀)*g₀   #t₀=0.0 => dφ(0)=(1.0-τ₀)*g₀
    nftot=h.nlp.counters.neval_obj+h.nlp.counters.neval_grad+h.nlp.counters.neval_hprod

    verbose && @printf("iter tim1        dφtim1        φtim1         ti        dφti        φti\n")
    verbose && @printf("%4d %7.2e %7.2e  %7.2e  %7.2e  %7.2e  %7.2e \n", i, tim1,dφtim1,φtim1,ti,dφti,φti)

    while nftot<max_eval
      φtim1=φti
      φti=φ(ti)
      if (φti>0.0) | ((φti>φtim1) & (i>1))
        (topt,good_grad,ht,i)=methode(h,h₀,g₀,tim1,ti,verbose=false)
        return (topt,good_grad,ht,i)
      end

      dφtim1=dφti
      dφti=dφ(ti)

      if ((dφti>=ɛa) & (dφti<=ɛb))
        iter=i
        topt=ti
        ht= φti + h₀ + τ₀*ti*g₀
        good_grad=true
        nftot=h.nlp.counters.neval_obj+h.nlp.counters.neval_grad+h.nlp.counters.neval_hprod
        return (topt,good_grad,ht,nftot)
      end

      if (dφti>= -t₀*h₀)
        (topt,good_grad,ht,i)=methode(h,h₀,g₀,ti,tim1,verbose=true)
        return (topt,good_grad,ht,i)
      end
      tim1=ti
      ti=(tim1+tmax)/2

      i=i+1
      nftot=h.nlp.counters.neval_obj+h.nlp.counters.neval_grad+h.nlp.counters.neval_hprod
      verbose && @printf("%4d %7.2e %7.2e  %7.2e  %7.2e  %7.2e  %7.2e \n", i, tim1,dφtim1,φtim1,ti,dφti,φti)
    end
    iter=i
    ht= φti + h₀ + τ₀*ti*g₀
    nftot=h.nlp.counters.neval_obj+h.nlp.counters.neval_grad+h.nlp.counters.neval_hprod
    return (t,true,ht,nftot)

end
