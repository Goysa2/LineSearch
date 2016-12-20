export trouve_intervalleA_ls
function trouve_intervalleA_ls(h :: AbstractLineFunction,
                               h₀ :: Float64,
                               g₀ :: Float64,
                               g :: Array{Float64,1};
                               methode :: Function=zoom_ls,
                               τ₀ :: Float64=1.0e-4,
                               τ₁ :: Float64=0.9999,
                               t₀ :: Float64=0.0,
                               tmax :: Float64=1000.0,
                               maxiter :: Int=50,
                               verbose :: Bool=false)

    #println("TROUVE_INTERVALLEA_LS")

    t=1.0
    ht = obj(h,t)
    gt = grad!(h, t, g)
    if Armijo(t,ht,gt,h₀,g₀,τ₀) && Wolfe(gt,g₀,τ₁)
        return (t,true,ht,0,0)
    end

    φ(t) = obj(h,t) - h₀ - τ₀*t*g₀  # fonction et
    dφ(t) = grad!(h,t,g) - τ₀*g₀    # dérivée

    tim1=t₀
    ti=(tim1+tmax)/2

    iter=0

    φtim1=0.0            #t₀=0.0 => φ(0)=0.0
    dφtim1=(1.0-τ₀)*g₀   #t₀=0.0 => dφ(0)=(1.0-τ₀)*g₀
    φti=φ(ti)
    dφti=dφ(ti)

    ɛa = (τ₁-τ₀)*g₀
    ɛb = -(τ₁+τ₀)*g₀

    #println("dans zoom_ls ɛa=",ɛa," ɛb=",ɛb)

    φtim1=0.0 #t₀=0.0 => φ(0)=0.0
    dφtim1=(1.0-τ₀)*g₀   #t₀=0.0 => dφ(0)=(1.0-τ₀)*g₀

    verbose && @printf("iter tim1        dφtim1        φtim1         ti        dφti        φti\n")
    verbose && @printf("%4d %7.2e %7.2e  %7.2e  %7.2e  %7.2e  %7.2e \n", iter, tim1,dφtim1,φtim1,ti,dφti,φti)

    while iter<maxiter
      φtim1=φti
      φti=φ(ti)
      if (φti>0.0) | ((φti>φtim1) & (iter>1))
        #println("premier appel de zoom")
        #println("tim1=",tim1," ti=",ti)
        (topt,good_grad,ht,i)=methode(h,h₀,g₀,tim1,ti)
        return (topt,good_grad,ht,iter,0)
      end

      dφtim1=dφti
      dφti=dφ(ti)

      if ((dφti>=ɛa) & (dφti<=ɛb))
        topt=ti
        ht= φti + h₀ + τ₀*ti*g₀
        #println("admissible dans TROUVE_INTERVALLESA_LS")
        return (topt,false,ht,iter,0)
      end

      if (dφti>= -t₀*h₀)
        #println("deuxième appel de zoom")
        (topt,good_grad,ht,iter)=methode(h,h₀,g₀,ti,tim1)
        return (topt,good_grad,ht,iter,0)
      end
      tim1=ti

      ti=(tim1+tmax)/2

      iter=iter+1
      verbose && @printf("%4d %7.2e %7.2e  %7.2e  %7.2e  %7.2e  %7.2e \n", iter, tim1,dφtim1,φtim1,ti,dφti,φti)
    end

    ht= φti + h₀ + τ₀*ti*g₀
    return (t,false,ht,iter,0)

end
