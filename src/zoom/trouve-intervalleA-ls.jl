export trouve_intervalleA_ls
function trouve_intervalleA_ls(h :: AbstractLineFunction,
                               h₀ :: Float64,
                               g₀ :: Float64,
                               g :: Array{Float64,1};
                               direction :: String="Nwt",
                               τ₀ :: Float64=1.0e-4,
                               τ₁ :: Float64=0.9999,
                               t₀ :: Float64=0.0,
                               tmax :: Float64=1000.0,
                               maxiter :: Int=50,
                               verbose :: Bool=false,
                               kwargs...)

    print_with_color(:blue,"on rentre dans trouve_intervalleA_ls \n")
    println("en entrant dans trouve_intervalleA_ls aappel de fonctions=",h.nlp.counters.neval_obj)

    # if h₀!=obj(h,0.0)
    #   verbose && print_with_color(:red,"erreur d'argument h₀")
    # else
    #   verbose && print_with_color(:green,"bon h₀")
    # end

    # if g₀!=grad(h,0.0)
    #   verbose && print_with_color(:green,"bon g₀")
    # end

    t=1.0
    ht = obj(h,t)
    gt = grad!(h, t, g)
    if Armijo(t,ht,gt,h₀,g₀,τ₀) && Wolfe(gt,g₀,τ₁)
        return (t,true,ht,0,0)
    end

    verbose && print_with_color(:green,"on utilise pas un pas de t=1.0 \n")

    φ(t) = obj(h,t) - h₀ - τ₀*t*g₀  # fonction et
    dφ(t) = grad!(h,t,g) - τ₀*g₀    # dérivée

    tim1=t₀
    #ti=(tim1+tmax)/2
    ti=1.0
    println("ti=",ti)
    iter=0

    φtim1=0.0            #t₀=0.0 => φ(0)=0.0
    dφtim1=(1.0-τ₀)*g₀   #t₀=0.0 => dφ(0)=(1.0-τ₀)*g₀
    φti=NaN
    dφti=NaN

    ɛa = (τ₁-τ₀)*g₀
    ɛb = -(τ₁+τ₀)*g₀

    #println("dans zoom_ls ɛa=",ɛa," ɛb=",ɛb)

    φtim1=0.0 #t₀=0.0 => φ(0)=0.0
    dφtim1=(1.0-τ₀)*g₀   #t₀=0.0 => dφ(0)=(1.0-τ₀)*g₀

    println("avant le while de trouve_intervalleA_ls aappel de fonctions=",h.nlp.counters.neval_obj)

    verbose && @printf("iter tim1        dφtim1        φtim1         ti        dφti        φti\n")
    verbose && @printf("%4d %7.2e %7.2e  %7.2e  %7.2e  %7.2e  %7.2e \n", iter, tim1,dφtim1,φtim1,ti,dφti,φti)

    while (iter<maxiter) & (ti<=tmax)
      φti=φ(ti)
      cond2=false
      if g₀<0
        cond2=(φti>φtim1)
      end
      if (φti>0.0) | (cond2 & (iter>1))
        (topt,good_grad,ht,i)=zoom_generic_ls(h,h₀,g₀,tim1,ti,direction=direction,verbose=verbose)
        return (topt,good_grad,ht,iter,0)
      end

      # dφtim1=dφti
      dφti=dφ(ti)
      # tim1=ti
      # ti=(tim1+tmax)/2


      if ((dφti>=ɛa) & (dφti<=ɛb))
        topt=ti
        ht= φti + h₀ + τ₀*ti*g₀
        return (topt,false,ht,iter,0)
      end

      if (dφti>= -t₀*h₀)
        (topt,good_grad,ht,iter)=zoom_generic_ls(h,h₀,g₀,ti,tim1,direction=direction,verbose=verbose)
        return (topt,good_grad,ht,iter,0)
      end
      tim1=ti
      φtim1=φti

      #ti=(tim1+tmax)/2
      tim1=ti
      φtim1=φti
      dφtim1=dφti
      ti*=2.0
      #φtim1=φti

      iter=iter+1
      verbose && @printf("%4d %7.2e %7.2e  %7.2e  %7.2e  %7.2e  %7.2e \n", iter, tim1,dφtim1,φtim1,ti,dφti,φti)
    end

    println("à la sortie de trouve_intervalleA_ls aappel de fonctions=",h.nlp.counters.neval_obj)

    ht= φti + h₀ + τ₀*ti*g₀
    return (t,false,ht,iter,0)

end
