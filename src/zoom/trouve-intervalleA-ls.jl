export trouve_intervalleA_ls
function trouve_intervalleA_ls(h :: AbstractLineFunction,
                               h₀ :: Float64,
                               g₀ :: Float64,
                               t₀ :: Float64,
                               tmax :: Float64,
                               methode :: Function,
                               g :: Array{Float64,1};
                               τ₀ :: Float64=1.0e-4,
                               τ₁ :: Float64=0.9999,
                               ϵ :: Float64=1e-5,
                               maxiter :: Int=30,
                               verbose :: Bool=false)

    t=1.0
    ht = obj(h,t)
    gt = grad!(h, t, g)
    println("on a t=",t," ht=",ht," gt=",gt)
    if Armijo(t,ht,gt,h₀,g₀,τ₀) && Wolfe(gt,g₀,τ₁)
        return (t,true,ht,0)
    end
    println("Armijo && Wolfe failed")

    φ(t) = obj(h,t) - h₀ - τ₀*t*g₀  # fonction et
    dφ(t) = grad!(h,t,g) - τ₀*g₀    # dérivée

    println("on a φ(t) & dφ(t)")

    tim1=t₀
    ti=(tim1+tmax)/2

    i=0.0

    φtim1=0.0            #t₀=0.0 => φ(0)=0.0
    dφtim1=(1.0-τ₀)*g₀   #t₀=0.0 => dφ(0)=(1.0-τ₀)*g₀
    φti=φ(ti)
    dφti=dφ(ti)


    φtim1=0.0 #t₀=0.0 => φ(0)=0.0
    dφtim1=(1.0-τ₀)*g₀   #t₀=0.0 => dφ(0)=(1.0-τ₀)*g₀

    verbose && @printf("iter tim1        dφtim1        φtim1         ti        dφti        φti\n")
    verbose && @printf("%4d %7.2e %7.2e  %7.2e  %7.2e  %7.2e  %7.2e \n", i, tim1,dφtim1,φtim1,ti,dφti,φti)

    while i<maxiter
      println("on est le while de trouve_intervalleA_ls")
      φtim1=φti
      φti=φ(ti)
      if (φti>0.0) | ((φti>φti) & (i>1))
        println("premier appel de zoom")
        (topt,admissible,ht,i)=methode(h,h₀,g₀,tim1,ti,verbose=true)
        return (topt,admissible,ht,i)
      end

      dφtim1=dφti
      dφti=dφ(ti)

      if (abs(dφti)<=-τ₁*g₀)
        iter=i
        topt=ti
        ht= φti + h₀ + τ₀*ti*g₀
        admissible=true #???
        return (topt,admissible,ht,iter)
      end

      if (dφti>= -t₀*h₀)
        println("deuxième appelle de zoom")
        (topt,admissible,ht,i)=methode(h,h₀,g₀,ti,tim1,verbose=true)
        return (topt,admissible,ht,i)
      end
      tim1=ti
      ti=(tim1+tmax)/2

      i=i+1
      verbose && @printf("%4d %7.2e %7.2e  %7.2e  %7.2e  %7.2e  %7.2e \n", i, tim1,dφtim1,φtim1,ti,dφti,φti)
    end
    iter=i
    ht= φti + h₀ + τ₀*ti*g₀
    return (t,false,ht,iter)

end
