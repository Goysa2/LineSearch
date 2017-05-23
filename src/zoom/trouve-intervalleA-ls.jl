export trouve_intervalleA_ls
function trouve_intervalleA_ls(h :: AbstractLineFunction,
                               h₀ :: Float64,
                               g₀ :: Float64,
                               g :: Array{Float64,1};
                               direction :: String="Nwt",
                               τ₀ :: Float64=1.0e-4,
                               τ₁ :: Float64=0.9,
                               t₀ :: Float64=0.0,
                               tmax :: Float64=1000.0,
                               maxiter :: Int=50,
                               verbose :: Bool=false,
                               kwargs...)

    #Before starting the algorithm we vcheck to see if a step of 1.0 satisfies both the Wolfe and the Armijo condition
    ti=1.0
    ht = obj(h,ti)
    gt = grad!(h, ti, g)
    if Armijo(ti,ht,gt,h₀,g₀,τ₀) && Wolfe(gt,g₀,τ₁)
        return (ti,true,ht,0,0)
    end

    #We redefine our h function into the φ function.
    φ(t) = obj(h,t) - h₀ - τ₀*t*g₀  # fonction et
    dφ(t) = grad!(h,t,g) - τ₀*g₀    # dérivée

    tim1=t₀
    #ti=(tim1+tmax)/2
    ti=1.0
    iter=1

    #By the definition of φ we know it's value for t=0.0
    φtim1=0.0            #t₀=0.0 => φ(0)=0.0
    dφtim1=(1.0-τ₀)*g₀   #t₀=0.0 => dφ(0)=(1.0-τ₀)*g₀
    #We will compute the value of the funtion and it's derivative at the current point when/if necessary
    #NaN is there for the first verbose
    φti=NaN
    dφti=NaN

    #With the h function the strong wolfe condition is: |h'(t)|<=-τ₁h'(0)
    #With φ the strong Wolfe condition is (τ₁-τ₀)*h'(0)<= φ'(t)<=-(τ₁+τ₀)*h'(0)
    ɛa = (τ₁-τ₀)*g₀
    ɛb = -(τ₁+τ₀)*g₀

    verbose && @printf("iter tim1        dφtim1        φtim1         ti        dφti        φti\n")
    verbose && @printf("%4d %7.2e %7.2e  %7.2e  %7.2e  %7.2e  %7.2e \n", iter, tim1,dφtim1,φtim1,ti,dφti,φti)

    while (iter<maxiter) & (ti<=tmax)
      #Implementation of the 3.5 Linesearch ALgorithm as presented by Nocedal & Wright
      φti=φ(ti)
      if (φti>0.0) | ((φti>φtim1) & (iter>1))
        #println("on rentre dans zoom_generic_ls 1")
        #println("(tim1,ti,h₀,g₀)=",(tim1,ti,h₀,g₀))
        (topt,good_grad,ht,i)=zoom_generic_ls(h,h₀,g₀,tim1,ti,direction=direction,verbose=verbose)
        return (topt,good_grad,ht,iter,0)
      end

      dφti=dφ(ti)

      if ((dφti>=ɛa) & (dφti<=ɛb))
        topt=ti
        ht= φti + h₀ + τ₀*ti*g₀
        return (topt,false,ht,iter,0)
      end

      if (dφti>= -t₀*h₀)
        #println("on rentre dans zoom_generic_ls 2")
        (topt,good_grad,ht,iter)=zoom_generic_ls(h,h₀,g₀,ti,tim1,direction=direction,verbose=verbose)
        return (topt,good_grad,ht,iter,0)
      end

      #The current step t becomes the former step t
      tim1=ti
      φtim1=φti
      dφtim1=dφti

      #Interpolation of the new current step t
      ti*=2.0
      #ti=(tim1+tmax)/2

      iter=iter+1
      verbose && @printf("%4d %7.2e %7.2e  %7.2e  %7.2e  %7.2e  %7.2e \n", iter, tim1,dφtim1,φtim1,ti,dφti,φti)
    end

    ht= φti + h₀ + τ₀*ti*g₀
    return (t,false,ht,iter,0)

end
