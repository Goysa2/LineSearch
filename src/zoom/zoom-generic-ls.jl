export zoom_generic_ls
function zoom_generic_ls(h :: AbstractLineFunction,
                 h₀ :: Float64,
                 g₀ :: Float64,
                 t₀ :: Float64,
                 t₁ :: Float64;
                 τ₀ :: Float64=1.0e-4,
                 τ₁ :: Float64=0.9999,
                 ϵ :: Float64=1e-5,
                 maxiter :: Int=50,
                 verbose :: Bool=false,
                 direction :: String="Nwt",
                 γ :: Float64 = 0.8,
                 kwargs...)

  #Definition of the φ function as defined in the trouve_intervalleA_ls algorithm
  φ(ti) = obj(h,ti) - h₀ - τ₀*ti*g₀  # fonction et
  dφ(ti) = grad(h,ti) - τ₀*g₀    # dérivée

  #If we are using the Newton Interpolation, the second derivative is needed
  if direction=="Nwt"
    ddφ(t)=hess(h,t)
  end

  #We estabilsh which of the 2 points is tlow and thi
  if φ(t₀)<φ(t₁)
    tlow=t₀
    thi=t₁
  else
    tlow=t₁
    thi=t₀
  end

  #We evaluate the values of tlow and thi
  φlow=φ(tlow)
  dφlow=dφ(tlow)
  φhi=φ(thi)
  dφhi=dφ(thi)

  #Depending and the Interpolation, different information are needed
  #Some are probably superfluous, but doesn't affect the number of iterations or function evalutions
  #The idea is same for all Interpolations: Keep track of the current point ti, the previous point tp and the previous quasi-newton point tqnp
  if direction=="Nwt" || direction=="Sec" || direction=="SecA" || direction=="Cub"
    ti=t₁
    tp=t₀
    tqnp=t₀
  else
    ti=(tlow+thi)/2
    φti=φ(ti)
    dφti=dφ(ti)
  end

  iter=0

  #For the same reasons as before, the strong wolfe conditions for φ is (τ₁-τ₀)*h'(0)<= φ'(t)<=-(τ₁+τ₀)*h'(0)
  ɛa = (τ₁-τ₀)*g₀
  ɛb = -(τ₁+τ₀)*g₀

  verbose && println("ɛa=",ɛa," ɛb=",ɛb)

  tired= iter > maxiter

  verbose && @printf(" iter        tlow        thi         ti        φlow       φhi         φt         dφt\n")
  verbose && @printf(" %7.2e %7.2e  %7.2e  %7.2e  %7.2e %7.2e %7.2e %7.2e\n", iter,tlow,thi,ti,φlow,φhi,φti,dφti)
  while !(tired)
    #zoom (3.6) algorithm as presented by Nocedal & Wright

    #Since the Interpolation computes φ(ti) and dφ(ti) we don't need to compute them everytime
    #The algorithm as presented doesn't compute the 2 values (φ(ti),dφ(ti)) everytime
    #But both them are needed for the Interpolation (espcially Cubic and Secant)
    if iter <1
      φti=φ(ti)
    end

    if (φti>0) | (φti>=φlow)
      thi=ti
      φthi=φti
      if iter < 1
        dφhi=dφ(ti)
      else
        dφhi=dφti
      end
    else
      if iter < 1
        dφti=dφ(ti)
      end

      if ((dφti>=ɛa) & (dφti<=ɛb))
        topt=ti
        ht = φti + h₀ + τ₀*ti*g₀
        return (topt,false,ht,iter)
      end

      if (dφti*(thi-tlow)>=-τ₀*g₀*(thi-tlow))
        thi=tlow
        φhi=φlow
        dφhi=dφlow
      end

      tlow=ti
      φlow=φti
      dφlow=dφti
    end

    #Depending on the previous if/else statement ti always becomes tlow or thi.
    #So for or interpolation we make sure that the lowest of the 2 points and it's associated values are placed first
    #Some information are probably superfluous for some Interpolations, but it doesn't affect the performance
    if direction=="Nwt"
      if tlow<thi
        (ti,tp,tqnp,tplus,φtm1,dφtm1,φti,dφti)=zoom_qn_interpolation(φ,dφ,ddφ,tlow,thi,φhi,dφhi,φlow,dφlow,tqnp,direction,γ,verbose=verbose)
      else
        (ti,tp,tqnp,tplus,φtm1,dφtm1,φti,dφti)=zoom_qn_interpolation(φ,dφ,ddφ,thi,tlow,φlow,dφlow,φhi,dφhi,tqnp,direction,γ,verbose=verbose)
      end
    elseif direction=="Sec" || direction=="SecA" || direction=="Cub"
      if tlow<thi
        (ti,tp,tqnp,tplus,φtm1,dφtm1,φti,dφti)=zoom_qn_interpolation(φ,dφ,dφ,tlow,thi,φhi,dφhi,φlow,dφlow,tqnp,direction,γ,verbose=verbose)
      else
        (ti,tp,tqnp,tplus,φtm1,dφtm1,φti,dφti)=zoom_qn_interpolation(φ,dφ,dφ,thi,tlow,φlow,dφlow,φhi,dφhi,tqnp,direction,γ,verbose=verbose)
      end
    else
      ti=(tlow+thi)/2
      φti=φ(ti)
      dφti=dφ(ti)
    end

    iter+=1
    tired = iter >= maxiter
    verbose && @printf(" %7.2e %7.2e  %7.2e  %7.2e  %7.2e %7.2e %7.2e %7.2e\n", iter,tlow,thi,ti,φlow,φhi,φti,dφti)
    verbose && println("  ")
  end

  topt=ti
  ht = φti + h₀ + τ₀*ti*g₀

  return (topt,false,ht,iter)
end
