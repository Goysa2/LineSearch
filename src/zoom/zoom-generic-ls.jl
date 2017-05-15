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

  print_with_color(:yellow,"avant de commencer zoom \n")
  println(" début zoom nbre éval total=",h.nlp.counters.neval_obj)

  verbose && println("on rentre dans zoom_generic_ls avec direction=",direction)

  φ(ti) = obj(h,ti) - h₀ - τ₀*ti*g₀  # fonction et
  dφ(ti) = grad(h,ti) - τ₀*g₀    # dérivée

  if direction=="Nwt"
    ddφ(t)=hess(h,t)
  end

  if φ(t₀)<φ(t₁)
    tlow=t₀
    thi=t₁
  else
    tlow=t₁
    thi=t₀
  end

  φlow=φ(tlow)
  dφlow=dφ(tlow)
  φhi=φ(thi)
  dφhi=dφ(thi)

  if direction=="Nwt" || direction=="Sec" || direction=="SecA" || direction=="Cub"
    ti=t₁
    tp=t₀
    tqnp=t₀
    # φti=φ(ti)
    # dφti=dφ(ti)
    # φtm1=dφ(tqnp)
    # dφtm1=dφ(tqnp)
  else
    ti=(tlow+thi)/2
    φti=φ(ti)
    dφti=dφ(ti)
  end

  iter=0

  ɛa = (τ₁-τ₀)*g₀
  ɛb = -(τ₁+τ₀)*g₀

  verbose && println("ɛa=",ɛa," ɛb=",ɛb)

  #admissible=false
  tired= iter > maxiter

  verbose && @printf(" iter        tlow        thi         ti        φlow       φhi         φt         dφt\n")
  verbose && @printf(" %7.2e %7.2e  %7.2e  %7.2e  %7.2e %7.2e %7.2e %7.2e\n", iter,tlow,thi,ti,φlow,φhi,φti,dφti)
  while !(tired)
    if iter <1
      φti=φ(ti)
    end
    #println("au calcul φti=",φti)
    if (φti>0) | (φti>=φlow)
      thi=ti
      φthi=φti
      if iter < 1
        dφhi=dφ(ti)
      else
        dφhi=dφti
      end
      #println("au calcul dφti=",dφhi)
    else
      if iter < 1
        dφti=dφ(ti)
      end
      #println("au calcul dφti=",dφti)
      if ((dφti>=ɛa) & (dφti<=ɛb))
        verbose && println("on arrête pcq bonne condition")
        verbose && print_with_color(:yellow,"fin de zoom")
        verbose && println("nbre éval total=",h.nlp.counters.neval_obj+h.nlp.counters.neval_grad+h.nlp.counters.neval_hprod)
        #break
        topt=ti
        ht = φti + h₀ + τ₀*ti*g₀
        println(" fin zoom nbre éval function total=",h.nlp.counters.neval_obj)
        println(" fin zoom nbre itérations=",iter)
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

    if direction=="Nwt"
      (ti,tp,tqnp,tplus,φtm1,dφtm1,φti,dφti)=zoom_qn_interpolation(φ,dφ,ddφ,tp,ti,φti,dφti,φtm1,dφtm1,tqnp,direction,γ,verbose=verbose)
    elseif direction=="Sec" || direction=="SecA" || direction=="Cub"
      if tlow<thi
        #println("if avant interpolation eval functions=",h.nlp.counters.neval_obj)
        (ti,tp,tqnp,tplus,φtm1,dφtm1,φti,dφti)=zoom_qn_interpolation(φ,dφ,dφ,tlow,thi,φhi,dφhi,φlow,dφlow,tqnp,direction,γ,verbose=verbose)
        #println("if après interpolation eval functions=",h.nlp.counters.neval_obj)
        #println("à la fin de zoom_qn_interpolation φti=",φti," dφti=",dφti)
      else
        #println("else avant interpolation eval functions=",h.nlp.counters.neval_obj)
        (ti,tp,tqnp,tplus,φtm1,dφtm1,φti,dφti)=zoom_qn_interpolation(φ,dφ,dφ,thi,tlow,φlow,dφlow,φhi,dφhi,tqnp,direction,γ,verbose=verbose)
        #println("else après interpolation eval functions=",h.nlp.counters.neval_obj)
        #println("à la fin de zoom_qn_interpolation φti=",φti," dφti=",dφti)
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

  verbose && print_with_color(:yellow,"fin de zoom")
  println(" fin zoom nbre éval funtionc total=",h.nlp.counters.neval_obj)

  return (topt,false,ht,iter)
end
