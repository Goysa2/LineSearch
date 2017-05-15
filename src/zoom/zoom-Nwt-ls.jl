export zoom_nwt_ls
function zoom_nwt_ls(h :: AbstractLineFunction,
                 h₀ :: Float64,
                 g₀ :: Float64,
                 t₀ :: Float64,
                 t₁ :: Float64;
                 τ₀ :: Float64=1.0e-4,
                 τ₁ :: Float64=0.9999,
                 ϵ :: Float64=1e-5,
                 maxiter :: Int=50,
                 verbose :: Bool=false)


  φ(t) = obj(h,t) - h₀ - τ₀*t*g₀  # fonction et
  dφ(t) = grad(h,t) - τ₀*g₀    # dérivée
  ddφ(t) = hess(h,t)

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

  ti=t₁
  tp=t₀
  tqnp=t₀
  iter=0

  φti=φ(ti)
  dφti=dφ(ti)
  φtm1=dφ(tqnp)
  dφtm1=dφ(tqnp)

  ɛa = (τ₁-τ₀)*g₀
  ɛb = -(τ₁+τ₀)*g₀

  admissible=false
  tired= iter > maxiter

  verbose && @printf(" iter        tlow        thi         ti        φlow       φhi         φt         dφt\n")
  verbose && @printf(" %7.2e %7.2e  %7.2e  %7.2e  %7.2e %7.2e %7.2e %7.2e\n", iter,tlow,thi,ti,φlow,φhi,φti,dφti)
  while !(admissible | tired)
    #φti=φ(ti)
    if (φti>0) | (φti>=φlow)
      thi=ti
      φthi=φti
      dφhi=dφti
    else
      #dφti=dφ(ti)
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

    ddφti=ddφ(ti)
    dN=-dφti/ddφti

    if ((tp-ti)*dN>0) & (dN/(tp-ti)<γ)
      tplus = ti + dN
      φplus = φ(tplus)
      dφplus = dφ(tplus)
      verbose && println("N")
    else
      tplus = (ti+tp)/2
      φplus = φ(tplus)
      dφplus = dφ(tplus)
      verbose && println("B")
    end

    if ti>tp
      if dφplus<0
        tp=ti
        tqnp=ti
        ti=tplus
      else
        tqnp=ti
        ti=tplus
      end
    else
      if dφplus>0
        tp=ti
        tqnp=ti
        ti=tplus
      else
        tqnp=ti
        ti=tplus
      end
    end

    φtm1=φti
    dφtm1=dφti
    φti=φplus
    dφti=dφplus
    iter+=1
    tired = iter > maxiter
    verbose && @printf(" %7.2e %7.2e  %7.2e  %7.2e  %7.2e %7.2e %7.2e %7.2e\n", iter,tlow,thi,ti,φlow,φhi,φti,dφti)
  end

  topt=ti
  ht = φti + h₀ + τ₀*ti*g₀

  #println("on sort parce que fini")
  return (topt,false,ht,iter)
end
