function zoom_nwt_ls(h :: AbstractLineFunction,
                 h₀ :: Float64,
                 g₀ :: Float64,
                 t₀ :: Float64,
                 t₁ :: Float64;
                 τ₀ :: Float64=1.0e-4,
                 τ₁ :: Float64=0.9999,
                 ϵ :: Float64=1e-5,
                 maxiter :: Int=30,
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

  γ=0.8
  t=t₁
  tp=t₀
  tqnp=t₀
  iter=0

  φt=φ(t)
  dφt=dφ(t)
  φtm1=dφ(tqnp)
  dφtm1=dφ(tqnp)

  ɛa = (τ₁-τ₀)*g₀
  ɛb = -(τ₁+τ₀)*g₀

  admissible=false
  tired=iter>maxiter

  verbose && @printf(" iter        tlow        thi         t        φlow       φhi         φt         dφt\n")
  verbose && @printf(" %7.2e %7.2e  %7.2e  %7.2e  %7.2e %7.2e %7.2e %7.2e\n", iter,tlow,thi,t,φlow,φhi,φt,dφt)
  while !(admissible | tired)
    #φt=φ(ti)
    if (φt>0) | (φt>=φlow)
      thi=t
      φthi=φt
      dφhi=dφt
    else
      #dφti=dφ(ti)
      if (abs(dφt)<-τ₁*g₀)
        println("abs(dφti)<ϵ")
        topt=t
        ht = φt + h₀ + τ₀*t*g₀
        admissible=true
        return (topt,admissible,ht,iter)
      end

      if (dφt*(thi-tlow)>=-τ₀*g₀*(thi-tlow))
        thi=tlow
        φhi=φlow
        dφhi=dφlow
      end

      tlow=t
      φlow=φt
      dφlow=dφt
    end

    ddφt=ddφ(t)
    dN=-dφt/ddφt

    if ((tp-t)*dN>0) & (dN/(tp-t)<γ)
      tplus = t + dN
      φplus = φ(tplus)
      dφplus = dφ(tplus)
      verbose && println("N")
    else
      tplus = (t+tp)/2
      φplus = φ(tplus)
      dφplus = dφ(tplus)
      verbose && println("B")
    end

    if t>tp
      if dφplus<0
        tp=t
        tqnp=t
        t=tplus
      else
        tqnp=t
        t=tplus
      end
    else
      if dφplus>0
        tp=t
        tqnp=t
        t=tplus
      else
        tqnp=t
        t=tplus
      end
    end


    φtm1=φt
    dφtm1=dφt
    φt=φplus
    dφt=dφplus


    admissible = (dφt>=ɛa) & (dφt<=ɛb)
    iter+=1
    tired = iter > maxiter
    verbose && @printf(" %7.2e %7.2e  %7.2e  %7.2e  %7.2e %7.2e %7.2e %7.2e\n", iter,tlow,thi,t,φlow,φhi,φt,dφt)
  end

  topt=t
  ht = φt + h₀ + τ₀*t*g₀

  return (topt,admissible,ht,iter)
end
