export zoom_Cub_ls
function zoom_Cub_ls(h :: AbstractLineFunction,
                 h₀ :: Float64,
                 g₀ :: Float64,
                 t₀ :: Float64,
                 t₁ :: Float64;
                 τ₀ :: Float64=1.0e-4,
                 τ₁ :: Float64=0.9999,
                 ϵ :: Float64=1e-5,
                 max_eval :: Int=100,
                 verbose :: Bool=false)

  φ(t) = obj(h,t) - h₀ - τ₀*t*g₀  # fonction et
  dφ(t) = grad(h,t) - τ₀*g₀    # dérivée

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
  nftot=h.nlp.counters.neval_obj+h.nlp.counters.neval_grad+h.nlp.counters.neval_hprod
  tired=nftot > max_eval

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
      if ((dφt>=ɛa) & (dφt<=ɛb))
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

    s=t-tqnp
    y=dφt-dφtm1

    dS=-dφt*s/y
    α=-s
    z=dφt+dφtm1+3*(φt-φtm1)/α
    discr=z^2-dφt*φtm1
    denom=dφt+dφtm1+2*z

    if (discr>0) & (abs(denom)>eps(Float64))
      #si on peut on utilise l'interpolation cubique
      w=sqrt(discr)
      dC=-s*(dφt+z+sign(α)*w)/(denom)
    else #on se rabat sur une étape de sécante
      dC=-dφt*s/y
    end

    if ((tp-t)*dC>0) & (dC/(tp-t)<γ)
      tplus = t + dC
      φplus = φ(tplus)
      dφplus = dφ(tplus)
      verbose && println("C")
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

    iter+=1
    nftot=h.nlp.counters.neval_obj+h.nlp.counters.neval_grad+h.nlp.counters.neval_hprod
    tired = nftot > max_eval
    verbose && @printf(" %7.2e %7.2e  %7.2e  %7.2e  %7.2e %7.2e %7.2e %7.2e\n", iter,tlow,thi,t,φlow,φhi,φt,dφt)
  end

  topt=t
  ht = φt + h₀ + τ₀*t*g₀

  return (topt,admissible,ht,iter)
end
