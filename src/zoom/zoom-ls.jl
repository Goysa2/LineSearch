function zoom_ls(h :: AbstractLineFunction,
                 h₀ :: Float64,
                 g₀ :: Float64,
                 t₀ :: Float64,
                 t₁ :: Float64;
                 τ₀ :: Float64=1.0e-4,
                 τ₁ :: Float64=0.9999,
                 ϵ :: Float64=1e-5,
                 maxiter :: Int=10,
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

  iter=0

  ti=(tlow+thi)/2
  φti=φ(ti)
  dφti=dφ(ti)

  ɛa = (τ₁-τ₀)*g₀
  ɛb = -(τ₁+τ₀)*g₀

  admissible=false
  tired=iter>maxiter

  verbose && @printf(" iter        tlow        thi         ti        φlow       φhi         φt         dφt\n")
  verbose && @printf(" %7.2e %7.2e  %7.2e  %7.2e  %7.2e %7.2e %7.2e %7.2e\n", iter,tlow,thi,ti,φlow,φhi,φti,dφti)
  while !(admissible | tired)
    φt=φ(ti)
    if (φt>0) | (φt>φlow)
      thi=ti
      φthi=φti
      dφhi=dφti
    else
      dφti=dφ(ti)
      if (abs(dφti)<ϵ)
        println("abs(dφti)<ϵ")
        topt=ti
        ht = φti + h₀ + τ₀*ti*g₀
        admissible=true
        return (topt,admissible,ht,iter)
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

    ti=(tlow+thi)/2
    admissible = (dφti>=ɛa) & (dφti<=ɛb)
    iter+=1
    tired = iter > maxiter
    verbose && @printf(" %7.2e %7.2e  %7.2e  %7.2e  %7.2e %7.2e %7.2e %7.2e\n", iter,tlow,thi,ti,φlow,φhi,φti,dφti)
  end

  topt=ti
  ht = φti + h₀ + τ₀*ti*g₀

  return (topt,admissible,ht,iter)
end
