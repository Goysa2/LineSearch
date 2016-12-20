export zoom_ls
function zoom_ls(h :: AbstractLineFunction,
                 h₀ :: Float64,
                 g₀ :: Float64,
                 t₀ :: Float64,
                 t₁ :: Float64;
                 τ₀ :: Float64=1.0e-4,
                 τ₁ :: Float64=0.9999,
                 ϵ :: Float64=1e-5,
                 maxiter :: Int=50,
                 verbose :: Bool=false)

  #println("ZOOM_LS")

  φ(ti) = obj(h,ti) - h₀ - τ₀*ti*g₀  # fonction et
  dφ(ti) = grad(h,ti) - τ₀*g₀    # dérivée

  if φ(t₀)<φ(t₁)
    tlow=t₀
    thi=t₁
  else
    tlow=t₁
    thi=t₀
  end

  #println("thi=",thi," tlow=",tlow)

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
        #println("on sort de zoom parce que admissible")
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

    ti=(tlow+thi)/2
    φti=φ(ti)
    dφti=dφ(ti)
    #println("ti=",ti)
    #println("φti=",φti)
    #println("version calculé φti=",φ(ti))
    #println("dφti=",dφti)
    #println("version calculé dφti=",dφ(ti))
    iter+=1
    tired = iter > maxiter
    verbose && @printf(" %7.2e %7.2e  %7.2e  %7.2e  %7.2e %7.2e %7.2e %7.2e\n", iter,tlow,thi,ti,φlow,φhi,φti,dφti)
    verbose && println("  ")
  end

  topt=ti
  ht = φti + h₀ + τ₀*ti*g₀

  #println("on sort de zoom parce que fini")
  return (topt,false,ht,iter)
end
