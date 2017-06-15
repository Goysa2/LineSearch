export trouve_intervalle_ls
function trouve_intervalle_ls(h :: AbstractLineFunction2,
                              h₀ :: Float64,
                              g₀ :: Float64,
                              g :: Array{Float64,1};
                              inc0 :: Float64=1.0,
                              τ₀ :: Float64=1.0e-4,
                              τ₁ :: Float64=0.9999,
                              maxiter :: Int=100,
                              verboseLS :: Bool=false,
                              kwargs...)

  #println("verboseLS=",verboseLS)
  iter=1
  t₀=0.0
  inc=inc0
  #sd=-sign(g₀)
  #t₁=t₀+sd*inc
  #println("")
  φ(t) = obj(h,t) - h₀ - τ₀*t*g₀  # fonction et
  dφ(t) = grad!(h,t,g) - τ₀*g₀    # dérivée
  #dφ(t) = grad(h,t) - τ₀*g₀

  #println("dans trouve_intervalle_ls h₀=",h₀," g₀=",g₀)
  #println("version calculé h₀=",obj(h,0.0)," g₀=",grad(h,0.0))

  φt₀ = 0.0          # on sait que φ(0)=0
  dφt₀ = (1.0-τ₀)*g₀ # connu dφ(0)=(1.0-τ₀)*g₀
  #dφt₀=dφ(t₀)
  sd=-sign(dφt₀)
  t₁=t₀+sd*inc
  φt1=φ(t₁)
  dφt1=dφ(t₁)


  ɛa = (τ₁-τ₀)*g₀
  ɛb = -(τ₁+τ₀)*g₀

  verboseLS && @printf("iter t        φt        dφt         t1        φt1        dφt1\n")
  verboseLS && @printf("%4d %7.2e %7.2e  %7.2e  %7.2e  %7.2e  %7.2e \n", iter, t₀,φt₀,dφt₀,t₁,φt1,dφt1)

  while (dφt1*sd<0.0) & (φt1<φt₀) & (iter<maxiter)
    #println("on entre ici premier while")
    inc=inc*4
    t₀=t₁
    φt₀=φt1
    dφt₀=dφt1
    t₁=t₀+sd*inc
    φt1=φ(t₁)
    dφt1=dφ(t₁)
    verboseLS && @printf("%4d %7.2e %7.2e  %7.2e  %7.2e  %7.2e  %7.2e \n", iter, t₀,φt₀,dφt₀,t₁,φt1,dφt1)
    iter=iter+1

  end

  while (dφt1*sd<0.0) & (iter<maxiter)
    #println("on entre ici deuxième while")
    tₘ=(t₁+t₀)/2
    φₘ=φ(tₘ)
    dφₘ=dφ(tₘ)
    if φₘ*sd>0
      t₁=tₘ
      φt1=φₘ
      dφt1=dφₘ
    else
      if φₘ<φt₀
        t₀=tₘ
        φt₀=φₘ
        dφt₀=dφₘ
      else
        t₁=tₘ
        φt1=φₘ
        dφt1=dφₘ
      end
    end
      verboseLS && @printf("%4d %7.2e %7.2e  %7.2e  %7.2e  %7.2e  %7.2e \n", iter, t₀,φt₀,dφt₀,t₁,φt1,dφt1)
    iter=iter+1
  end

  ta=min(t₀,t₁)
  tb=max(t₀,t₁)


  return (ta,tb)

end
