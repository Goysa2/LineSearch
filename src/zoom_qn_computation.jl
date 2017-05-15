export zoom_qn_interpolation
function zoom_qn_interpolation(φ::Function,
                            dφ::Function,
                            ddφ::Function,
                            tp :: Float64,
                            ti :: Float64,
                            φti :: Float64,
                            dφti :: Float64,
                            methode :: String,
                            γ :: Float64;
                            verbose :: Bool=false,
                            kwargs...)

  if methode=="Nwt"
    ddφti=ddφ(ti)
    dN=-dφti/ddφti
  elseif methode=="Sec"
    s=t-tqnp
    y=dφt-dφtm1
    dN=-dφt*s/y
  elseif methode=="SecA"
    s=t-tqnp
    y=dφt-dφtm1
    Γ=3*(dφt+dφtm1)*s-6*(φt-φtm1)
    if y*s+Γ<eps(Float64)*(s^2)
      yt=y
    else
      yt=y+Γ/s
    end
    dN=-dφt*s/yt
  elseif methode=="Cub"
    s=t-tqnp
    y=dφt-dφtm1
    α=-s
    z=dφt+dφtm1+3*(φt-φtm1)/α
    discr=z^2-dφt*φtm1
    denom=dφt+dφtm1+2*z
    if (discr>0) & (abs(denom)>eps(Float64))
      #si on peut on utilise l'interpolation cubique
      w=sqrt(discr)
      dN=-s*(dφt+z+sign(α)*w)/(denom)
    else #on se rabat sur une étape de sécante
      dN=-dφt*s/y
    end
  end


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

  return (ti,tp,tqnp,tplus,φtm1,dφtm1,φti,dφti)
end
