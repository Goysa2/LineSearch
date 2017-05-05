export Nwt_computation_ls
function Nwt_computation_ls(t::Float64,
                            d::Float64,
                            φtestTR::Float64,
                            h::AbstractLineFunction,
                            dφ ::Function)
  t = t + d

  φt = φtestTR
  dφt = dφ(t)
  ddφt = hess(h,t)

  return (t,φt,dφt,ddφt)

end
