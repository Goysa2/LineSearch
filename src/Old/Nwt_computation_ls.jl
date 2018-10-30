export Nwt_computation_ls
function Nwt_computation_ls(t::Float64,
                            d::Float64,
                            φtestTR::Float64,
                            dφtestTR::Float64,
                            h::LineModel)
  t = t + d

  φt = φtestTR
  dφt = dφtestTR
  ddφt = hess(h,t)

  return (t,φt,dφt,ddφt)

end
