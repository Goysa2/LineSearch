export algorithms, interfaced_algorithms
export Newton_linesearch

algorithms = []
interfaced_algorithms = []
Newton_linesearch = []

#ARC methods
push!(algorithms,ARC_Cub_ls)
push!(algorithms,ARC_Nwt_ls)
push!(algorithms,ARC_Sec_ls)
push!(algorithms,ARC_SecA_ls)

#TR methods
push!(algorithms,TR_Cub_ls)
push!(algorithms,TR_Nwt_ls)
push!(algorithms,TR_Sec_ls)
push!(algorithms,TR_SecA_ls)

#bissection methods
push!(algorithms,Biss_ls)
push!(algorithms,Biss_Cub_ls)
push!(algorithms,Biss_Nwt_ls)
push!(algorithms,Biss_Sec_ls)
push!(algorithms,Biss_SecA_ls)

#zoom methods
# push!(algorithms, trouve_intervalleA_ls)
push!(algorithms, zoom_Cub_ls)
push!(algorithms, zoom_ls)
push!(algorithms, zoom_nwt_ls)
push!(algorithms, zoom_secA_ls)
push!(algorithms, zoom_sec_ls)

#linesearch algorithms infaced from LineSearches
push!(interfaced_algorithms,_backtracking2!)
push!(interfaced_algorithms,_hagerzhang2!)
push!(interfaced_algorithms,_morethuente2!)
push!(interfaced_algorithms,_strongwolfe2!)

push!(algorithms,_backtracking2!)
push!(algorithms,_hagerzhang2!)
push!(algorithms,_morethuente2!)
push!(algorithms,_strongwolfe2!)

#linesearch using a Nwt interolation therefore requiring a C2LineFunction
push!(Newton_linesearch,TR_Nwt_ls)
push!(Newton_linesearch,ARC_Nwt_ls)
push!(Newton_linesearch, Biss_Nwt_ls)
push!(Newton_linesearch, zoom_nwt_ls)
push!(Newton_linesearch,trouve_intervalleA_ls)

#"basic" linesearch
push!(algorithms, Newarmijo_wolfe)
