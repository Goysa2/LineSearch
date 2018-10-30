export ls_algorithms, interfaced_ls_algorithms
export Newton_linesearch

ls_algorithms = []
interfaced_ls_algorithms = []
Newton_linesearch = []

#ARC methods
push!(ls_algorithms,ARC_Cub_ls)
push!(ls_algorithms,ARC_Nwt_ls)
push!(ls_algorithms,ARC_Sec_ls)
push!(ls_algorithms,ARC_SecA_ls)

#TR methods
push!(ls_algorithms,TR_Cub_ls)
push!(ls_algorithms,TR_Nwt_ls)
push!(ls_algorithms,TR_Sec_ls)
push!(ls_algorithms,TR_SecA_ls)

#bissection methods
push!(ls_algorithms,Biss_ls)
push!(ls_algorithms,Biss_Cub_ls)
push!(ls_algorithms,Biss_Nwt_ls)
push!(ls_algorithms,Biss_Sec_ls)
push!(ls_algorithms,Biss_SecA_ls)

#zoom methods
# push!(ls_algorithms, find_intervalA_ls)
push!(ls_algorithms, zoom_Cub_ls)
push!(ls_algorithms, zoom_ls)
push!(ls_algorithms, zoom_Nwt_ls)
push!(ls_algorithms, zoom_SecA_ls)
push!(ls_algorithms, zoom_Sec_ls)

#linesearch ls_algorithms infaced from LineSearches
push!(interfaced_ls_algorithms,_backtracking2!)
push!(interfaced_ls_algorithms,_hagerzhang2!)
push!(interfaced_ls_algorithms,_morethuente2!)
push!(interfaced_ls_algorithms,_strongwolfe2!)

push!(ls_algorithms,_backtracking2!)
push!(ls_algorithms,_hagerzhang2!)
push!(ls_algorithms,_morethuente2!)
push!(ls_algorithms,_strongwolfe2!)

#"basic" linesearch
push!(ls_algorithms, Newarmijo_wolfe)
