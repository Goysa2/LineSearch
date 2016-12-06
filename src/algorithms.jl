export algorithms,algorithms_zoom

algorithms=[]
algorithms_zoom=[]
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
#push!(algorithms2,trouve_intervalle_ls)

push!(algorithms,Biss_ls)
push!(algorithms,Biss_Cub_ls)
push!(algorithms,Biss_Nwt_ls)
push!(algorithms,Biss_Sec_ls)
push!(algorithms,Biss_SecA_ls)

#zoom methods
push!(algorithms,zoom_Cub_ls)
push!(algorithms,zoom_nwt_ls)
push!(algorithms,zoom_sec_ls)
push!(algorithms,zoom_secA_ls)

#push!(algorithms3,trouve_intervalleA_ls)
