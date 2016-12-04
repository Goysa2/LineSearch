export algorithms,algorithms2

algorithms=[]
algorithms2=[]
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
push!(algorithms2,trouve_intervalle_ls)

push!(algorithms,Biss_ls)
push!(algorithms,Biss_Cub_ls)
push!(algorithms,Biss_Nwt_ls)
push!(algorithms,Biss_Sec_ls)
push!(algorithms,Biss_SecA_ls)
