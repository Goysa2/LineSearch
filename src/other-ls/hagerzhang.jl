export hagerzhang

function hagerzhang(h :: LineModel, stop_ls :: LS_Stopping; kwargs...)
	return _hagerzhang2!(h, stop_ls, obj(h, 0.0), grad(h, 0.0), Float64[])
end
