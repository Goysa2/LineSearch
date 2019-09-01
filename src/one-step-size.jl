export one_step_size

"""
Returns 1.0 step size
"""
function one_step_size(h  :: LineModel,  stop_ls :: LS_Stopping;
                       f_meta = LS_Function_Meta(),
                       φ_dφ   :: Function = (x, y) -> phi_dphi(x, y),
                       verboseLS :: Bool = false, kwargs...)
    state = stop_ls.current_state
    update!(state, x = 1.0)

    return (state, true)
end # function
