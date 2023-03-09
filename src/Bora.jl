module Bora


using Base: @kwdef
using AbstractEmulator

abstract type AbstractξℓEmulator end

@kwdef mutable struct ξℓEmulator <: AbstractξℓEmulator
    TrainedEmulator::AbstractTrainedEmulators
    rgrid::Array
    InMinMax::Matrix{Float64} = zeros(7,2)
    OutMinMax::Array{Float64} = zeros(40,2)
end

abstract type AbstractCompleteEmulator end

@kwdef mutable struct CompleteEmulator <: AbstractCompleteEmulator
    rgrid::Array
    PℓMono::AbstractξℓEmulator
    PℓQuadru::AbstractξℓEmulator
    PℓHexadeca::AbstractξℓEmulator
end

function get_ξℓs(input_params, Pℓs_emu::CompleteEmulator)
    output_l0 = get_ξℓ(input_params, Pℓs_emu.PℓMono)
    output_l2 = get_ξℓ(input_params, Pℓs_emu.PℓQuadru)
    output_l4 = get_ξℓ(input_params, Pℓs_emu.PℓHexadeca)
    return Array(hcat(output_l0, output_l2, output_l4)')
end


function get_ξℓ(input_params, Pℓ_emu::ξℓEmulator)
    input = deepcopy(input_params)
    maximin_input!(input, Pℓ_emu.InMinMax)
    output = Array(run_emulator(input, Pℓ_emu.TrainedEmulator))
    inv_maximin_output!(output, Pℓ_emu.OutMinMax)
    return output
end

function get_broadband(r, bbpar::Vector{T}) where T
    ℓs = [0,2,4]
    BB = zeros(T, length(ℓs),length(r))
    bbpar_reshaped = reshape(bbpar, 3,3)'
    norm=0.0015#norm rappresenting the value of xi at r=rref
    rref=80.
    for l in 1:3
        for i in 1:3
            BB[l,:] .+= bbpar_reshaped[l,i]*r.^(-i+1)*norm*rref^(i-1)
        end
    end
    return BB
end

function get_broadband(r, bbpar::Matrix{T}) where T
    ℓs = [0,2,4]
    dim_pars, dim_vect = size(bbpar)
    BB = zeros(T, length(ℓs),length(r), dim_vect)
    bbpar_reshaped = zeros(3,3,dim_vect)
    for i in 1:dim_vect
        bbpar_reshaped[:,:,i] = reshape(bbpar[:,i], 3,3)'
    end
    norm=0.0015#the value of xi at r=rref
    rref=80.
    for l in 1:3
        for i in 1:3
            for v in 1:dim_vect
                BB[l,:,v] .+= bbpar_reshaped[l,i,v]*r.^(-i+1)*norm*rref^(i-1)
            end
        end
    end
    return BB
end

function get_ξℓs(cosmo_params, bb_params, Pℓs_emu::CompleteEmulator)
    Pls = get_ξℓs(cosmo_params, Pℓs_emu)
    BB = get_broadband(Pℓs_emu.rgrid, bb_params)
    return Pls .+ BB
end

end # module
