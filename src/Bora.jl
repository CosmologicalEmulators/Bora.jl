module Bora


using Base: @kwdef
using AbstractCosmologicalEmulators
import AbstractCosmologicalEmulators.get_emulator_description
import AbstractCosmologicalEmulators.init_emulator
using NPZ
using JSON
import JSON.parsefile

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
    ξℓMono::AbstractξℓEmulator
    ξℓQuad::AbstractξℓEmulator
    ξℓHexa::AbstractξℓEmulator
end

function load_multipole_emulator(path, ℓ; s_file="s.npy", weights_file="weights.npy", inminmax_file="inminmax.npy",
    outminmax_file="outminmax.npy", nn_setup_file="nn_setup.json")
    
    NN_dict = parsefile(joinpath(path, nn_setup_file))
    
    weights = npzread(joinpath(path, string(ℓ), weights_file))
    s_test = npzread(joinpath(path, s_file))
    
    trained_emu = init_emulator(NN_dict, weights, SimpleChainsEmulator)
    ξℓ_emu = Bora.ξℓEmulator(TrainedEmulator = trained_emu, rgrid=s_test,
                                InMinMax = npzread(joinpath(path, string(ℓ), inminmax_file)),
                                OutMinMax = npzread(joinpath(path, string(ℓ), outminmax_file)))
    return ξℓ_emu
end

function load_complete_emulator(path; s_file="s.npy", weights_file="weights.npy", inminmax_file="inminmax.npy",
    outminmax_file="outminmax.npy", nn_setup_file="nn_setup.json")
    @info "🔄 Loading BORA emulator"
    ξℓ0_emu = load_multipole_emulator(path, 0, s_file=s_file, weights_file=weights_file, inminmax_file=inminmax_file,
        outminmax_file=outminmax_file, nn_setup_file=nn_setup_file)
    ξℓ2_emu = load_multipole_emulator(path, 2, s_file=s_file, weights_file=weights_file, inminmax_file=inminmax_file,
        outminmax_file=outminmax_file, nn_setup_file=nn_setup_file)
    ξℓ4_emu = load_multipole_emulator(path, 4, s_file=s_file, weights_file=weights_file, inminmax_file=inminmax_file,
        outminmax_file=outminmax_file, nn_setup_file=nn_setup_file)
    complete_ξℓ = Bora.CompleteEmulator(rgrid=ξℓ0_emu.rgrid, ξℓMono=ξℓ0_emu, ξℓQuad=ξℓ2_emu, ξℓHexa=ξℓ4_emu);
    @info "✅ Loading completed"
    return complete_ξℓ
end

function get_ξℓs(input_params::Vector, ξℓs_emu::CompleteEmulator)
    output_l0 = get_ξℓ(input_params, ξℓs_emu.ξℓMono)
    output_l2 = get_ξℓ(input_params, ξℓs_emu.ξℓQuad)
    output_l4 = get_ξℓ(input_params, ξℓs_emu.ξℓHexa)
    return Array(hcat(output_l0, output_l2, output_l4)')
end


function get_ξℓs(input_params::Matrix, ξℓs_emu::CompleteEmulator)
    dim_f, dim_v = size(input_params)
    len_r = length(ξℓs_emu.rgrid)
    output_l0 = get_ξℓ(input_params, ξℓs_emu.ξℓMono)
    output_l2 = get_ξℓ(input_params, ξℓs_emu.ξℓQuad)
    output_l4 = get_ξℓ(input_params, ξℓs_emu.ξℓHexa)
    result = zeros(3, len_r, dim_v)
    for i in 1:dim_v
        result[:,:,i] = Array(hcat(output_l0[:,i], output_l2[:,i], output_l4[:,i])')
    end
    return result
end

function get_ξℓ(input_params, ξℓ_emu::ξℓEmulator)
    input = maximin(input_params, ξℓ_emu.InMinMax)
    output = Array(run_emulator(input, ξℓ_emu.TrainedEmulator))
    output_params = inv_maximin(output, ξℓ_emu.OutMinMax)
    return output_params
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

function get_ξℓs(cosmo_params, bb_params, ξℓs_emu::CompleteEmulator)
    Pls = get_ξℓs(cosmo_params, ξℓs_emu)
    BB = get_broadband(ξℓs_emu.rgrid, bb_params)
    return Pls .+ BB
end

function get_emulator_description(Clemu::AbstractξℓEmulator)
    get_emulator_description(Clemu.TrainedEmulator)
end

end # module
