module Bora


using Base: @kwdef
using LoopVectorization
using SimpleChains

function maximin_input!(x, in_MinMax)
    for i in eachindex(x)
        x[i] -= in_MinMax[i,1]
        x[i] /= (in_MinMax[i,2]-in_MinMax[i,1])
    end
end

function inv_maximin_output!(x, out_MinMax)
    for i in eachindex(x)
        x[i] *= (out_MinMax[i,2]-out_MinMax[i,1])
        x[i] += out_MinMax[i,1]
    end
end

abstract type AbstractTrainedEmulators end

@kwdef mutable struct SimpleChainsEmulator <: AbstractTrainedEmulators
    Architecture
    Weights
end

abstract type AbstractPℓEmulator end

@kwdef mutable struct PℓEmulator <: AbstractPℓEmulator
    TrainedEmulator::AbstractTrainedEmulators
    rgrid::Array
    InMinMax::Matrix{Float64} = zeros(7,2)
    OutMinMax::Array{Float64} = zeros(40,2)
end

abstract type AbstractCompleteEmulator end

@kwdef mutable struct CompleteEmulator <: AbstractCompleteEmulator
    rgrid::Array
    PℓMono::AbstractPℓEmulator
    PℓQuadru::AbstractPℓEmulator
    PℓHexadeca::AbstractPℓEmulator
end

function ComputePℓs(input_params, Pℓs_emu::CompleteEmulator)
    output_l0 = ComputePℓ(input_params, Pℓs_emu.PℓMono)
    output_l2 = ComputePℓ(input_params, Pℓs_emu.PℓQuadru)
    output_l4 = ComputePℓ(input_params, Pℓs_emu.PℓHexadeca)
    return Array(hcat(output_l0, output_l2, output_l4)')
end


function ComputePℓ(input_params, Pℓ_emu::PℓEmulator)
    input = deepcopy(input_params)
    maximin_input!(input, Pℓ_emu.InMinMax)
    output = Array(run_emulator(input, Pℓ_emu.TrainedEmulator))
    inv_maximin_output!(output, Pℓ_emu.OutMinMax)
    return output#reshape(output, Int(length(output)/length(Pℓ_emu.rgrid)), :)
end

function run_emulator(input, trained_emulator::SimpleChainsEmulator)
    return trained_emulator.Architecture(input, trained_emulator.Weights)
end

function ComputeBroadBand(r, bbpar::Array{T}) where T
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

function ComputeBora(input_params, Pℓs_emu::CompleteEmulator)
    cosmo_params = input_params[1:7]
    bb_params = input_params[8:16]
    Pls = ComputePℓs(cosmo_params, Pℓs_emu)
    BB = ComputeBroadBand(Pℓs_emu.rgrid, bb_params)
    return Pls .+ BB
end

function ComputeBora(input_cosmo, input_broadband, Pℓs_emu::CompleteEmulator)
    Pls = ComputePℓs(input_cosmo, Pℓs_emu)
    BB = ComputeBroadBand(Pℓs_emu.rgrid, input_broadband)
    return Pls .+ BB
end

function ComputeBoraReduced(input_params, Pℓs_emu::CompleteEmulator)
    cosmo_params = input_params[1:6]
    bb_params = input_params[7:15]
    Pls = ComputePℓs(cosmo_params, Pℓs_emu)
    BB = ComputeBroadBand(Pℓs_emu.rgrid, bb_params)
    return Pls .+ BB
end

end # module
