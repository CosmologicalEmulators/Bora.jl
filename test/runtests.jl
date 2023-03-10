using Test
using SimpleChains
using Static

mlpd = SimpleChain(
  static(6),
  TurboDense(tanh, 64),
  TurboDense(tanh, 64),
  TurboDense(tanh, 64),
  TurboDense(tanh, 64),
  TurboDense(tanh, 64),
  TurboDense(identity, 40)
)

weights = SimpleChains.init_params(mlpd)
emu = Bora.SimpleChainsEmulator(Architecture = mlpd, Weights = weights)
r_test = Array(LinRange(0,200, 40))
bora_emu = Bora.ξℓEmulator(TrainedEmulator = emu, rgrid=r_test, InMinMax = rand(6,2),
                                OutMinMax = rand(6,2))

bora_complete_emu = Bora.CompleteEmulator(rgrid=r_test, ξℓMono=bora_emu, ξℓQuad=bora_emu,
                                          ξℓHexa=bora_emu)

@testset "Bora tests" begin
    cosmo = rand(6)
    cosmo_vec = hstack(cosmo, cosmo)
    bb = rand(9)
    bb_vec = hstack(bb, bb)
    output = Bora.get_ξℓs(cosmo, bb, bora_complete_emu)
    output_vec = Bora.get_ξℓs(cosmo_vec, bb_vec, bora_complete_emu)
    tests_zeros = Bora.get_broadband(r_test, zeros(9,100))
    @test any(tests_zeros .== 0)
    @test any(output .== output_vec[1,:,:])
end
