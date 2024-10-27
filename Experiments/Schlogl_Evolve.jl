using ITensors
using LinearAlgebra
#using StaticArrays
#using BenchmarkTools
#using Plots
#using Distributions
using JLD2
using HDF5
#using LaTeXStrings
#using Profile
using KrylovKit: exponentiate
#using DelimitedFiles
include("Tensor_Operations_NoQn.jl")
include("MPO_Mechanisms.jl")

# Evolves The Schlogl chemical reaction network with diffusion on a 1D lattice.
# Initially starting from a uniform distribution the program will evolve the system 
# To the stationary distribution. Code was used to find stationary distribution in 
# S.B. Nicholson T.R. Gingrich PRX 2023
#----------Schlogl_Parameters---------------------------------------
    L = 2                   # Number of sites  #2,3,4
    a = 1.06; b= .68;       # concentrations of chemostated species A and B
    nA =  6.02214076e23;    #Avogadros constant

    #k = [0.18,2.1e-4,2200,37.6]
    #k = [0.19,2.1001e-4,2200,37.6]

    vc = 1    # The parameter set counter
    # Parameters to run -----------------
        M=85                                            # Maximum occupation number
        Vol = 1.2e-25                                   # Volume of each voxel
        nstar = [15; 25]                                # The edges of basin B and A
        k = [0.1824459,.000210046,2199.855,37.88136]    # Kinetic rate constants used
        Dx = 2e-14                                      # Diffusion coefficient
        dval=2
        vsim = 1;                                       # parameter used to specify parameter lists
        BD = 20                                         # The bond dimension used for tdvp
    # ----------------------------------------------
      MMDim = [BD,50] # For L < 10                      # Minimum/maximum bond dimensions used if varying BD
#    MMDim = [24,50] # For L >= 10

    A = a*nA*Vol                                        # Number of chemostated species A
    B = b*nA*Vol                                        # Number of chemostated species B

    # Diffuison terms
    h = (Vol*1000)^(1/3)                                # voxel length converted from volume
#---------------------------------------------------------
    c = [k[1]*A/(nA*Vol)^2,k[2]/(nA*Vol)^2,k[3]*B,k[4]]; # Stochastic coefficients assuming kinetic coefficients are one
# ---------------------------------------------------------------------------
    n = M+1                                             # Number of states per physical index
    cX = Dx/h^2                                         # stochastic diffusion coefficient
    dt = .0001                                          # time step
    T = 1                                               # Total evolution time
    TLen = Int(floor(T/dt))                             # Number of time steps
    @show TLen

    ti= dt:dt:dt*TLen                                   # specify time increments
    Cutoff = 1e-13^2                                    # Cutoff error for growing the bond dimension of psi

# Below each function specifies a Doi-Peliti operator which can/is used to generate the effective Hamiltonians
ITensors.space(::SiteType"S_Class") = M+1
function ITensors.op!(Op::ITensor,
                      ::OpName"ac",
                      ::SiteType"S_Class",
                      s::Index)
                      for a=1:M
                          Op[s'=>a,s=> a+1] = a
                      end
end

function ITensors.op!(Op::ITensor,
                      ::OpName"ac_dag",
                      ::SiteType"S_Class",
                      s::Index)
                      for a=1:M
                          Op[s'=>a+1,s=>a] =1
                      end
end

function ITensors.op!(Op::ITensor,
                      ::OpName"IX",
                      ::SiteType"S_Class",
                      s::Index)
                      for a=1:M
                          Op[s'=>a,s=>a] =1
                      end
end

# Make a normalization operator for B+X -> Y+D
function ITensors.op!(Op::ITensor,
                      ::OpName"Fix_Norm",
                      ::SiteType"S_Class",
                      s::Index)
                      Op[s'=>M+1,s=>M+1] = 1
end
# Make the number operator
function ITensors.op!(Op::ITensor,
                      ::OpName"NumOp",
                      ::SiteType"S_Class",
                      s::Index)
                      for a=1:M+1
                          Op[s'=>a,s=>a] = a-1
                      end
end
function ITensors.op!(Op::ITensor,
                      ::OpName"Id",
                      ::SiteType"S_Class",
                      s::Index)
                      for a=1:M+1
                          Op[s'=>a,s=>a] = 1
                      end
end

sites = siteinds("S_Class",L::Int64)                    # Defining the basis for the space you are in
#Save out Data -----------------------------------------------------------------

MinTen = MMDim[1]
# Create the file names
psiName = "psiStation"*"$MinTen"*"BD_"*"$M"*"Mol_"*"$L"*"sites_Dx"*"$dval"*".jlds"
FileName = "TNStruct"*"$MinTen"*"BD_"*"$M"*"Mol_"*"$L"*"sites_Dx"*"$dval"*".h5"

# Local file path --------------------------------------------------------------
  FilePath = "C:\\Users\\snich\\Documents\\MATLAB\\Tensor_Networks\\Schlogl\\Figures\\Set_S2\\"
# Zinc file path ---------------------------------------------------------------
  # FilePath = "/home/sbn6912/RxnDiffusion/Schlogl/Data/StatDists/"
# Quest file path ---------------------------------------------------------------
#   FilePath = "/projects/p31555/RxnDiffusion/Schlogl/Data/StatDists/"
#-------------------------------------------------------------------------------

# --- Construct initial distrbution from well mixed stationary distribution

# Local Path
#    peqPath = "C:/Users/snich/Documents/Julia_Programs/Rxn_Diffusion/Schlogl/WellMixed_StatDists/"
    #Pr_Dist= readdlm(peqPath*"Dist_"*"$M"*"Mol.txt", ',', Float64)
# Zinc/Quest Path
    #Pr_Dist= readdlm("Dist_"*"$M"*"Mol.txt", ',', Float64)

    # Generate the random uniform distribution for the first site
    Pr_Dist = rand(n)                               # Generate the random uniform distribution for the first site
    Pr_Dist = Pr_Dist./sum(Pr_Dist)
        psi = MPS(L)                                # Create an empty MPS of length L
            psi[1] = ITensor(Pr_Dist,sites[1])      # Fill the first entry of the MPS
        for k =1:L-1                                # Generate and svd each pair of tensors to create an independent MPS
            Pr_Dist = rand(n)
            Pr_Dist = Pr_Dist./sum(Pr_Dist)
            ip = inds(psi[k])
            Ten = psi[k]*ITensor(Pr_Dist,sites[k+1])
            u,s,v = svd(Ten,ip,cutoff=1e-12,mindim=MMDim[1])
            psi[k] = u;
            psi[k+1] = s*v;
        end
orthogonalize!(psi::MPS,1)                  # Brings psi into mixed cannonical form, centered at site 1

#---------------------------------------------------------------------------------
# Thie function generates an MPO from a pre specified list of mechanisms
#H = MPO_Mechanism("sites","length of MPO", "Molecules cap", "stochastic coefficients", "Name of Mechanism", Diffusion coefficient)
H = MPO_Mechanisms(sites::Vector{Index{Int64}},L::Int64,M::Int64,c::Vector{Float64},"Schlogl",cX)
println("H has been constructed")

Prob = Contract_MPS(psi::MPS,L::Int64,n::Int64)                 # Calculates <1|ψ>, which is the total probability in ψ
ExX = Calculate_Expected_Molecules(psi::MPS,L::Int64,n::Int64); # Calculates <1|̂n|ψ>

N = Int(L-1)
psi0 = deepcopy(psi::MPS)

# Create vectors to store information about system
MProb = zeros(TLen,1)                       # Stores total probability in MPS
MProb[1] = Prob
Exp_XMol = zeros(L,TLen)                    # Stores average number of molecules at each site
Exp_XMol[:,1] = ExX
Step = 2;
NObs = Int(floor(TLen/Step))
psiNorm = zeros(TLen,1)                     # Stores ⟨ψ|H|ψ⟩ = ⟨ ̇dψ/dt⟩, which is a measure of proximity to the stationary distribution

# Evolve |ψ> over the desired time ----------------------
    for t = 2:TLen
        @time begin
        orthogonalize!(psi::MPS,1)
        TDVP_FwrdPass(psi::MPS,H::MPO,L::Int,dt::Float64,MMDim::Vector{Int64})  # Implements one time step of single site TDVP
        
        # Save observables -----
        MProb[t] = Contract_MPS(psi::MPS,L::Int64,n::Int64)
        Exp_XMol[:,t]  = Calculate_Expected_Molecules(psi::MPS,L::Int64,n::Int64);
        psiNorm[t] = Eq_Dist(psi::MPS,H::MPO)       
        @show t
        @show MProb[t]
        end
    end


return MProb,Exp_XMol

#Save out Data ---------------------------------------------------------
@save(FilePath*psiName, psi)

 fid = h5open(FilePath*FileName,"w")
 write(fid,"PrCons", [MProb dt:dt:TLen*dt])
 write(fid,"ExMol",Exp_XMol)
 write(fid,"psiNorm", psiNorm)  # This has not been run for L > 3!!!!!!
 close(fid)
