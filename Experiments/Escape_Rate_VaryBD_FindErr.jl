using ITensors
using LinearAlgebra
using JLD2
using HDF5
using KrylovKit: exponentiate
using Combinatorics

include("Tensor_Operations_NoQn.jl")
include("MPO_Mechanisms.jl")
include("Proj_Operator.jl")


# The goal is to evolve a collection of rates at fixed L and varying bond dimension -
# - For each L, find the rate that is within a certain error. Label the corresponding bond dimension d^*
# - Plot how the bond dimension scales for L
#
# Parameters
L = 4 # Number of sites
a = 1.06; b= .68;
Continue_Evolution =0

# -- Test parameters ----
    vc = 1;
    # Parameters to test exact rate -----------------
        M=85
        Vol = 1.2e-25 # M=85
        nStar = [15;25]
        vsim = 1
        k = [0.1824459,.000210046,2199.855,37.88136]
        DiffCoeffs = LinRange(0,2e-14,20)
        dval = 20
        Dx = DiffCoeffs[dval]
        BD = 30    # minimum bond dimension
    # ----------------------------------------------
    CoordDiv = nStar
    MMDim = [BD,50] #M=60, L < 10
n = M+1
N = Int(L-1)

#---------------------------------------------
MinTen = MMDim[1]
# Create the file
nA =  6.02214076e23;    #Avogadros constant
A = a*nA*Vol
B = b*nA*Vol
Conc = [1,1,1,2.27]
c = [k[1]*A/(nA*Vol)^2,k[2]/(nA*Vol)^2,k[3]*B,k[4]]; # Stochastic coefficients assuming kinetic coefficients are one
# Random diffusion values

h = (Vol*1000)^(1/3)
cX = Dx/h^2;
dt = .0001;    # time steps
T = 1.5;
TLen = Int(floor(T/dt))

    psiName = "psiStation30BD_"*"$M"*"Mol_"*"$L"*"sites_Dx2.jlds"
    to = 1
    ProbAB = zeros(TLen,2)
    Exp_XMol = zeros(L,TLen)
    MProb = zeros(TLen,1);
    Max_BondDim = zeros(L-1,TLen);
 
#Local Path
 Path = "C:/Users/snich/Documents/Julia_Programs/Rxn_Diffusion/Schlogl/Stationary_Dists/"
 @load(Path*psiName, psi)

# Zinc Path/ Quest Path
#@load(psiName, psi)

sites = siteinds(psi::MPS)  # Just defining the basis for the space you are in
psi0 = deepcopy(psi)

SaveData = 1
SavePsi = 0         #  DONT SAVE ALL THE DISTRIBUTIONS FOR THIS ONE!!!!!
PsiStep = 10
GrowBonds_Step = 10;
Cutoff = 1e-12^2    # Cutoff error for growing the bond dimension of psi

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

function ITensors.op!(Op::ITensor,
                  ::OpName"ProjA",
                  ::SiteType"S_Class",
                  s::Index)
                  for a=CoordDiv[2]+1:M+1
                      Op[s'=>a,s=>a] = 1
                  end
end
function ITensors.op!(Op::ITensor,
                  ::OpName"ProjB",
                  ::SiteType"S_Class",
                  s::Index)
                  for a=1:CoordDiv[1]+1
                      Op[s'=>a,s=>a] = 1
                  end
end
function ITensors.op!(Op::ITensor,
                  ::OpName"Proj_NotB",
                  ::SiteType"S_Class",
                  s::Index)
                  for a=CoordDiv[1]+1:M+1
                      Op[s'=>a,s=>a] = 1
                  end
end

function ITensors.op!(Op::ITensor,
                  ::OpName"QProj",
                  ::SiteType"S_Class",
                  s::Index)
                  Op[s'=>CoordDiv[1]+1,s=>CoordDiv[1]+1] = 1
end
function ITensors.op!(Op::ITensor,
                  ::OpName"NProj",
                  ::SiteType"S_Class",
                  s::Index)
                for a=1:CoordDiv[1]+1
                    Op[s'=>a,s=>a] = 1
                end
end
# Define the compliments of N and Q operators
function ITensors.op!(Op::ITensor,
                  ::OpName"QProjc",
                  ::SiteType"S_Class",
                  s::Index)
                  for a = 1:M+1
                      Op[s' =>a,s=>a] = 1
                  end
                  Op[s'=>CoordDiv[1]+1,s=>CoordDiv[1]+1] = 0
end
function ITensors.op!(Op::ITensor,
                  ::OpName"NProjc",
                  ::SiteType"S_Class",
                  s::Index)
                #  Op[s'=>CoordDiv[1]+1,s=>CoordDiv[1]] = 1
                for a = CoordDiv[1]+2:M+1
                    Op[s' =>a,s=>a] = 1
                end
end

#-------------------------------------------------------------------
function ProjectOutA!(psi::MPS,M::Int64,CoordDiv::Vector{Int64},L::Int64)
    sites = siteinds(psi)
    ProjVec = diagm([ones(CoordDiv[2]); zeros(M+1-CoordDiv[2])])
    for k = 1:L
        psi[k] = noprime!( psi[k]*ITensor(ProjVec,sites[k],sites[k]') )
    end
    return psi
end

function ProjectOutB!(psi::MPS,M::Int64,CoordDiv::Vector{Int64},L::Int64)
    psiB = deepcopy(psi)
    for k = 1:L
        ProjCTensor!(psiB[k]::ITensor,k::Int64,M::Int64,CoordDiv[1]::Int64)
    end
    return +(psi,-psiB,cutoff=1e-15,maxdim=80,mindim=20)
end
#-----------------------------------------------------------
H = MPO_Mechanisms(sites::Vector{Index{Int64}},L::Int64,M::Int64,c::Vector{Float64},"Schlogl_NoBOutFlow",cX)

    for k = 1:L
        ProjATensor!(psi[k]::ITensor,k::Int64,M::Int64,CoordDiv[2]::Int64)
    end
    #psi .= ProjectBoundary(psi::MPS,M::Int64,CoordDiv::Vector{Int64},L::Int64)
    LDims = linkdims(psi)
    # If MMDim[1] < we need to truncate our MPS
    if LDims[1] > MMDim[1]
        for k = 1:L-1
            @show k
            phi = psi[k]*psi[k+1]
            iphi = sites[k]
            if k > 1
                iphi = [iphi;commoninds(psi[k-1],psi[k])[1]]
            end
            @show iphi

            U,S,V =svd(phi,iphi,cutoff=1e-12,mindim=MMDim[1])

            psi[k] = U;
            psi[k+1] = S*V
        end
    end
    Z = Contract_MPS(psi::MPS,L::Int64,n::Int64)
    Z = Z^(1/L)
    psi = psi./Z

Proj_BComp =  Proj_NotB(sites::Vector{Index{Int64}},L::Int64,M::Int64,CoordDiv::Vector{Int64})
Proj_B = Proj_IntoB(sites::Vector{Index{Int64}},L::Int64,M::Int64,CoordDiv::Vector{Int64})

Prob = Contract_MPS(psi::MPS,L::Int64,M+1::Int64)
orthogonalize!(psi::MPS,1)

PrBChk = zeros(TLen,2)

MProb[1] = Prob
ProbAB[1,2] = Prob  #[C,B,A]
Exp_XMol[:,1] = Calculate_Expected_Molecules(psi::MPS,L::Int64,n::Int64);
Max_BondDim[:,1] = linkdims(psi)
ti = dt:dt:dt*TLen

function ProjectedProb(psi::MPS,CoordDiv::Vector{Int64},L::Int64,M::Int64)
    PrAB = zeros(1,2)
    psiC = deepcopy(psi)
    psiA = deepcopy(psi)

    for k = 1:L
          ProjCTensor!(psiC[k]::ITensor,k::Int64,M::Int64,CoordDiv[1]::Int64)
    end
    PrAB[1] = Contract_MPS(psiC::MPS,L::Int64,n::Int64)

    for k = 1:L
       psiA[k] = ProjATensor!(psiA[k]::ITensor,k::Int64,M::Int64,CoordDiv[2]::Int64)
    end
    PrAB[2] = Contract_MPS(psiA::MPS,L::Int64,n::Int64)
    PrAB;
end
# Define the path to save out data and make the initial values
#---------------------------------------------------------------
if SaveData == 1
    MinTen = MMDim[1]
    # Local Path ----------------------------------------
   # PathOut = "C:\\Users\\snich\\Documents\\MATLAB\\Tensor_Networks\\Schlogl\\Figures\\Rate_Constants\\EscapeVaryD\\"
    # Zinc Path -----------------------------------------
    # PathOut = "/home/sbn6912/RxnDiffusion/Schlogl/Data/EscapeRates/"
    # Quest Path
     PathOut = "/projects/p31555/RxnDiffusion/Schlogl/Data/EscapeErr_BD/"

    FileOut = "EscapeProb_Err_VBD"*"$M"*"Mol_"*"$L"*"sites_"*"$MinTen"*"BD_Dx"*"$dval"*"_V3.h5"
    fo = h5open(PathOut*FileOut,"w")
    write(fo,"ReactCoordV2",[ProbAB ti])
    write(fo,"ExpMol",Exp_XMol)
    write(fo,"ProbConsv",MProb)
    write(fo,"MaxBondDim",Max_BondDim)
    close(fo)

end
#----------------------------------------------------------------------------
# This is where the MPS data is saved out too.
if SavePsi == 1
    # Local Path ----------------------------------------
       #PsiPath = "C:\\Users\\snich\\Documents\\MATLAB\\Tensor_Networks\\Schlogl\\Figures\\Distributions\\PsiVaryD_"*"$M"*"Mol_"*"$L"*"sites_Dx"*"$dval"*"\\"
    # Quest Path
      PsiPath = "/projects/p31555/RxnDiffusion/Schlogl/Data/Distributions/PsiVaryBD_"*"$M"*"Mol_"*"$L"*"sites_"*"$BD"*"BondDim_"*"$dval"*"Dx/"
end

tTot = @elapsed begin
    for t=to+1:TLen
        @time begin

        orthogonalize!(psi::MPS,1)
        TDVP_FwrdPass(psi::MPS,H::MPO,L::Int,dt::Float64,MMDim::Vector{Int64})

        # Check total probability is conserved.
        MProb[t] = Contract_MPS(psi::MPS,L::Int64,n::Int64)
        Exp_XMol[:,t]  = Calculate_Expected_Molecules(psi::MPS,L::Int64,n::Int64);

        # Projected prob into basin B is wrong! its over counting.
        ProbAB[t,:] = ProjectedProb(psi::MPS,CoordDiv::Vector{Int64},L::Int64,M::Int64)
        Max_BondDim[:,t] = linkdims(psi)
#-------------------------------------------------------------------------------
    # Open the file we are saving and append to it.
        if  SaveData == 1
            fid = h5open(PathOut*FileOut,"w")
            # write the new appended files
            write(fid,"ReactCoordV2",[ProbAB ti])
            write(fid,"ExpMol",Exp_XMol)
            write(fid,"ProbConsv",MProb)
            write(fid,"MaxBondDim",Max_BondDim)
            close(fid)
        end
        if SavePsi == 1 && mod(t,PsiStep) == 0
            # Save out psi
            DistName = "psiStation"*"$M"*"Mol_"*"$L"*"sites_"*"$t"*"Time.jlds"
            @save(PsiPath*DistName, psi)
        end
#-------------------------------------------------------------------------------
       end
   end
end

if SaveData == 1
    fid = h5open(PathOut*FileOut,"cw")
    # Update fid by adding the total time of the calculation
     write(fid,"Func_Time",tTot)
    close(fid)
end
