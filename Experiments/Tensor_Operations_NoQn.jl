function UpdateKp!(Kp::ProjMPO,psi::MPS,i::Int64)
        # Converts the single site operator H(i) into the zero site or imaginary operator K(i)

        Kp.H[i] = Kp.H[i]*psi[i]*prime(dag(psi[i]))
end

function TDVP(psi::MPS,H::MPO,L::Int,dt::Float64,MMDim::Vector{Int64})
# Runs one forward and one reverse TDVP sweep on psi using H, for time step dt
# Note: currently not guaranteed to work, use TDVP_FwrdPass below instead 

    MinD = MMDim[1]                     # Minimum bond dimension of MPS
    MaxD = MMDim[2]                     # Maximum bond dimension of MPS
    # This really effects accuracy and time to make a tdvp sweep!
    Cut_Off = 1e-20                     # maximum singular value size which is discarded
    err = 1e-12                         # The desired maximum error for Krylov expansion
    expCutOff = err./dt                 # Error value which is specified into exponentiate
    KryDim = 30                         # Number of Krylov basis vectors used
    InpParam = [MinD,MaxD,Cut_Off, expCutOff, KryDim]   # Collect parameters into one vector for each of use

# We will have three functions in this program, a forward loop, calculate N and a backwards loop

#function ComputeForward!(psi::MPS,Hp::ProjMPO,H::MPO,MaxD::Int64,MinD::Int64,Cut_Off::Float64,dt::Float64,N::Int64,L::Int64)
function ComputeForward!(psi::MPS,H::MPO,dt::Float64,L::Int64,InpParam::Vector{Float64})
    # Sweep over each voxel 
    Hpf = ProjMPO(H::MPO)
    Hpf.nsite = 1;

    for i = 1:L-1
        position!(Hpf::ProjMPO,psi::MPS,i::Int64);
        # Now evolve forward one time step
        newpsi,info = exponentiate(Hpf::ProjMPO,dt/2::Number,psi[i]::ITensor;tol=InpParam[4]::Float64,krylovdim=Int(InpParam[5])::Int64);

        # implement QR decompositions
        if i == 1
            Q,R = qr(newpsi::ITensor,sites[1]::Index{Int64},positive=true)
        else
            Q,R = qr(newpsi::ITensor,(inds(newpsi,tags="Site")[1]::Index{Int64},commoninds(inds(newpsi),psi[i-1])[1]::Index{Int64}),positive=true)
        end
        psi[i] = noprime(Q::ITensor)

        Kpf = deepcopy(Hpf::ProjMPO)    # Zero site Hamiltonian
        UpdateKp!(Kpf::ProjMPO,psi::MPS,i::Int64)
        Ctilde_Back,info = exponentiate(Kpf,-dt/2::Number,R::ITensor;tol=InpParam[4]::Float64,krylovdim=Int(InpParam[5])::Int64)
        info.converged==0 && throw("exponentiate did not converge")
        psi[i+1] *= Ctilde_Back

    end
    # Evolve the L'th site forward
    position!(Hpf::ProjMPO,psi::MPS,L::Int64);
    psiOut,info = exponentiate(Hpf::ProjMPO,dt/2::Number,psi[L]::ITensor;tol=InpParam[4]::Float64,krylovdim=Int(InpParam[5])::Int64);
    psi[L] = psiOut;

    return psi
end
#-------------------------------------------------------------------------------

#----------- Evolve every index from 2L-1 to 1 forward another dt/2-------------
function ReverseCompute!(psi::MPS,H::MPO,dt::Float64,L::Int64,InpParam::Vector{Float64})
    Hpf = ProjMPO(H::MPO)       # The matrix projection for the MPO H
    Hpf.nsite = 1;              # Specifying that the matrix projection is one-site instead of two
    phi = deepcopy(psi)
    for i = L:-1:2
        orthogonalize!(psi::MPS,i)
        position!(Hpf::ProjMPO,psi::MPS,i::Int64);  # Shifts the site of Hpf which is not projected onto
        # Now evolve forward one time step
        newpsi,info = exponentiate(Hpf::ProjMPO,dt/2::Number,psi[i]::ITensor;tol=InpParam[4]::Float64,krylovdim=Int(InpParam[5])::Int64);

      # Use svd here or figure out how to make R orthonormal
        Q,R = qr(newpsi::ITensor,noncommoninds(psi[i],psi[i-1]))        # Going right to left, we need the right matrix to be orthonormal, this amounts to switching QR -> RQ in the decomposition
        psi[i] = noprime(Q::ITensor)

       
        Kpr = deepcopy(Hpf::ProjMPO)
        UpdateKp!(Kpr::ProjMPO,psi::MPS,i::Int64)
        Ctilde_Back,info = exponentiate(Kpr,-dt/2::Number,R::ITensor;tol=InpParam[4]::Float64,krylovdim=Int(InpParam[5])::Int64)
        info.converged==0 && throw("exponentiate did not converge")
      
        # Convert psi[i] and phi to Arrays and compute the norm of their difference -------
        psi[i-1] *= Ctilde_Back
    end
    position!(Hpf::ProjMPO,psi::MPS,1::Int64);

    psiOut,info = exponentiate(Hpf::ProjMPO,dt/2::Number,psi[1]::ITensor;tol=InpParam[4]::Float64,krylovdim=Int(InpParam[5])::Int64);
    psi[1] = psiOut;
    return psi;
end

ComputeForward!(psi::MPS,H::MPO,dt::Float64,L::Int64,InpParam::Vector{Float64})
ReverseCompute!(psi::MPS,H::MPO,dt::Float64,L::Int64,InpParam::Vector{Float64})
return psi
end

#----------------------------------------------------------------------
function ExpContract(Hctr::ITensor,Lind::Vector{Index{Int64}},Rind::Vector{Index{Int64}}, psi::ITensor)
    # Calculates the matrix exponential of a one-site MPO as an Array, before converting back to an ITensor
    # NOTE: only for developmental use, do not use for real calculations!!!!!!!!!!!
    CL = prime(combiner(inds(psi)))
    CR = combiner(inds(psi))
    @show Lind
    @show Rind
    ITensor(exp(Array(CL*Hctr*CR,Lind,Rind)),prime(inds(psi)),inds(psi))*psi
end
#-----------------------------------------------------------------------
function ProjATensor!(psi::ITensor,k::Int64,M::Int64,CoordDiv::Int64)
    # Basin A is specified by (CoordDiv[2]:M)
    # Projects psi onto the A basin

    IndPsi = inds(psi,tags="n="*"$k")
    ProjArray = zeros(M+1,M+1)
    PAd = diagind(ProjArray)
    ProjArray[PAd[CoordDiv+1:end]] .= 1
    Proj = ITensor(ProjArray,IndPsi,IndPsi')
    psi .= noprime!(Proj*psi)
end
#-----------------------------------------------------------------------
function ProjBTensor!(psi::ITensor,k::Int64,M::Int64,CoordDiv::Vector{Int64})
    # B will be the middle set non-inclusive i.e. (CoordDiv[1]+2 to CoordDiv[2]-1)
    # Projects onto the hypercube in between the two metastable basins
    IndPsi = inds(psi,tags="n="*"$k")
    ProjArray = zeros(M+1,M+1)
    PAd = diagind(ProjArray)
    ProjArray[PAd[CoordDiv[1]+2:1:CoordDiv[2]-1]] .= 1
    Proj = ITensor(ProjArray,IndPsi,IndPsi')
    psi .= noprime!(Proj*psi)
end
#-----------------------------------------------------------------------
function ProjCTensor!(psi::ITensor,k::Int64,M::Int64,CoordDiv::Int64)
    # C will be the small M region, i.e. left most region (0 to CoordDiv[1]+1)
    # Note: this basin is region B in the DP-TN paper

    IndPsi = inds(psi,tags="n="*"$k")
    ProjArray = zeros(M+1,M+1)
    PAd = diagind(ProjArray)
    ProjArray[PAd[1:1:CoordDiv+1]] .= 1
    Proj = ITensor(ProjArray,IndPsi,IndPsi')
    psi .= noprime!(Proj*psi)
end
#-----------------------------------------------------------------------
function ProjSingleTensor!(psi::ITensor,k::Int64,M::Int64,Coord::Int64)
    # Project onto one value of O in the GTS model
    IndPsi = inds(psi,tags="n="*"$k")
    ProjArray = zeros(M+1,M+1)
    PAd = diagind(ProjArray)
    ProjArray[PAd[Coord]] = 1.0

    Proj = ITensor(ProjArray,IndPsi,IndPsi')
    noprime!(Proj*psi)
end


#--------------------------------------------------------------------
function AbsoluteMPS!(psi::MPS)
    # Takes the absolute value of every core tensor in the MPS
    for k = 1:length(psi)
        psi[k] = broadcast(abs,psi[k])
    end
    psi;
end

function Normalize!(psi::MPS,Z::Float64,L::Int64)
    # Given an MPS normalizes in the sense that ⟨1|ψ⟩ = 1
    for k = 1:L
        psi[k] .= psi[k]./Z
    end
end
#-----------------------------------------------------------------------

function Norm_Of_MPS(psi2::MPS,psi::MPS)
# Compute a norm between the core tensors of two MPS i.e. ⟨ψ|ϕ⟩/n

n = length(psi)
NVec = zeros(n,1)
for k = 1:n
    Ipsi = inds(psi[k])
    Iphi = inds(psi2[k])
    swapinds!(psi2[k],Iphi,Ipsi)
    NVec[k] =  norm(psi[k]-psi2[k])
end

return sum(NVec)/n, psi2
end

#-------------------------------------------------------------------------------
function Calculate_Expected_Molecules(psi::MPS,L::Int64,n::Int64)
# Read in a state |ψ⟩ and calculate the expected number of molecules in each site
# n = M+1 molecules allowed per subspace
# L: number of lattice sites

ExX = zeros(L,1)
sites = siteinds(psi)               # Extracts the site indices from psi
# Run through each site and calculate the average
for l = 1:L
    nX = op("NumOp",sites[l])       # Calls the number operator tensor. Must be specified in base function
    Ex_Temp = ITensor()
    if l == 1
        EX_temp  = psi[1]*nX*prime( ITensor(ones(1,n),sites[1]) )
    else
        EX_temp = psi[1]*ITensor(ones(1,n),sites[1])
    end

    for k = 2:L
        # Calculate the expectation of X
        if k == l
            # We are applying the expectation here
            EX_temp  *= psi[k]*nX*prime( ITensor(ones(1,n),sites[k]) )
        else
            EX_temp *= psi[k]*ITensor(ones(1,n),sites[k])
        end
    end
    ExX[l] = scalar(EX_temp)
end
ExX;
end 
# ------------------------------------------------------------------------------
function Contract_MPS(psi::MPS,L::Int64,n::Int64)
    # Read in an MPS and efficiently contract all indicies
    # of use when seeing if for instance a joint distribution conserves probaiblity
    sites = siteinds(psi)
    psiProd  = psi[1]*ITensor(ones(1,n),sites[1])
    for k = 2:L
        psiProd *= psi[k]*ITensor(ones(1,n),sites[k])
    end
    scalar(psiProd);
end
#-------------------------------------------------------------------------------
function Overlap_MPS(psi::MPS,L)
    sites = siteinds(psi)
    psiProd = psi[1]*prime(psi[1],tags="Link")

    for k = 2:L
        psiProd *= psi[k]*prime(psi[k],tags="Link")
    end
    return scalar(psiProd)
end

#-------------------------------------------------------------------------------
function Eq_Dist(psi::MPS,H::MPO)
# Compute ⟨ψ|H|ψ⟩ = ⟨dψ/dt⟩

L = length(psi)
Eq_Dist = prime(psi[1])*H[1]*psi[1]
for k = 2:L
    Eq_Dist *= prime(psi[k])*H[k]*psi[k]
end

scalar(Eq_Dist)
end

#---------------------------------------------------
function DMRG_Initialization!(psi::MPS,H::MPO,MMDim::Vector{Int64},Noise::Float64)
    # Used to initialize a DMRG calculation, initially made to work with the gene toggle switch program
    sweeps = Sweeps(1)
    setmaxdim!(sweeps, MMDim[2])
    setmindim!(sweeps,1)
    setcutoff!(sweeps, 1E-12)
    setnoise!(sweeps, 0)

    energy, psi0 = dmrg(H,psi, sweeps, outputlevel=1::Int,ishermitian=false::Bool);
    return psi0, energy
end
#---------------------------------------------------------------
function Marginal_Dist(psi::MPS,k::Int64,n::Int64)
# Read in an MPS and calculate the marginal distribution for site k
PrMarg = zeros(n,1) # The vector which will be output
L = length(psi)
sites = siteinds(psi)
# Run through each site and calculate the average
# Compute the left side
    if k > 1
        LTen = psi[1]*ITensor(ones(1,n),sites[1])
        for j = 2:k-1
            LTen = LTen*psi[j]*ITensor(ones(1,n),sites[j])
        end
    end
    if k < L
        RTen = psi[L]*ITensor(ones(1,n),sites[L])
        for j = L-1:-1:k+1
            RTen = RTen*psi[j]*ITensor(ones(1,n),sites[j])
        end
    end

    if k == 1
        PrMarg = array(psi[1]*RTen)
    elseif k == L
        PrMarg = array(LTen*psi[L])
    else
        PrMarg = array(LTen*psi[k]*RTen)
    end
    PrMarg
end
# ----------------------------------------------------------
function TDVP_FwrdPass(psi::MPS,H::MPO,L::Int,dt::Float64,MMDim::Vector{Int64})
    # -Makes on left to right sweep of an MPS and implements a single site TDVP algorithm
    # -Program was used for all results in the DP-TN rate calculation paper

    MinD = MMDim[1]         # Minimum bond dimension for MPS
    MaxD = MMDim[2]         # Maximum bond dimension of MPS (only needed for )
    Cut_Off = 1e-50         # This really effects accuracy and time to make a tdvp sweep!
    #err = 1e-15
    #Cut_Off = 1e-15
    err = 1e-30             # The overall error parameter guiding exponentiation
    expCutOff = err./dt     # err is variable with timestep so using this version is more efficient.
    KryDim = 30             # The number of Krylov vectors in your basis set
    InpParam = [MinD,MaxD,Cut_Off, expCutOff, KryDim]   # Group you input variables to be more efficient
    Hpf = ProjMPO(H::MPO)       # The matrix projection for the MPO H
    Hpf.nsite = 1;              # Specifying that the matrix projection is one-site instead of two
    for i = 1:L-1
       
        position!(Hpf::ProjMPO,psi::MPS,i::Int64);  # Shifts the site of Hpf which is not projected onto
        # Now evolve forward one time step
        # Computes the matrix exponetial and moves the ith site of ψ forward by dt
        newpsi,info = exponentiate(Hpf::ProjMPO,dt::Number,psi[i]::ITensor;tol=InpParam[4]::Float64,krylovdim=Int(InpParam[5])::Int64);
        info.converged==0 && throw("exponentiate did not converge")

        # Do a QR decomposition in order to form the imaginary tensor which is evolved back in time
        if i == 1
            Q,R = qr(newpsi::ITensor,sites[1]::Index{Int64},positive=true)
        else
            Q,R = qr(newpsi::ITensor,(inds(newpsi,tags="Site")[1]::Index{Int64},commoninds(inds(newpsi),psi[i-1])[1]::Index{Int64}),positive=true)
        end

        psi[i] = noprime(Q::ITensor)
        # Compute the imaginary center site MPO
        Kpf = deepcopy(Hpf::ProjMPO)
        UpdateKp!(Kpf::ProjMPO,psi::MPS,i::Int64)       # Convert the 1 site Hamiltonian into the zero site Hamiltonian
        # Evolve the zero site tensor backwards ----
        Ctilde_Back,info = exponentiate(Kpf,-dt::Number,R::ITensor;tol=InpParam[4]::Float64,krylovdim=Int(InpParam[5])::Int64)
        info.converged==0 && throw("exponentiate did not converge")
        psi[i+1] *= Ctilde_Back                         # After back evolution contract zero site tensor with i+1 tensor

    end
    # Evolve the L'th site forward --------------------
    position!(Hpf::ProjMPO,psi::MPS,L::Int64);
    psiOut,info = exponentiate(Hpf::ProjMPO,dt::Number,psi[L]::ITensor;tol=InpParam[4]::Float64,krylovdim=Int(InpParam[5])::Int64);
    info.converged==0 && throw("exponentiate did not converge")
    psi[L] = psiOut;
    return psi;
end
#-------------------------------------------------------------------------------
function Process_JointDistributions(psi::MPS,Nj::Int64)
    # Project a full MPS down onto joint distributions over smaller subspaces. For example,
    # One can project onto all two site marginals, sites Pr(n_1,n_2) and Pr(n_2,n_3) etc.
    # In future update so one can ask for joints between non-neighboring sites ------

    sites = siteinds(psi)           # Extract the sites from ψ           
    L = length(psi)                 # Number of sites in ψ
    # Local FilePath
    FilePath = "C:\\Users\\snich\\Documents\\MATLAB\\Tensor_Networks\\Schlogl\\Figures\\JointDistributions\\"*"$Nj"*"Joint_"*"$L"*"Sites\\"
    # If Nj =2 we want every two site joint i.e. (1,2), (2,3), ....
    if Nj == 2
        for l = 1:L-1
            # The MPS we act on
            phi = deepcopy(psi)

            # Contract all but l and l+1 physical indices
            for k = 1:l-1
                phi[k] *= ITensor(ones(1,n),sites[k])
            end
            for k = l+2:L
                phi[k] *= ITensor(ones(1,n),sites[k])
            end
            # Convert the remaining Itensor into a normal array
            PJoint = Array(prod(phi),sites[l],sites[l+1]);

            # Save out the array 
            FileName = "PJoint_"*"$l"*".h5"
            fid = h5open(FilePath*FileName,"w")
             write(fid,"Prob_Joint",PJoint)
             close(fid)
         end

    end

    # Projects onto all three variable joints between neighboring sites, Pr(n_1,n_2,n_3) Pr(n_2,n_3,n_4) etc
    if Nj == 3
        for l = 1:L-2
            phi = deepcopy(psi)
            # Contract all but l, l+1 and l+2 physical indices
            for k = 1:l-1
                phi[k] *= ITensor(ones(1,n),sites[k])
            end
            for k = l+3:L
                phi[k] *= ITensor(ones(1,n),sites[k])
            end
            PJoint = Array(prod(phi),sites[l],sites[l+1],sites[l+2]);

            FileName = "PJoint_"*"$l"*".h5"
            fid = h5open(FilePath*FileName,"w")
             write(fid,"Prob_Joint",PJoint)
             close(fid)
         end
     end
end
#-------------------------------------------------------------------------------
function GrowSpace!(psi::MPS,MNew::Int64)
    # Given an MPS which has a maximum physical dimension of M, this functions expands it to a new dimension MNew
    # Originally developed for the Gene toggle switch model

    sites2 = siteinds(n->in([3,4,5]).(n) ? "SConsv" : "SNoConsv", 7; conserve_qns=true)
    psi2 = MPS(sites2)
    for k = 1:7
        psiTemp = ITensor(sites2[k],inds(psi[k],tags="Link"))
        psi_inds = inds(psiTemp)
        psid = dense(psi[k])
        ip = inds(psid)
            if k == 1 || k == 7
                psiAr = Array(psid,ip[1],ip[2])
            else
                psiAr = Array(psid,ip[1],ip[2],ip[3])
            end
            nzCoords = findall(!=(0),psiAr)
            psik = psi[k]
            ipk = inds(psik)
            Len_nzElm = length(nzCoords)
            for l = 1:Len_nzElm
                if k == 1 || k == 7
                    psiTemp[psi_inds[1]=>nzCoords[l][1],psi_inds[2] =>nzCoords[l][2]] = psik[ipk[1]=>nzCoords[l][1],ipk[2] =>nzCoords[l][2]]
                else
                    psiTemp[psi_inds[1]=>nzCoords[l][1],psi_inds[2] =>nzCoords[l][2],psi_inds[3]=>nzCoords[l][3]] = psik[ipk[1]=>nzCoords[l][1],ipk[2] =>nzCoords[l][2],ipk[3]=>nzCoords[l][3]]
                end
            end
       psi2[k] = psiTemp
    end
    return psi2;
end
#-------------------------------------------------------------------------------
function TwoSite_GrowBonds!(psi::MPS,d::Int64,dInc::Int64,Cutoff::Float64)
# Given an MPS with a fixed bond index N, this function with grow the bond index to a new value M, which is not larger than N^2
# psi, the input MPS
# d, the old bond dimension
# dInc, the amount we want to grow the bond dimension

L = length(psi)
    # Sweep combine two tensors at a time, svd and grow the bond dimension

    for l = 1:L-1
        phi = psi[l]*psi[l+1]
        iphi = inds(phi)
        u,s,v = svd(phi,commoninds(psi[l],iphi),mindim=MMDim[1],maxdim=MMDim[2],cutoff=Cutoff)
        psi[l] = u
        psi[l+1] = s*v
    end
    return psi
end
#--------------------------------------------------------------------------------
