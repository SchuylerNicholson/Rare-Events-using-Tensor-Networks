# Measuring Rare-Events using Tensor Networks
<img src="Git_Figures/JointPDF_250M_2L.png" width=50% height=50%>
Above, the stationary joint probability distribution for the Schlogle model over two neighboring reaction voxels. Each point is the probability of observing the number of molecules in voxel one and voxel two.

# Installation
Required packages can be added using the Julia package manager. From the Julia Repl, type `]` to enter the package manager and run for example,
```julia
julia> ]

pkg> add ITensors
```

# Usage
Results from the paper can be reproduced by running files from the ``Experiments`` directory.

``Schlogl_Evolve.jl`` Will run an initially uniform distribuiton for a set amount of time, eventually reaching the stationary distribution.

The stationary distribution can be used as an input into ``Proj_EscapeProb.jl`` to calculate the rate of transitioning from basin A to basin B

``Escape_Rate_Vary_Diff.jl`` is a variant that was used to calculate the transition for various diffusion coefficients.

``Escape_Rate_VaryBD_FindErr`` is another variant used to calculate transition rates for a fixed number of voxels but varying bond-dimension.

# Referencing
If you found this repository useful please consider citing,

Nicholson, Schuyler B., and Todd R. Gingrich. "Quantifying rare events in stochastic reaction-diffusion dynamics using tensor networks." Physical Review X 13, no. 4 (2023): 041006.

```
@article{nicholson2023quantifying,
  title={Quantifying rare events in stochastic reaction-diffusion dynamics using tensor networks},
  author={S.B.~Nicholson, and T.R.~Gingrich},
  journal={Physical Review X},
  volume={13},
  number={4},
  pages={041006},
  year={2023},
  publisher={APS}
}
```
