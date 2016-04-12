# Optimal Qubit Measurement

The goal of this project is to improve the estimation of a qubit Hamiltonian parameter, $$\Delta B_z$$. This parameter represents the hyperfine interaction between the qubit, a double quantum dot in GaAs, with the magnetic field from many GaAs nuclei. While this interaction with the environment could be interpreted as noise that will cause decoherence of the qubit state, if it can be estimated accurately, the effect can be tracked so information is not lost. 

In practice, estimating Delta B_z is a matter of preparing the qubit in a known state and then making measurements of its state after some evolution time. This can be used to glean some information about the value of $$\Delta B_z$$. This is done in current schemes by taking ~100 measurements at linearly increasing evolution times and then estimating $$\Delta B_z$$ before doing the actual operations on the qubit. The quality of the estimation, and thus the improvement of coherence time, is limited by only being able to make ~100 measurements before $$\Delta B_z$$ significantly changes and the fact that linearly increasing measurement times is not necessarily optimal.

For my project, I aim to account for drift and diffusion in $$\Delta B_z$$ so that more than 100 measurements can be used to estimate $$\Delta B_z$$, even though the drift and diffusion will still mean that there are diminishing returns. Also, I aim to implement an adaptive Bayesian measurement scheme that can determine the evolution time that minimizes variance in $$\Delta B_z$$ before each measurement is made.
