# DeepONet physics inferring for unknown parameters of partially-observable system

A framework to use Deep Operator Network to infer unknown parameters of partially-observable physical system. 
The sample problem we solve here is Allen-Cahn equation.

##  Motivation

Previous work on physics informed neural network (PINN) can infer solution of given physics problems assuming the whole differential equation is known in advance \cite{Raissi2020,Raissi2019}. However, in most dynamics or fluid dynamics problem, though we can affirm the governing differential equation, some parameters might remain unknown. Meanwhile, in some experiments, there are limitations that we cannot measure all the states. Therefore, it is worthwhile developing an architecture that can solve or approximate the solution only based on limited information. Moreover, field data is difficult to measure in a large extend, usually what we can measure are time snapshots at sparse locations. This motivates us to find a method which can be helping in inferring both the parameter and the whole physics field in this situation. 

For example, for a time-evolving physical system, a swimming fish in liquid, we can estimate the governing equation of the flow field, but the parameters are unknown. And we only can experimentally measure the velocity at limited locations, for example using PIV. 

How to infer unknown parameters of partially-observable physical system? We propose this DeepONet in the following sections.

## Objective

For a system with knowing governing equation $\mathcal{L}[\mathbf{u}(\mathbf{x},t);\boldsymbol{\alpha}] = 0$, unknown parameters $\boldsymbol{\alpha}$, and some experimental or numerical measurements $\mathbf{u}_s$, we want to recover the full field and find the unknown parameters. Particularly, we are solving 1D phase separation evolution system. The Allen–Cahn equation is a mathematical model for phase separation processes. The equation describes the time evolution of a scalar-valued state variable u on 1D space domain $x$ during a time interval $t$.

where $\epsilon$ is a small number, and $f'(u) = \frac{d f(u)}{du}$, with $f(u) = \frac{1}{4}(u^2-1)^2$, as a double-well potential. \par
Here, we consider dimensionless governing equation of state $u(x,t)$ on domain $x\in[-1,1],\; t\in[0,1]$ with periodic boundary conditions. Choose an initial state, we have the system $u_t = \alpha_1 u_{xx} - \alpha_2 (u^3 -  u)$ with I.C.: $u(t=0) = x^2 \cos(\pi x)$ and B.C.: $u(x=-1) = u(x=1)$a and $u_x(x=-1) = u_x(x=1)$.

where $\alpha_1\ll 1$, and $\alpha_2 \gg \alpha_1$. 

Assume we have some measurements of this system, and we want to use these measurements to recover the whole field and find the associated parameters set. The schematic diagram is in Fig.\ref{fig:sample}; assume we only have spatially sparse distributed sensors along these four lines where the arrows are pointing at. Thus, we can measure the field at these four specific locations along the whole time. 

The goal is using those measurements and above Allen-Cahn equation with IC\&BCs to recover the field $u$ and parameters $\alpha_1$, $\alpha_2$.


| ![Schematic](https://github.com/chenchenhuang/DeepONet_physics_inferring/blob/main/figures/sample_problem_v2.png) | 
|:--:| 
| *Schematic of sample problem* |



| ![Architecture](https://github.com/chenchenhuang/DeepONet_physics_inferring/blob/main/figures/NN_diagram.png) | 
|:--:| 
| *Basic architecture* |



| ![loss](https://github.com/chenchenhuang/DeepONet_physics_inferring/blob/main/figures/loss.png) | 
|:--:| 
| *Training loss and validation loss of DeepONet* |

| ![test](https://github.com/chenchenhuang/DeepONet_physics_inferring/blob/main/figures/partial_ob.png) | 
|:--:| 
| *Result with partial observations* |


