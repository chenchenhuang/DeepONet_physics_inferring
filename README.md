# DeepONet physics inferring for unknown parameters of partially-observable system

A framework to use Deep Operator Network to infer unknown parameters of partially-observable physical system. 
The sample problem we solve here is Allen-Cahn equation.

##  Motivation

Prior studies on Physics-Informed Neural Networks (PINNs) have made significant strides in estimating solutions for specified physics-related problems, premised on the assumption that the complete differential equation associated with the problem is pre-established. This assumption, however, may not hold true in most scenarios, especially in dynamic or fluid dynamic contexts, wherein the governing differential equations may be determined, yet certain parameters might remain unidentified.

Furthermore, the reality of experimental limitations often implies that not all states can be measured comprehensively. Therefore, the development of an architecture capable of solving or approximating solutions based on restricted information becomes crucial and significant.

This challenge is further compounded by the complexity of obtaining field data in large extents. More often than not, measurements can only be taken at discrete intervals, in the form of temporal snapshots at sparsely located points. These circumstances present a strong case for a method that can not only infer the unknown parameters but also reconstruct the complete physics field from limited information.

Consider, for instance, a time-evolving physical system like a fish swimming in a liquid medium. Here, while the governing equation of the flow field can be reasonably estimated, the specific parameters may remain unknown. Moreover, our experimental capabilities might only allow us to measure the velocity at certain limited locations, potentially using methods like Particle Image Velocimetry (PIV).

This raises a pertinent question: how can we infer the unknown parameters of such a partially-observable physical system? To address this, we propose the use of a novel method called DeepONet, which we will explore and explain in the ensuing sections. This approach seeks to enhance our ability to comprehend and interact with complex physical systems, enabling more accurate predictions and simulations.

## Objective

For a system with knowing governing equation $\mathcal{L}[\mathbf{u}(\mathbf{x},t);\boldsymbol{\alpha}] = 0$, unknown parameters $\boldsymbol{\alpha}$, and some experimental or numerical measurements $\mathbf{u}_s$, we want to recover the full field and find the unknown parameters. Particularly, we are solving 1D phase separation evolution system. The Allen–Cahn equation is a mathematical model for phase separation processes. The equation describes the time evolution of a scalar-valued state variable u on 1D space domain $x$ during a time interval $t$.

where $\epsilon$ is a small number, and $f'(u) = \frac{d f(u)}{du}$, with $f(u) = \frac{1}{4}(u^2-1)^2$, as a double-well potential. \par
Here, we consider dimensionless governing equation of state $u(x,t)$ on domain $x\in[-1,1],\; t\in[0,1]$ with periodic boundary conditions. Choose an initial state, we have the system $u_t = \alpha_1 u_{xx} - \alpha_2 (u^3 -  u)$ with I.C.: $u(t=0) = x^2 \cos(\pi x)$ and periodic B.C.: $u(x=-1) = u(x=1)$ and $u_x(x=-1) = u_x(x=1)$.

where $\alpha_1\ll 1$, and $\alpha_2 \gg \alpha_1$. 

Assume we have some measurements of this system, and we want to use these measurements to recover the whole field and find the associated parameters set. The schematic diagram is in Fig 1; assume we only have spatially sparse distributed sensors along these four lines where the arrows are pointing at. Thus, we can measure the field at these four specific locations along the whole time. 

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


