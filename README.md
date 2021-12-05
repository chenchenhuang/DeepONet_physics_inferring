# DeepONet_physics_inferring

A framework to use Deep Operator Network to infer unknown parameters of partially-observable physical system. This is created for AME 508 FINAL PROJECT. 



The sample problem we solve here is Allen-Cahn equation.

 <img src="https://latex.codecogs.com/svg.latex?u_t=\alpha_1 u_{xx} -\alpha_2 (u^3-u)" title="\Large u_t=\alpha_1 u_{xx} -\alpha_2 (u^3)-u" />

The parameter space is <img src="https://latex.codecogs.com/svg.latex?\alpha_1" title="\Large \alpha_1" /> and <img src="https://latex.codecogs.com/svg.latex?\alpha_2" title="\Large \alpha_2" />.



We first train DeepONet to fit the model across a whole range of parameters space and use gradient descent to infer unknown parameters based on partial measurements. 