# Easy 21

This repository contains the solution to Easy 21 assignment from [David Silver's Reinforcement Learning Series](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html). The actual assignment can be found [here](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/Easy21-Johannes.pdf).

Algorithms implemented include

- Monte Carlo Evaluation
- Monte Carlo Control
- TD Control (SARSA  $\lambda$ with eligibility traces)
- Linear Function Approximation
## Sarsa
![sarsa](/assets/sarsa.gif)
## Monte Carlo
![mc](/assets/mcpc.gif)

All of these are implemented in Easy 21, a spin-off of the classic Blackjack, with non-standard rules.

[requirements.txt](/requirements.txt) contains all the dependencies, and to install them run
`pip install -r requirements.txt` in the root directory of the project.

To reproduce the results, try running,
`python code/[agent].py` where agents include the ones defined above.
