
# Lower-back environments for simulated ultrasound navigation 

Code for: 
```
@misc{hase2020ultrasoundguided,
	title={Ultrasound-Guided Robotic Navigation with Deep Reinforcement Learning},
	author={Hannes Hase and Mohammad Farid Azampour and Maria Tirindelli and Magdalini Paschali and Walter Simson and Emad Fatemizadeh and Nassir Navab},
	year={2020},
	eprint={2003.13321},
	archivePrefix={arXiv},
	primaryClass={cs.LG}
}
```

# The project
This project aims at learning a policy for autonomously navigating to the sacrum in simulated lower back environments from volunteers. As for the deep reinforcement learning agent, we use a double dueling DQN with a prioritized replay memory. 

For the implementation of this project, we used the [rl-zoo](https://github.com/araffin/rl-baselines-zoo) framework, a slightly adapted [stable-baselines](https://github.com/hhase/stable-baselines) library and an [environment](https://github.com/hhase/gym_sacrum_env) built using the [gym](https://gym.openai.com/) toolkit. 

# Environment
The environment is constructed following the gym-environment structure. Herefore, we implement three fundamental methods.

 - `init`: defines the dynamics of the environment. The environment's state transition function is defined in the function `build_transition_dict`. 
 - `step`: defines the agent-environment interaction for a single timestep.
 - `reset`: randomly initializes the agent in a random environment and position and also empties the memory buffers for previous transitions.

Additionally, we implement the `set` method, to initialize the environment at a given position to facilitate experiments and the `quiver_plot` function to visualize the agents performance at a given time during training.


