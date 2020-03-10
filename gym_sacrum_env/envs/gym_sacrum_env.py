import os
import gym
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.transform import resize
from gym import error, spaces, utils
from mpl_toolkits.axes_grid1 import make_axes_locatable

class SacrumNavEnv(gym.Env):

    def __init__(self, env_params={}):
        self.verbose = env_params['verbose']
        self.chebishev = env_params['chebishev']

        self.resize_x = 272
        self.resize_y = 258

        self.test_patients = env_params['test_patients']
        self.val_patients = env_params['val_patients']

        print("Validating on {} patients".format(self.val_patients))
        print("Testing on {} patients".format(self.test_patients))

        self.max_time_steps = env_params['max_time_steps']
        self.time_step_limit = env_params['time_step_limit']
        self.no_nop = env_params['no_nop']
        self.action_memory = env_params['prev_actions']
        self.frame_history = env_params['prev_frames']
        self.shuffles = env_params['shuffles']
        self.defined_test_set = env_params['test_set']
        self.defined_val_set = env_params['val_set']

        if self.verbose > 0 and self.chebishev:
            print("Training with Chebishev distance!")

        if not env_params['data_path']:
            raise ValueError('Valid patient file is needed for this environment to work')

        self.counter = 0                                                        # Timestep counter in each episode
        self.correct_counter = 0                                                # Correct decision counter each episode
        self.num_rows = 11                                                      # Grid heigth component
        self.num_cols = 15                                                      # Grid width component
        self.max_row = self.num_rows - 1                                        # Vertical movement limitation
        self.max_cols = self.num_cols - 1                                       # Horizontal movement limitation

        patient_files = os.listdir(env_params['data_path'] + "Patient_files/")
        [patient_files.remove(pat_file) if not ".txt" in pat_file else "" for pat_file in patient_files]
        patient_files.sort()
        print(patient_files)
        self.training = True

        self.num_patients = len(patient_files)                                  # Total n° patients
        patient_idxs = np.array(list(range(self.num_patients)))
        np.random.shuffle(patient_idxs)

        if np.any(self.defined_test_set):
            self.test_patient_idxs = self.defined_test_set
            rest_patients = np.array([patient for patient in patient_idxs if patient not in self.test_patient_idxs])
        else:
            self.test_patient_idxs, rest_patients = np.split(patient_idxs, [self.test_patients])

        if np.any(self.defined_val_set):
            self.val_patient_idxs = self.defined_val_set
            self.train_patient_idxs = np.array(
                [patient for patient in rest_patients if patient not in self.val_patient_idxs])
        else:
            for _ in range(self.shuffles):
                np.random.shuffle(rest_patients)
            self.val_patient_idxs, self.train_patient_idxs = np.split(rest_patients, [self.val_patients])

        if self.verbose > 0: print(self.train_patient_idxs)
        if self.verbose > 0: print(self.val_patient_idxs)
        if self.verbose > 0: print(self.test_patient_idxs)

        self.patient_idx = self.reset_patient()  # Current patient index
        self.goals = []                                                         # Goal coordinates for patients - y * row_size + x format
        self.goal_avg = []                                                      # Center of goals for distance calculation
        self.frames_per_state = 5
        self.frames = []                                                        # US-frames for DQN input

        self.num_states = self.num_rows * self.num_cols
        self.val_reachability = 0

        if self.chebishev:
            self.action_space = spaces.Discrete(9)
            self.num_actions = 9  # 0 - UP | 1 - DOWN | 2 - LEFT | 3 - RIGHT | 4 - NOP | 5 - NE | 6 - SE | 7 - SW | 8 - NW
        elif self.no_nop:
            self.action_space = spaces.Discrete(4)
            self.num_actions = 4  # 0 - UP | 1 - DOWN | 2 - LEFT | 3 - RIGHT
        else:
            self.action_space = spaces.Discrete(5)
            self.num_actions = 5  # 0 - UP | 1 - DOWN | 2 - LEFT | 3 - RIGHT | 4 - NOP

        self.observation_space = spaces.Box(low=0, high=1, shape=(self.resize_x, self.resize_y, self.frame_history + 1),
                                            dtype=np.float)

        self.state = self.reset_state()

        self.prev_actions = np.zeros(self.num_actions * self.action_memory)
        self.prev_frames = np.zeros((self.frame_history), dtype=int)

        pat_counter = 0
        for i, patient in zip(range(len(patient_files)), patient_files):
            pat_counter += 1
            goals, frames = self.load_patient(env_params['data_path'], patient)
            goal_avg = np.average(goals)

            self.goals.append(goals)
            self.frames.append(frames)
            self.goal_avg.append(goal_avg)
            if self.verbose > 0: print("Patient {}/{} loaded!".format(pat_counter, self.num_patients))

        self.reward_dict = {'goal_correct': 1.0,  # 1.0
                            'goal_incorrect': -0.25,  # -0.25
                            'closer': 0.05,  # 0.05
                            'border_collision': -0.1,  # -0.25
                            'further': -0.1}  # -0.1

        self.P = self.build_transition_dict()

        if self.verbose > 0:
            print("Environment loaded!")
            print("Training patients: {}".format(",".join(map(str, self.train_patient_idxs))))
            print("Testing patients: {}".format(",".join(map(str, self.test_patient_idxs))))

    def step(self, action):
        if self.counter == 0 and self.verbose > 1: print("Changing to patient nr {}".format(self.patient_idx))
        self.counter += 1
        info = dict()
        if self.verbose > 1: print("Training step {} - coords: {},{} doing {}!".format(self.counter, self.get_row(self.state), self.get_col(self.state), action))

        transition = self.P[self.patient_idx][self.state][action]

        p, state, reward, done = transition[0]
        goals = self.goals[self.patient_idx]

        if reward == self.reward_dict['closer'] or reward == self.reward_dict['goal_correct']:
            self.correct_counter += 1

        if (self.num_actions == 5 and reward == self.reward_dict['goal_correct']) or (
                self.num_actions == 4 and state in goals):
            done = True
            if self.verbose > 1: print("Done!")
            info = self.log_final_state(is_success=True)
            self.reset_params()

        elif self.max_time_steps and self.counter == self.time_step_limit:
            done = True
            info = self.log_final_state(is_success=None)
            self.reset_params()

        self.state = state
        self.prev_frames = np.roll(self.prev_frames, 1, axis=0)
        self.prev_frames[0] = state
        if np.any(self.prev_actions):
            self.prev_actions = np.roll(self.prev_actions, self.num_actions, axis=0)
            self.prev_actions[:self.num_actions].fill(0)
            self.prev_actions[action] = 1

        observation = self.generate_observation()

        info.update({"coords": self.val_to_coords(self.state)})
        return (observation, reward, done, info)

    def reset(self):
        self.reset_params()
        self.patient_idx = self.reset_patient()
        self.state = self.reset_state()
        self.prev_actions.fill(0)
        self.prev_frames.fill(self.state)

        observation = self.generate_observation()

        return observation

    def set(self, patient, state=0):
        self.reset_params()
        self.patient_idx = patient
        self.state = state
        self.prev_actions.fill(0)
        self.prev_frames.fill(self.state)

        observation = self.generate_observation()

        return observation

    def generate_observation(self):

        frames = np.zeros((self.resize_x, self.resize_y, len(self.prev_frames)), dtype=np.float64)

        for i, prev_state in zip(range(len(self.prev_frames)), self.prev_frames):
            frame_idx = np.random.randint(self.frames_per_state)
            frame = self.frames[self.patient_idx][prev_state][frame_idx]
            frames[:, :, i] = frame.reshape(self.resize_x, self.resize_y)

        action_history = np.zeros((self.resize_x, self.resize_y, 1))
        action_history[0, 0:len(self.prev_actions), 0] = self.prev_actions
        observation = np.dstack((frames, action_history))

        return observation

    def dist_to_goal(self, patient, state):
        x, y = self.val_to_coords(state)
        goals_x = []
        goals_y = []
        for goal in self.goals[patient]:
            goal_x, goal_y = self.val_to_coords(goal)
            goals_x.append(goal_x)
            goals_y.append(goal_y)
        goals_x = np.array(goals_x)
        goals_y = np.array(goals_y)

        return np.min(np.abs(goals_x - x)) + np.min(np.abs(goals_y - y))

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        pass

    def build_transition_dict(self):

        transition_dict = {
            patient: {state: {action: [] for action in range(self.num_actions)} for state in range(self.num_states)}
            for patient in range(self.num_patients)}

        for pat in range(self.num_patients):
            goals = self.goals[pat]
            goal_avg = self.goal_avg[pat]
            for row in range(self.num_rows):
                for col in range(self.num_cols):
                    state = row * self.num_cols + col
                    for action in range(self.num_actions):
                        new_row = row
                        new_col = col
                        reward = 0
                        done = False
                        if action == 0:  # UP
                            new_row = max(row - 1, 0)
                        elif action == 1:  # DOWN
                            new_row = min(row + 1, self.max_row)
                        elif action == 2:  # LEFT
                            new_col = max(col - 1, 0)
                        elif action == 3:  # RIGHT
                            new_col = min(col + 1, self.max_cols)
                        elif action == 5:  # UP-RIGHT
                            if self.check_boundaries(row - 1, col + 1):
                                new_row = max(row - 1, 0)
                                new_col = min(col + 1, self.max_cols)
                        elif action == 6:  # DOWN-RIGHT
                            if self.check_boundaries(row + 1, col + 1):
                                new_row = min(row + 1, self.max_row)
                                new_col = min(col + 1, self.max_cols)
                        elif action == 7:  # DOWN-LEFT
                            if self.check_boundaries(row + 1, col - 1):
                                new_row = min(row + 1, self.max_row)
                                new_col = max(col - 1, 0)
                        elif action == 8:  # UP-LEFT
                            if self.check_boundaries(row - 1, col - 1):
                                new_row = max(row - 1, 0)
                                new_col = max(col - 1, 0)

                        next_state = new_row * self.num_cols + new_col

                        if action == 4:  # NOP
                            if next_state in goals:
                                reward = self.reward_dict['goal_correct']
                                done = True
                            else:
                                reward = self.reward_dict['goal_incorrect']
                        else:
                            if next_state == state:
                                reward = self.reward_dict['border_collision']
                            else:
                                distance_before = self.distance_to_goal(pat, state)
                                distance_after = self.distance_to_goal(pat, next_state)
                                closer = (distance_after < distance_before)
                                # closer = self.is_closer(state, next_state, goal_avg)
                                if closer:
                                    reward = self.reward_dict['closer']
                                else:
                                    reward = self.reward_dict['further']
                        if self.no_nop and next_state in goals:
                            done = True
                        transition_dict[pat][state][action].append((1.0, next_state, reward, done))

        return transition_dict

    def quiver_plot(self, patient=None, states=None, actions=None, q_values=None, state_vals=None, reachability=False):
        forward_dict = {state: [] for state in range(self.num_states)}

        q_val_plots = True if np.any(q_values) else False
        state_val_plot = True if np.any(state_vals) else False
        reach_plot = True if reachability else False

        patient = patient if patient else self.patient_idx
        correctness = 0

        u_dict = {0: 0.0, 1: 0.0, 2: -1.0, 3: 1.0, 4: 0.0, 5: 0.707, 6: 0.707, 7: -0.707,
                  8: -0.707}  # Horizontal component
        v_dict = {0: 1.0, 1: -1.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.707, 6: -0.707, 7: -0.707,
                  8: 0.707}  # Vertical component

        if np.sum([q_val_plots, state_val_plot, reach_plot]) >= 2:
            raise NotImplementedError("Cannot plot Q and state-values simultaneously!")

        chosen_value = 5
        if q_val_plots:
            fig, ax = plt.subplots(self.num_actions + 1, 1,
                                   figsize=(1.05 * chosen_value, (self.num_actions + 1) * 0.8 * chosen_value),
                                   subplot_kw={'aspect': 1}, squeeze=False)
        elif state_val_plot or reach_plot:
            fig, ax = plt.subplots(2, 1, figsize=(1.05 * chosen_value, 2 * 0.8 * chosen_value), squeeze=False)
        else:
            fig, ax = plt.subplots(1, 1, figsize=(1.05 * chosen_value, 2 * 0.8 * chosen_value), squeeze=False)

        ax[0, 0].set_xticks(np.arange(0, self.num_cols, 1.0))
        ax[0, 0].set_ylim(-0.5, self.num_rows - 0.5)
        ax[0, 0].set_yticks(np.arange(0, self.num_rows, 1.0))
        ax[0, 0].set_ylim(ax[0, 0].get_ylim()[::-1])  # Invert y-axis

        U = np.zeros(len(states))
        V = np.zeros(len(states))

        for state in states:
            row, col = self.val_to_coords(state)
            transition_tuple = self.P[patient][state][actions[state]]

            reward = transition_tuple[0][2]

            forward_dict[state].append(transition_tuple[0][1])

            if self.no_nop and state in self.goals[patient]:
                U[state] = 0
                V[state] = 0
                reward = 1.0
            else:
                U[state] = u_dict.get(actions[state])
                V[state] = v_dict.get(actions[state])

            if reward > 0:
                correctness += 1

        x = np.tile(np.arange(self.num_cols), self.num_rows)  # 0,1,2,3,4,0,1,2,3,4,0...
        y = np.repeat(np.arange(self.num_rows), self.num_cols)  # 0,0,0,0,0,1,1,1,1,1,2...

        ax[0, 0].set_title("Patient {} | Policy correctness {}".format(patient, correctness / self.num_states))

        for goal in self.goals[patient]:
            row, col = self.val_to_coords(goal)
            ax[0, 0].scatter(col, row, marker='s', c="green", s=200)
            if reach_plot:
                ax[1, 0].scatter(col, row, marker='s', c="red", s=50)
        ax[0, 0].scatter(x, y)
        ax[0, 0].quiver(x, y, U, V, scale=35)

        # REACHABILITY GRAPH

        if reach_plot:
            reachability_graph = self.reachability_plot(patient, forward_dict)
            reachability_graph = reachability_graph * 1
            reachability_graph = reachability_graph.astype(int)
            reachability_graph = reachability_graph.reshape(self.num_rows, self.num_cols, order='C')
            ax[1, 0].set_title("Reached goal from:")
            ax[1, 0].matshow(reachability_graph, cmap='Greens')
            ax[1, 0].xaxis.set_ticks_position('bottom')

        # STATE VALUE PLOTS

        elif state_val_plot:
            ax[1, 0].set_title("State value estimates")
            state_vals = np.array(state_vals)
            state_vals = state_vals.reshape(self.num_rows, self.num_cols, order='C')
            im = ax[1, 0].matshow(state_vals, cmap="Blues")
            ax[1, 0].xaxis.set_ticks_position('bottom')
            divider = make_axes_locatable(ax[1, 0])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax)

        # NOP LIKELIHOOD PLOT
        elif q_val_plots:
            min_val = np.min(q_values)
            max_val = np.max(q_values)
            for i in range(self.num_actions):
                plot_q_vals = np.array(q_values[:, i])
                ax[i + 1, 0].set_title("NOP q-values - max value: {}".format(np.max(plot_q_vals)))
                heat_map = plot_q_vals
                heat_map = heat_map.reshape(self.num_rows, self.num_cols, order='C')
                ax[i + 1, 0].set_title("Action {}".format(i))
                im = ax[i + 1, 0].matshow(heat_map, cmap='Reds', vmin=min_val, vmax=max_val)
                ax[i + 1, 0].xaxis.set_ticks_position('bottom')
                divider = make_axes_locatable(ax[i + 1, 0])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                fig.colorbar(im, cax=cax)

        if reach_plot:
            return fig, correctness / self.num_states, reachability_graph
        else:
            return fig, correctness / self.num_states

    def reachability_plot(self, patient, forward_dict):
        backward_dict = {state: [] for state in range(self.num_states)}
        reach_map = np.zeros([self.num_states])
        queue = [goal for goal in self.goals[patient]]
        # queue = [goal for goal in self.get_vicinity(self.goals[patient])]

        for k, v in forward_dict.items():
            if v:
                backward_dict[v[0]].append(k)

        while queue:
            # print(queue)
            state = queue.pop()
            reach_map[state] = 1
            if backward_dict[state]:
                [queue.append(elem) for elem in backward_dict[state] if elem not in queue]
                backward_dict[state] = []

        return reach_map

    def test_model(self, states, actions, test_patient):
        correctness = 0
        for state in states:
            transition_tuple = self.P[test_patient][state][actions[state]]

            if self.no_nop and state in self.goals[test_patient]:
                reward = 1.0
            else:
                reward = transition_tuple[0][2]

            if reward > 0:
                correctness += 1

        return correctness / self.num_states

    def check_boundaries(self, new_row, new_col):
        return True if new_row in range(self.num_rows) and new_col in range(self.num_cols) else False

    def get_vicinity(self, goals):
        neighbours = []
        for goal in goals:
            goal_x, goal_y = self.val_to_coords(goal)
            for x in range(goal_x - 1, goal_x + 2):
                for y in range(goal_y - 1, goal_y + 2):
                    state = self.coords_to_val(x, y)
                    if self.check_boundaries(x, y) and state not in neighbours:
                        neighbours.append(state)
        return neighbours

    def reset_params(self):
        self.counter = 0
        self.correct_counter = 0

    def reset_patient(self):
        if self.training == True:
            new_patient = np.random.choice(self.train_patient_idxs)
        else:
            new_patient = np.random.choice(self.test_patient_idxs)
        return new_patient

    def reset_state(self):
        new_state = np.random.randint(self.num_states)
        return new_state

    def log_final_state(self, is_success=None):
        info_dict = {
            'is_success': is_success,
            'correct_decision_rate': self.correct_counter / self.counter * 100
        }
        return info_dict

    def val_to_coords(self, value):
        return value // self.num_cols, value % self.num_cols

    def coords_to_val(self, row, col):
        return row * self.num_cols + col

    def get_row(self, value):
        return value // self.num_cols

    def get_col(self, value):
        return value % self.num_cols

    def distance_to_goal(self, pat, state):
        goals = self.goals[pat]
        dist_to_goals = [self.distance_to_state(state, goal) for goal in goals]
        return np.min(np.array(dist_to_goals))

    def distance_to_state(self, state_1, state_2):
        row_1 = self.get_row(state_1)
        col_1 = self.get_col(state_1)
        row_2 = self.get_row(state_2)
        col_2 = self.get_col(state_2)
        if self.chebishev:
            return np.max((np.abs(row_1 - row_2), np.abs(col_1 - col_2)))
        else:
            return np.abs(row_1 - row_2) + np.abs(col_1 - col_2)

    def is_closer(self, prev_state, curr_state, goal_avg):
        prev_row = self.get_row(prev_state)
        prev_col = self.get_col(prev_state)
        curr_row = self.get_row(curr_state)
        curr_col = self.get_col(curr_state)
        goal_row = self.get_row(goal_avg)
        goal_col = goal_avg - goal_row * self.num_cols
        if self.chebishev:
            prev_dist = np.max((np.abs(prev_row - goal_row), np.abs(prev_col - goal_col)))
            curr_dist = np.max((np.abs(curr_row - goal_row), np.abs(curr_col - goal_col)))
        else:
            prev_dist = np.abs(prev_row - goal_row) + np.abs(prev_col - goal_col)
            curr_dist = np.abs(curr_row - goal_row) + np.abs(curr_col - goal_col)

        return True if curr_dist < prev_dist else False

    def load_patient(self, file_path, patient_name):
        with open(file_path + 'Patient_files/{}'.format(patient_name)) as file:
            self.name = file.readline().strip()
            print(self.name)
            _ = file.readline()
            frame_path = file_path + "Sacrum_{}/sacrum_sweep_frames/".format(self.name)
            goal_coords = file.readline().strip().replace(";", ":").split(":")[1:]

            goals = []
            for i in range(len(goal_coords)):
                row, col = map(int, goal_coords[i].split(","))
                goals.append(row * self.num_cols + col)

            frame_array = np.zeros((self.num_states, self.frames_per_state, self.resize_x, self.resize_y))
            for _ in range(self.num_states):
                coords, frames = file.readline().strip().split(":")
                row, col = map(int, coords.split(","))
                frames = list(map(int, frames.split(",")))
                for i in range(self.frames_per_state):
                    frame = self.load_frame(frame_path, frames[i])
                    frame = self.process_frame(frame)
                    frame_array[row * self.num_cols + col, i, :, :] = frame
        return goals, frame_array

    def load_frame(self, frame_path, idx):
        frame = Image.open(frame_path + "sacrum_translation{:04d}.png".format(idx))
        frame = np.array(frame)
        return frame

    def process_frame(self, frame):
        frame = resize(frame, (self.resize_x, self.resize_y))
        frame = (frame - np.min(frame)) / np.ptp(frame)
        return frame

