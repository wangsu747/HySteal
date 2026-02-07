import argparse
import os
from pettingzoo.butterfly import pistonball_v3
from pettingzoo.utils import aec_to_parallel
from pettingzoo.mpe import simple_tag_v3
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd


from MADDPG import MADDPG
from main import get_env


def get_csv_dimensions(file_path):
    df = pd.read_csv(file_path,header=None)

    num_rows = df.shape[0]
    num_columns = df.shape[1]

    return num_rows, num_columns

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str, default='simple_adversary_v3', help='name of the env',
                        choices=['simple_adversary_v3', 'simple_spread_v3', 'simple_tag_v3'])
    parser.add_argument('folder', type=str, help='name of the folder where model is saved')
    parser.add_argument('--episode_num', type=int, default=1000, help='total episode num during evaluation')
    parser.add_argument('--episode_length', type=int, default=30, help='steps per episode')

    args = parser.parse_args()

    model_dir = os.path.join('./results', args.env_name, args.folder)
    assert os.path.exists(model_dir)
    gif_dir = os.path.join(model_dir, 'gif_new')
    if not os.path.exists(gif_dir):
        os.makedirs(gif_dir)
    gif_num = len([file for file in os.listdir(gif_dir)])

    env, dim_info = get_env(args.env_name, args.episode_length, render_mode="rgb_array")

    maddpg = MADDPG.load(dim_info, os.path.join(model_dir, 'model.pt'))

    agent_num = env.num_agents
    episode_rewards = {agent: np.zeros(args.episode_num) for agent in env.agents}

    data = {
        'states': [],
        'actions': [],
        'next_states': []
    }
    file_path=''

    for episode in range(args.episode_num):
        states = env.reset()
        if isinstance(states, tuple):
            states = states[0]


        agent_reward = {agent: 0 for agent in env.agents}
        frame_list = []
        step_count = 0
        episode_data = {
            'states': [],
            'actions': [],
            'next_states': []
        }
        i=1

        while env.agents:


            actions = maddpg.select_action(states)
            next_states, rewards, dones, truncations, infos = env.step(actions)



            step_count += 1
            episode_data['states'].append(states)
            episode_data['actions'].append(actions)

            episode_data['next_states'].append(next_states)

            states = next_states

            for agent_id, reward in rewards.items():
                agent_reward[agent_id] += reward



        data['states'].extend(episode_data['states'])
        data['actions'].extend(episode_data['actions'])
        data['next_states'].extend(episode_data['next_states'])
        message = f'Episode {episode + 1}, '
        for agent_id, reward in agent_reward.items():
            message += f'{agent_id}: {reward:.4f}; '
        print(message)
    actions_array = []
    states_array = []
    next_states_array = []
    landmark = []

    i = 1
    for states_dict in data['states']:
        row = []
        for agent in ['adversary_0', 'adversary_1', 'adversary_2', 'agent_0']:
            if agent in states_dict:
                flattened_state = states_dict[agent].flatten()

                padding_size = 16 - len(flattened_state)
                if padding_size > 0:
                    flattened_state = np.append(flattened_state, [0] * padding_size)


                row.extend(flattened_state)
            else:
                row.extend([np.nan] * 16)
        states_array.append(row)
        if i == 1 or i ==2:
            i +=1
    states_np_array = np.array(states_array)

    for action_dict in data['actions']:
        row = []
        for agent in ['adversary_0', 'adversary_1', 'adversary_2', 'agent_0']:
            if agent in action_dict:
                row.append(action_dict[agent])
            else:
                row.append(np.nan)
        actions_array.append(row)
    actions_np_array = np.array(actions_array)

    for next_states_dict in data['next_states']:
        row = []
        landmark_row = []
        for agent in ['adversary_0', 'adversary_1', 'adversary_2', 'agent_0']:
            if agent in next_states_dict:
                flattened_state = next_states_dict[agent].flatten()
                padding_size = 16 - len(flattened_state)
                if padding_size > 0:
                    flattened_state = np.append(flattened_state, [0] * padding_size)
                row.extend(flattened_state)
            else:
                row.extend([np.nan] * 16)

        for agent in ['adversary_0']:
            if agent in next_states_dict:
                flattened_state = next_states_dict[agent].flatten()
                agent_pos = flattened_state[2:4]

                for i in range(2):
                    landmark_rel_position = flattened_state[4 + 2 * i: 6 + 2 * i]
                    landmark_abs_position = agent_pos + landmark_rel_position
                    landmark_row.extend(landmark_abs_position)
                    print('agent_0{}landmark = {}'.format(i,landmark_abs_position))

        landmark.append(landmark_row)

        next_states_array.append(row)
    next_states_np_array = np.array(next_states_array)
    landmark_np_array = np.array(landmark)



    print("Shape of actions_np_array:", actions_np_array.shape)
    print("Shape of states_np_array:", states_np_array.shape)
    print("Shape of next_states_np_array:", next_states_np_array.shape)
    print('Shape of landmark_np_array : ',landmark_np_array.shape)
    states_np_array = np.concatenate(
        (states_np_array[:, 0:4], states_np_array[:, 16:20], states_np_array[:, 32:36], states_np_array[:, 48:52]),
        axis=1)
    next_states_np_array = np.concatenate(
        (next_states_np_array[:, 0:4], next_states_np_array[:, 16:20], next_states_np_array[:, 32:36], next_states_np_array[:, 48:52]),
        axis=1)
    combined_array_ = np.concatenate((states_np_array, actions_np_array, next_states_np_array, landmark_np_array), axis=1)
    print("Shape of combined_array:", combined_array_.shape)
    combined_array = combined_array_[::-1]




    print('combined_array  = {}'.format(combined_array.shape))
    base_directory = "/path/of"
    directory = os.path.join(base_directory, args.folder)
    file_name = "simple_tag_v3_1000_30_reverse_xy.csv"
    file_path = os.path.join(directory, file_name)

    os.makedirs(directory, exist_ok=True)

    df = pd.DataFrame(combined_array)
    df.to_csv(file_path, index=False, header=False)

    saved_df = pd.read_csv(file_path, header=None)
    print(f"Number of rows in saved CSV: {saved_df.shape[0]}")

    print(f"Array has been saved to {file_path}")
    rows, columns = get_csv_dimensions(file_path)
    print(f'The CSV file has {rows} rows and {columns} columns.')

    message = f'Episode {episode + 1}, '
    for agent_id, reward in agent_reward.items():
        message += f'{agent_id}: {reward:.4f}; '
    print(message)

    gif_path = os.path.join(gif_dir, f'episode_{episode + 1}.gif')
    frame_list[0].save(gif_path, save_all=True, append_images=frame_list[1:], duration=1, loop=0)

    env.close()

    fig, ax = plt.subplots()
    x = range(1, args.episode_num + 1)
    for agent_id, rewards in episode_rewards.items():
        ax.plot(x, rewards, label=agent_id)
    ax.legend()
    ax.set_xlabel('episode')
    ax.set_ylabel('reward')
    total_files = len([file for file in os.listdir(model_dir)])
    title = f'evaluate result of maddpg solve {args.env_name} {total_files - 3}'
    ax.set_title(title)
    plt.savefig(os.path.join(model_dir, title))

