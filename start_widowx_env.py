"""Starts a real WidowX playground environment that you can step through with commanded actions."""
import imageio
import numpy as np
import os
import sys
import time
from collections import defaultdict
sys.path.append("/iris/u/moojink/prismatic-dev/experiments/robot/")  # hack so that the interpreter can find widowx_real_env
from widowx_real_env import JaxRLWidowXEnv

# Initialize important constants and pretty-printing mode in NumPy.
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
np.set_printoptions(formatter={"float": lambda x: "{0:0.2f}".format(x)})


def get_env():
    """Get WidowX control environment."""

    class AttrDict(defaultdict):
        __slots__ = ()
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__

    variant = AttrDict(lambda: False)
    env_params = {
        "fix_zangle": True,  # do not apply random rotations to start state
        "move_duration": 0.2,
        "adaptive_wait": True,
        "move_to_rand_start_freq": 1,
        "override_workspace_boundaries": [[0.1, -0.25, 0.095, -1.57, 0], [0.4, 0.25, 0.4, 1.57, 0]],
        "action_clipping": "xyz",
        "catch_environment_except": True,
        "add_states": variant.add_states,
        "from_states": variant.from_states,
        "reward_type": variant.reward_type,
        "start_transform": None,
        "randomize_initpos": "full_area",
    }
    env = JaxRLWidowXEnv(env_params)
    return env


def save_rollout_gif(rollout_images, base_dir, idx):
    """Saves a GIF of an episode."""
    gif_path = os.path.join(base_dir, f"rollout-{DATE_TIME}-{idx}.gif")
    imageio.mimsave(gif_path, rollout_images, loop=0)
    print(f"Saved rollout GIF at path {gif_path}")


def user_input_to_action(user_input):
    if user_input == 'w':  # forward
        action = np.array([0.01, 0., 0., 0., 0., 0., 1.])
    elif user_input == 's':  # backward
        action = np.array([-0.01, 0., 0., 0., 0., 0., 1.])
    elif user_input == 'a':  # left
        action = np.array([0., 0.01, 0., 0., 0., 0., 1.])
    elif user_input == 'd':  # right
        action = np.array([0., -0.01, 0., 0., 0., 0., 1.])
    elif user_input == 'e':  # up
        action = np.array([0., 0., 0.01, 0., 0., 0., 1.])
    elif user_input == 'q':  # down
        action = np.array([0., 0., -0.01, 0., 0., 0., 1.])
    elif user_input == 'r':  # close gripper
        action = np.array([0., 0., 0., 0., 0., 0., 0.])
    else:  # open gripper
        action = np.array([0., 0., 0., 0., 0., 0., 1.])
    return action


def main():
    # Initialize the WidowX environment.
    env = get_env()
    MAX_STEPS = 50
    # Start episodes.
    episode_idx = 0
    while True:
        rollout_images = []
        env.reset()
        env.start()
        t = 0
        step_duration = 0.2  # divide 1 by this to get control frequency
        input(f"Press Enter to start episode {episode_idx+1}...")
        last_tstamp = time.time()
        while t < MAX_STEPS:
            try:
                # Get environment observations.
                obs = env._get_obs()
                img = obs["pixels"][0]
                rollout_images.append(img)
                if time.time() > last_tstamp + step_duration:
                    print(f"t: {t}")
                    last_tstamp = time.time()
                    # Get action from user.
                    user_input = input("Enter a command for the robot ('w': forward, 's': backward, 'a': left, 'd': right, 'e': up, 'q': down, 'r': close gripper, 'f': open gripper): ")
                    action = user_input_to_action(user_input)
                    # Execute action in environment.
                    tstamp_return_obs = last_tstamp + step_duration
                    print(f"Action: {action}")
                    _, _, _, _ = env.step({"action": action, "tstamp_return_obs": tstamp_return_obs})
                    t += 1
            except Exception as e:
                print(f"Caught exception: {e}")
                break
        user_input = input("If you want to save a rollout GIF of this episode, enter the directory to " \
                           "save the GIF in (example: /iris/u/moojink/). If not, just press Enter: ")
        if user_input != "":
            save_rollout_gif(rollout_images, base_dir=user_input, idx=episode_idx)
        episode_idx += 1
        input("Press Enter to continue to the next episode, or Ctrl-C to exit...")


if __name__ == "__main__":
    main()
