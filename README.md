# Running WidowX with Controller from Bridge Dataset

## Launching Playground Environment

Copy Moo Jin's installed ROS and InterbotiX packages to your own directory:
```bash
rsync -a /iris/u/moojink/interbotix_ws <DESTINATION> -v --progress
rsync -a /iris/u/moojink/catkin_ws <DESTINATION> -v --progress
```

Clone the `widowx_control` repo for the Bridge WidowX control stack:
```bash
git clone https://github.com/moojink/widowx_control.git
```

Create a bash source file (.widowx_profile) with the following contents, replacing all `/PATH/TO/YOUR` with your own base directories:
```bash
# For ROS Noetic environment variables
source /opt/ros/noetic/setup.bash
source /opt/ros/noetic/setup.sh

# For WidowX control
source /PATH/TO/YOUR/catkin_ws/devel/setup.sh
source /PATH/TO/YOUR/interbotix_ws/devel/setup.sh
export ROBONETV2_ARM=wx250s
export PYTHONPATH="${PYTHONPATH}:/PATH/TO/YOUR/widowx_control/widowx_envs"
export PYTHONPATH="${PYTHONPATH}:/PATH/TO/YOUR/interbotix_ws/src/interbotix_ros_toolboxes/interbotix_xs_toolbox"
export PYTHONPATH="/PATH/TO/YOUR/catkin_ws/devel/lib/python3/dist-packages:${PYTHONPATH}"
```

On `iris-ws-18`, run the following commands to launch the WidowX controller (built on ROS):

```
cd widowx_control
source .widowx_profile
sudo v4l2-ctl --list-devices
./scripts/run.sh -v 0 # *** CHANGE THE CAMERA INDEX BASED ON ABOVE OUTPUT ***
```

NOTE: The second command above will print a list of the cameras that the workstation has access to, like `/dev/video0` and `/dev/video1`. If you don't see any cameras, they might not be connected to the workstation. Let's say you want to use the camera labeled as `/dev/video1`. Then, run the command below to launch the WidowX control stack (I like to do this in a tmux session). The `-v 1` arg tells the launcher script that you want the camera at index `1` to be mapped to `camera0` in the robot infrastructure code. If you want to use `/dev/video0` instead of `/dev/video1`, then you would use the argument `-v 0`.

Now, in another Terminal, run the commands below. Modify args as needed.

```
# Setup
cd widowx_control
source .widowx_profile
source /scr/moojink/miniconda3/bin/activate base
conda activate /scr/moojink/miniconda3/envs/prisma

# Start playground script
python start_widowx_env.py
```

This will boot up a playground WidowX environment. You can step through the environment and see the actions being executed on the real robot.

## Launching Evaluations

The commands below show you an example of how to run real-world Bridge evals on `iris-ws-18`:

```
# Setup
source /scr/moojink/miniconda3/bin/activate base
conda activate /scr/moojink/miniconda3/envs/prisma
cd /iris/u/moojink/prismatic-dev

# Start evals
python experiments/robot/eval_vla_on_bridge_env.py \
    --model.type siglip-224px+7b \
    --pretrained_checkpoint /scr/moojink/checkpoints/tri/lr-2e5+siglip-224px+mx-bridge+n1+b32+x7/checkpoints/step-080000-epoch-09-loss=0.0987.pt \
    --data_stats_path ./experiments/dataset_statistics/bridge_stats.json
```
