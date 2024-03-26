# Running WidowX with Controller from Bridge Dataset

## Launching Playground Environment

On `iris-ws-18`, run the following commands to launch the WidowX controller (built on ROS):

```
source /iris/u/moojink/.openvla_widowx_profile
sudo v4l2-ctl --list-devices
/iris/u/moojink/widowx_control/scripts/run.sh -v 0 # *** CHANGE THE CAMERA INDEX BASED ON ABOVE OUTPUT ***
```

NOTE: The second command above will print a list of the cameras that the workstation has access to, like `/dev/video0` and `/dev/video1`. If you don't see any cameras, they might not be connected to the workstation. Let's say you want to use the camera labeled as `/dev/video1`. Then, run the command below to launch the WidowX control stack (I like to do this in a tmux session). The `-v 1` arg tells the launcher script that you want the camera at index `1` to be mapped to `camera0` in the robot infrastructure code. If you want to use `/dev/video0` instead of `/dev/video1`, then you would use the argument `-v 0`.

Now, in another Terminal, run the commands below. Modify args as needed.

```
# Setup
source /iris/u/moojink/.openvla_widowx_profile
source /scr/moojink/miniconda3/bin/activate base
conda activate /scr/moojink/miniconda3/envs/prisma
cd /iris/u/moojink/widowx_control

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