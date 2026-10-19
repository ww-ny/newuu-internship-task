# Part B: G1 walking in a circle

## What I changed
I ran the pretrained G1 walking policy from unitree_rl_gym using the MuJoCo
deployment. The robot walks forward by default.

I changed one line in g1.yaml:

    cmd_init: [0.5, 0, 0]  ->  cmd_init: [0.5, 0, 0.8]

The command has three numbers: forward speed, sideways speed and turning speed.
I kept the robot walking forward and added a turning speed.

The code reads this command once and gives the same command to the policy all the
time. If a robot keeps moving forward and keeps turning at the same rate, its path
is a circle.

## 1. What inputs does the policy receive, and where do they come from?
The policy gets a list of 47 numbers, 50 times per second. In the code this list is
called obs. It has:
- angular velocity of the robot's body (3 numbers), from MuJoCo's d.qvel
- gravity direction (3), calculated from the body's orientation in d.qpos, which
  tells the policy how much the body is tilted
- the command (3), from cmd_init in g1.yaml
- joint positions (12), from d.qpos, compared to the default standing pose
- joint speeds (12), from d.qvel
- the previous action (12), saved from the last time the policy ran
- a clock (2 numbers, sin and cos), calculated from the simulation time, which
  repeats every 0.8 seconds and helps the legs step in rhythm

In the simulation, MuJoCo gives the body and joint values. On a real robot they
would come from its sensors. The command comes from the config file.

## 2. What does the policy output, and how is that converted into joint torques?
The policy outputs 12 numbers, one for each leg joint. The code does:

    target_dof_pos = action * action_scale + default_angles

So the output is a small change from the default standing pose, and it gives the
target angle for each joint. Then a PD controller turns the target into a torque
on every physics step:

    torque = (target angle - current angle) * kp + (0 - current speed) * kd

kp pulls the joint toward the target like a spring, and kd slows it down so it
does not swing too much. The kp and kd numbers come from g1.yaml. The torques are
sent to MuJoCo (d.ctrl) and the physics moves one step. The policy itself runs
every 10 physics steps, which is 50 times per second.
