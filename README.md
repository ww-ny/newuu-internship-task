## How did you determine what the columns in the CSV represent?

The CSV did not have any headers, so I first had to find out what the columns mean.

Finding the format. The file has 36 columns and was exported as "LAFAN" format
for the Unitree G1 at 30 fps. I searched online, with help from an AI assistant
, and found Unitree's LAFAN1 retargeting dataset documentation. It describes
each G1 row as: root position (x, y, z), root orientation quaternion (x, y, z, w),
then the joint angles in a fixed joint order. 3 + 4 + 29 joints = 36, which matches.
So columns 0-2 are the root position, 3-6 the quaternion, and 7-35 the 29 joint
angles.

To check it I wrote an inspect mode in my script (`--inspect`) and ran it on my CSV:
- The quaternion norm was 1.0000 in every frame, so columns 3-6 behave like a rotation.
- The root height was 0.76-0.79 m, a sensible standing pelvis height, so columns 0-2 are positions in metres.
- Mapping the 29 columns to the joints of the G1 29-DoF model by name, every joint
  stayed inside its joint limits (0 joints outside limits). This also shows the
  values are in radians, not degrees.
I used an AI assistant for help understanding the task and writing the script
