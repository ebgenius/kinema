# Kinema

**Animation-ready robot rigs in Blender.** Import a robot description and get a single clean
armature you can actually animate — with IK that understands singularities, joint limits and
multi-turn joints. A built-in catalogue of 186 real robots says where to find one. A
description's meshes can be COLLADA, STL, OBJ, PLY or glTF (`.glb`, `.gltf`).

> Status: released as [v0.6.0]. Import (URDF, xacro or MJCF), rig, pose, solve and
> bake all function, and the built extension installs and runs from a clean Blender
> profile. Nothing is downloaded, no threads are started, and no environment
> variables are written — an import blocks Blender while it works.
>
> Docs: <https://ebgenius.github.io/kinema/>

[v0.6.0]: https://github.com/ebgenius/kinema/releases/tag/v0.6.0

## Why this exists

Every Blender robotics tool today — [Phobos], [LinkForge], [urdf_importer], [blender-urdf] —
treats Blender as a *design and export* station for URDF/SDF. They optimise for producing
simulation assets.

Kinema optimises for the opposite thing: **open a .blend, find a clean rig, animate it,
hit render.** Blender is the destination, not a stop along the way.

Concretely, that means:

- **One armature.** Not armatures nested inside armatures.
- **One 1-DoF bone per joint**, with the bone's local Y aligned to the URDF joint axis, so
  every joint is a true single-axis control with real limits.
- **Bone collections** (`Kinema/FK`, `Kinema/IK`, `Kinema/TCP`, `Kinema/Mechanism`) so an
  animator sees controls, not machinery.
- **IK that behaves like a normal Blender IK control** — keyframe the TCP and move on —
  but is solved by [PyRoki] rather than Blender's built-in solver.
- **A bake step**, so the finished .blend renders anywhere, with or without Kinema.

[Phobos]: https://github.com/dfki-ric/phobos
[LinkForge]: https://extensions.blender.org/add-ons/linkforge/
[urdf_importer]: https://github.com/HoangGiang93/urdf_importer
[blender-urdf]: https://github.com/kralf/blender-urdf
[PyRoki]: https://github.com/chungmin99/pyroki

## The Kinema tab

Everything is in the 3D viewport's sidebar (N), on the **Kinema** tab:

- **The quick bar**, first, names the robot shown in its header and holds the buttons used
  all the time: **Live IK** (or **Add IK Target**, where there is none); **Configurations**
  (Find, then stepping through the arm's other ways to the pose) beside **Auto Config**;
  **Rest Pose** beside **Snap to Tool**, which is greyed out while the IK target is on the
  tool and pressable once it is left off it; and **Motion Check** beside **Graph Editor**. Which buttons, and their order, is yours to choose in the add-on's preferences,
  under **Quick Bar**: move them up or down, remove them, add others (Key All, Record
  Waypoint, Generate, Optimize, Bake IK), or Reset. Two sit to a row; with an odd number,
  the first spans the bar. With no robot, the quick bar offers to import one.
- **Joints (FK)**, **Inverse Kinematics**, **Waypoints**, **Motions** and **Motion Check**,
  the panels used most, each on its own: drag one by its grip to put it elsewhere, collapse
  it, or Shift-click its header to pin it on every tab, as Blender's own sidebar panels go.
  Rest Pose, Graph Editor and Key All are also icons in the Joints header.
- **Robot Settings**, last, holds what is set once, as subpanels: **Import** (with Find a
  Robot and Import Options), **Bones**, **Tool Centre Point**, **External Axes**, **Motion
  Limits**, **Collision** and **Solver**. Its header names the robot.

## Dressing a rig, and where IK aims

The **Bones** list, under Robot Settings, is one row per bone, and it answers the two questions that come
up as soon as a robot is imported.

**Hang something off a bone.** Pick an object or a collection in a row and a linked copy
rides that bone — a gripper on the flange, a cable harness down the forearm. It shares its
mesh and materials with what you picked, so the same harness can dress six links and still
be edited in one place, and the source can live in any scene in the file. What arrives is a
placement and nothing more: the copy is stripped of the source's animation and constraints,
so nothing reaches in from outside to move it.

The offset is measured from the bone's **head**. Blender's own bone parenting measures from
the tail, which puts a fresh attachment at the far end of the bone and makes every offset
you then dial in relative to a frame that has nothing to do with the joint. Kinema cancels
that, so the attachment's plain location/rotation/scale *is* its offset from the joint —
edit it in the panel, or grab the object and use G/R/S, and keyframe it like anything else.

Clearing a row's picker removes the copy; the **✕** button instead unparents it and leaves
it in the scene exactly where it appears, for when you want to keep what you placed.

To resize an attachment, scale **the attachment** — its scale is in the *Offset from the
Bone* block with the rest of its transform. Scaling the source object does not carry over,
because the copy shares the source's *geometry* but keeps its own transform, and that
transform is what positions it on the bone. (Applying scale on the source will not work
either: Blender refuses `Ctrl+A` on a mesh with more than one user, which is exactly what a
linked copy makes it.)

**Choose what IK aims at.** The radio button in each row points the solver at that bone, the
tool centre point included — that one is the default. This matters on redundant robots: a
Panda imports with its tool frame on a fingertip, which puts gripper finger joints inside the
chain. They are slides, so **Add IK Target** holds them the way it holds a rail (see *Spare
joints* below), but the arm is still solving to a finger rather than to its hand. Aim at the
flange instead and the chain is the seven arm joints it should be. Switching snaps the IK
control onto the new target, so the arm does not jump.

The choice is keyframable, so a shot can hand the goal from the wrist to the elbow part-way
through. **Key Target Bone**, beside the field in the IK panel, is the way to key it: the
target is an index rather than a quantity, and those keys have to *step* between values.
Interpolated, a hand-off from the tool point to joint 3 would pass through joints 1 and 2 on
the frames in between and solve two chains nobody asked for. Baking follows a keyed target
across the range and keys every joint that is active anywhere in it.

Switching to a bone whose link the solver has not seen before pays a one-off compile, the
same wait as the first solve after adding an IK target. A handful of recently used ones are
kept compiled, so scrubbing back and forth over a hand-off is free after the first pass.

## Teaching a job

Rigging a robot answers *how do I get one into Blender*. The **Waypoints** panel answers
what it should do.

Pose the robot, press **Record**, and give the row a name — `home`, `approach`, `pick`,
`drop`. Each waypoint stores the tool pose *and* the joint vector it was taught in, because
a pose alone does not say which of a robot's up-to-eight solutions you meant. **Go To**
restores that exact configuration rather than re-solving and landing somewhere else.

**Record** also leaves an arrows Empty at the tool, the waypoint's marker, and the marker
*is* the waypoint. Drag it, or snap it to a feature on the part, and the next **Generate
Motion** re-aims every move to and from it, joint moves included:
- the joints are re-solved at the marker, walked there from the ones taught so they keep
  the same configuration;
- they replace the ones taught, as **Update** would after posing the robot there, and
  Generate Motion names each waypoint it re-solved;
- a marker the robot can't reach from its taught configuration is refused, with its name.

Baked keys are never touched.

A waypoint holds a pose and the joints taught there, and no time. When the robot moves, and
how, is a **motion**, in the **Motions** list under the waypoints: a move from one waypoint
to another, with its own **Start** and **End** frame, its length in frames shown beside
them. So one waypoint can end one motion and start the next, a job can go back to `home`
without teaching it twice, and a motion is retimed without touching the poses.
- **Record** on a later frame than the last waypoint recorded also adds a joint motion from
  that waypoint to the new one, ending on this frame. Teaching a job pose by pose, frame by
  frame, is what it was. **Add** puts a motion after the last one, a second long.
- The job runs in frame order, as strips in the NLA do, not in list order. A gap where the
  next motion starts at the waypoint the last one ended on holds the robot there. A gap to
  another waypoint gets a joint move across it, and Generate Motion says so: Optimize
  Motion keeps it clear, and its speed is yours to check.
- Motions that overlap are refused, with a warning naming both, and nothing is generated
  until one is moved ([#133](https://github.com/ebgenius/kinema/issues/133)). So is a
  motion that starts at another waypoint on the frame the last one ends.
- Removing a waypoint removes the motions to and from it. **Go To** jumps to the first
  frame the job is at the waypoint.
- A file saved before motions opens with them: each pair of waypoints in frame order
  becomes a motion with their frames and move, and the job plays as it did.

A waypoint can be taught **relative to an object**, a workpiece or a fixture, set in its
**Relative To** field under the list. Its marker is then parented to the object without a
parent inverse, as Blender's *Make Parent without Inverse* (Shift+Ctrl+P) parents it, so
the marker's own Location and Rotation are the pose in the object's space. Parenting a
marker with that operator does the same, and Kinema reads the marker's parent.
- Changing the field keeps the waypoint where it is in the scene. Empty means the robot's
  base, as before.
- **Record** goes on relative to what the highlighted waypoint is, so a job taught on a
  workpiece stays on it.
- Move or animate the object and the job follows it. Wherever the job meets a waypoint
  away from where it was taught, the joints are walked there from the taught ones, keeping
  their configuration. Unlike a marker dragged at the base, these don't replace the taught
  joints, since the object can be somewhere else on every visit: **Update** re-teaches it.
- Delete the object and its waypoints go back where they were taught, relative to the
  robot, and Generate Motion says so. A marker parented to something the robot carries is
  refused, and so is one parented to a bone or vertices of another object: parent it to
  the object, or to an Empty on the bone.
- On an object that moves while the robot is on it, a linear move keeps to its line where
  the object has it on every frame, and a hold stays on the object, as on a conveyor. Both
  need the IK target. A joint move ends where the object is, its path free between.
- The Motion Check and **Optimize Motion** measure and aim for the moving line. The panel
  marks the check outdated when the object or the robot is moved, or the object re-keyed,
  re-eased or retimed in the NLA, or a bone of another armature it hangs from is, but not
  when you scrub.
  What moves it from elsewhere, a constraint or a driver, isn't seen: **Check Again**
  catches that.

(The keys **Generate Motion** writes are the *output*. Dragging those in the dope sheet
retimes the animation, as it would for anything else, but the next regeneration writes the
motion's own frames again, so the Motions list is where a retime belongs if you want it to
stick.)

**The job on the NLA.** Generate Motion lays the job out on the robot's NLA: one strip per
motion, named after it and covering its frames, on two *Kinema Motions* tracks that take
turns. Under them, the *Kinema Job* track has a strip for each pause and each joint move
bridging a gap, and *Before* and *After* strips holding whatever is animated outside the
job, hand keys included. All the strips show the one action that holds the job's keys, each
its own frames, so the robot plays exactly as it would from that action alone.
- **Retime a motion in the NLA editor** by moving or scaling its strip. With the motions on
  tracks of their own, a strip has room to move. The next Generate takes its new frames into
  the Motions list and lays the job out at them, and a Motion Check takes them in too; the
  report says so. Until then the stretch a strip left holds the pose before it, and Optimize
  Motion and Bake IK ask for that Generate first. An operator that refuses changes nothing in
  the Motions list. A strip duplicated in the NLA editor retimes nothing, since which copy
  is the motion's can't be told.
- Changing a motion's frames in the Motions list moves its strip on the next Generate. If a
  motion was retimed in both places, the Motions list's frames win, and the report says so.
- A strip for a pause or a bridging joint move follows the motions around it, so moving it
  retimes nothing, and the next Generate puts it back.
- Nothing is left in the robot's active action. Keys you set by hand afterwards land in a
  new active action, on top of the job, and the next Generate moves them into the job's
  action.
- Regenerating replaces the job's strips. In NLA tweak mode, Generate writes into the strip
  being tweaked, as it always has, and leaves the job's strips as they are.
- The strips replace what is under them at full strength, whatever the blend and influence
  set for the active action.
- A track of your own named like Kinema's is left alone: Kinema knows its tracks by what
  their strips show, and a motion's strip by the frames it shows, so renaming one is fine.
- To keep the job in the active action instead, turn off **Job as NLA Strips** in the
  add-on's preferences. The next Generate puts a job already on the NLA back there.

Each motion says how the robot moves:

- **Joint** interpolates the joints. Fast, always reachable, and the tool takes whatever
  path falls out. What you want for crossing a workspace.
- **Linear** drives the tool along the straight line between the two poses. What a process
  move needs. On a KR120 the tool holds that line to 0.002 mm over a 300 mm plunge.

**Generate Motion** writes it out as ordinary keyframes, and keys the live-IK switch so each
span is solved the way it should be:
- a joint move keys the joint channels;
- a linear move keys the IK goal, plus the joints at both ends in the configurations they
  were taught in. IK solves every frame of the move from those, so the line runs in the taught
  configuration.

The goal turns the short way between the two orientations. A turn longer than 30° gets extra
keys along it, so Blender's interpolation stays on the shortest turn.

A straight line can't change configuration, so a linear move whose two ends were taught in
different ones is refused, the way a robot controller refuses it. Make it a joint move, or
re-teach one end.

It can't change a joint's turns either. On a wrist that turns ±350°, one pose can be held
with a joint a whole turn apart: 20° and −340° put the tool in the same place. An end taught
a whole turn from where the line arrives keeps the turn the line arrives with, the way a KUKA
LIN ignores its target's Turn. Generate Motion says which joint and both values, and the
move's row in the Motion Check keeps saying so. Re-teach that waypoint if you meant the
other turn.

Nothing about the result is special, which is the point, and Blender's own graph editor
shapes it afterwards. Linear moves come out LINEAR, so the tool's speed changes all at once
at each waypoint. Easing them is up to you:
- **Easing a linear move** changes the *speed profile along* the straight line without
  bending it. X, Y and Z share the interpolation shape, so only the parameterisation changes.
  That is a real robot's acceleration ramp.
- **Bezier handles across a corner** of three waypoints bow the path through it. That is the
  blend radius a real controller would apply.

**Graph Editor**, in the quick bar and the Joints (FK) header, takes one click to open
those curves:
- **Where.** A Graph Editor already in the window is reused. Otherwise the bottom third of
  the 3D viewport becomes one, and the Timeline stays. Or, as the preferences say, it opens
  in a window of its own.
- **What.** The robot becomes the selected object, and the active one, with its joint bones
  selected and none of its other bones. Other objects are deselected, except one being
  edited or painted. *Only Show Selected* is on, so the editor lists the joint curves, each
  with a slider, plus the live-IK switch where a job keys it.
- **How.** The curves are normalised, so a rail's metres and a wrist's radians share one
  scale. The sidebar is redrawn during playback, so the joint sliders and Motion Limits
  follow the animation.
- **Undo.** Ctrl+Z puts the selection back as it was. The editor stays as it is, since
  Blender never undoes a change to the layout or to an editor's settings. **View › Area ›
  Close Area** closes it.

Each setting is in Kinema's preferences, with a Reset. The Graph Editor's own sidebar has a
Kinema tab with **Kinema View**, which sets it up again, and **Blender Defaults**, which
puts back what Blender gives a new Graph Editor.

Regenerating replaces the previous motion rather than layering on it, so moving a waypoint
from frame 40 to frame 30 does not leave the robot visiting frame 40 as well.

Generating ends with a **Motion Check**: the job is played through once, solved as playback
solves it, and each move gets a row listing what went wrong:
- how far a linear move's tool strayed from its line, or turned off the shortest turn;
- a joint that reached its limit;
- the fastest joint against its velocity limit;
- the hardest-accelerating joint against its acceleration limit, measured at the waypoints
  too, from the frames either side: a linear move changes speed there at once;
- a joint that jumped more than a radian (or 25 cm) in one frame, which is how a
  configuration flip shows.

**Check Again** plays it through after you've edited the curves by hand. Once the waypoints
change, the panel says the check is out of date. On a job Optimize Motion wrote, it also says
how many frames a move still over a limit needs.

**Optimize Motion**, under Generate Motion, solves the generated job again as one motion that
keeps within the robot's limits. It keeps:
- the tool at each waypoint, in the configuration taught there, and on every linear move's
  line;
- the job's first and last frames, with the robot at rest before and after them;
- held joints, and joints the tool doesn't hang from, where the job has them.

Everything else may change. Within each joint's range, top speed and acceleration limit, and
clear of the obstacles (see [Checking for collisions](#checking-for-collisions)), it takes the
smoothest path it can. Then it keys every free joint on every frame with live IK
off, so the job plays as it was solved, and checks it again. A free joint is one on the
chain to the TCP that isn't held.
- On a KR10 R1100-2 with its own `joint_limits.yaml`, a four-waypoint job with joint 2 at three
  times its acceleration limit came back within every limit, solved in 0.2 s once compiled.
- On a KR210 with moves of 100-150 frames at 24 fps, joint 3 was at 7.3 times its
  acceleration limit where a joint move met a linear one. It came back within every limit,
  at most 78% of one, solved in 3 s.

A linear move keeps the timing Generate Motion gave it: the tool runs along the line at a
steady speed, all the way to its ends. The joint moves either side ease into it and out of
it, but the tool still turns from one direction to the other at the line's ends. Give the line
more frames to make that gentler.

Where a move hasn't the frames to keep within its limits, it comes out over them by as little
as the waypoints leave room for, and its row in the Motion Check says about how many frames it
needs. Give its motion more frames and optimize again: waypoints are never moved, and the job
is never retimed.
- It optimizes the job as generated, so after editing the waypoints or motions, Generate
  Motion again first. Generate Motion afterwards replaces it.
- Acceleration limits come from a `joint_limits.yaml` (Load Joint Limits). Without them, it
  keeps to the speeds and ranges.
- From the panel it runs while Blender stays live, and Esc abandons it, writing nothing. If the
  waypoints or the robot change before it finishes, it writes nothing either: a joint held or
  released, a held joint moved or its keys edited, a limit, the TCP, Root's pose, or the frame
  rate. So do the obstacles, the capsules, where the robot stands, and Keep Clear By, while it
  keeps clear. Scrubbing the timeline meanwhile is fine.
- The first run for a robot and a job length compiles first, which holds Blender up for
  seconds. JAX's compile cache keeps it for the sessions after.
- It needs the PyRoki solver, whichever solver live IK uses.

When the shot is right, **Bake IK to Keyframes** turns the whole thing into plain joint
curves that render with Kinema uninstalled. Joint moves bake as they play, and linear moves
bake as IK solves them. A job on the NLA is baked into a copy of its action, shown as a
strip on a *Kinema Baked* track above the job over the baked range, and the job's own keys
are left as they were: mute the *Kinema Baked* tracks to go back to the live job.
- A bake over part of an earlier one goes on a track of its own above it, so the earlier
  bake still plays outside the new one's frames. A bake never goes onto a muted track.
- With the job on the NLA, the next Generate mutes the bakes over the job's frames, since
  they show the job as it was, and says so. With **Job as NLA Strips** off, the job in the
  active action plays over them.
- A bake onto a new NLA strip needs two frames or more, since a strip needs a length. In
  NLA tweak mode, Bake keys the tweaked action, and one frame is enough.
- A bake onto a strip is refused while the robot's NLA is off or one of its tracks is
  soloed: the strip wouldn't play, and a soloed track would hide what is baked.

## Checking motion limits

A robot can't follow motion faster than its motors, and Blender plays back a joint keyed
through half a turn in two frames as smoothly as one given two seconds. Most robot
descriptions give each joint a top speed in `<limit velocity>`, and a `joint_limits.yaml`
can add how hard it may accelerate. Kinema keeps both on the joint and checks the frame on
screen against them.

A joint over either limit is flagged in two places:

- **In the panel.** Its slider in **Joints (FK)** turns red. **Motion Limits** lists every
  joint's speed against its limit under **Speed**, and its acceleration under
  **Acceleration**.
- **In the viewport.** Its dial or arrow is redrawn in red, with its name and how far over
  each limit it is: `joint2  speed 130%  accel 150%`.

During playback, Blender redraws the sidebar only with **Timeline › Playback › Play In ›
Properties and Sidebars** on, and it's off by default. Without it, the sliders and rows
follow the animation only where live IK is solving it, though the viewport follows every
frame either way. Blender reads the setting from the window playback is started in, so
**Graph Editor** turns it on in the window it's pressed in, and in the Graph Editor's own
window if it opens one.

**Ignore Motion Limits**, in that section, turns both off for one rig: for a shot that will
never run on a robot, or a description whose limits are placeholders. The limits stay
listed, greyed out, and nothing is measured for that rig until you untick it.

| Joint | Measured against | After a jump or a scrub |
|---|---|---|
| Keyed, including joint moves from Generate Motion and baked IK | its own curve: one frame back for speed, two for acceleration | exact |
| Driven by live IK, including linear moves | the poses seen one and two frames earlier | `—` until you play or step a frame, or two for acceleration |
| Neither | the poses seen one and two frames earlier | still |

Both use the scene's frame rate. A gap of up to three frames, from playback dropping frames,
is averaged. An average can miss a spike inside the gap, but never reports one that isn't
there. A joint pressed against its range stop doesn't count as moving, whatever its curve
does.

**What it costs.** Each rig's readings are worked out once per frame, and shared by the
sliders, the rows and the viewport. It was measured on a desktop (i7-12700K) with a Unitree
G1 with hands, which has 43 joints, keyed on every frame and limited in speed and
acceleration:

| | Cost |
|---|---|
| Working out the rig's readings | 1.2 ms a frame |
| Recording each frame, for the joints live IK drives | 0.2 ms |
| A window redraw with nothing changed, such as orbiting the view | 1 ms more than with the rig ignored, mostly drawing the joints flagged |
| A frame of playback, with the Kinema sidebar open | 2.3–2.8 ms more than with the rig ignored |

Playback held 24 fps either way. Six 7-joint arms, 42 joints in all, take 1.6 ms a frame
between them. With **Ignore Motion Limits** ticked, a rig costs nothing.

Jerk and effort limits are kept on each joint too, and listed under **Not Checked**. Only
speed and acceleration are checked. Effort would need the robot's masses and inertias, which
a rig doesn't carry. Jerk would add little: between frames, a joint passes a jerk limit only
if its acceleration changes by more than that limit over the frame rate, within one frame.
For a Franka Panda at 24 fps that is a change of over 300 rad/s², against an acceleration
limit of 15, so the acceleration check has flagged it already.

**Load Joint Limits…** reads a MoveIt or ros2_control `joint_limits.yaml`. That is often
where a robot's real running speeds live, and where its acceleration limits are, since URDF
has no field for them. It reads velocity, acceleration, jerk and effort, joint by joint and
kind by kind:

- `has_acceleration_limits: true`, with `max_acceleration`, sets that limit;
- `false` turns it off;
- a joint or a kind the file doesn't mention keeps the description's value.

MJCF has no velocity or acceleration limits, so an MJCF rig only gets them from such a file.
Its effort limits come from each joint's `actuatorfrcrange`.

## Checking for collisions

The Motion Check also measures how close the robot comes to what stands in its way, on every
frame of the job. Both sides are simple shapes, so the distances are exact and quick. The
**Collision** panel sets them up.

**The robot is a set of capsules.** **Fit Capsules** fits them round the meshes on each joint
bone, and on the TCP for an attached tool:
- a link's imported collision meshes where it has them, otherwise what the link shows, and
  either way whatever was added to it, an attached tool;
- each link split into up to four capsules, the fewest that come close to the tightest fit,
  so an L-shaped link isn't one fat capsule. On a KR210 the forearm gets four, at most
  401 mm in radius where one would be 513 mm, and the whole arm fourteen, fitted in
  4.5 s. The same meshes give the same capsules in any pose;
- each capsule the tightest found round its part, along the part's main direction or the
  least-volume cylinder's, trimmed so its rounded ends only just take in the mesh. Not that
  cylinder with its ends rounded, as PyRoki fits one: each end would stand a whole radius past
  a face, and a squat link's least-volume cylinder stands on its shortest side. A KR120's wrist,
  210 mm thick, got a capsule 579 mm tall that way, and its turret one reaching 0.36 m past
  its meshes;
- every point of the meshes inside them;
- not the base, which stands on whatever floor there is.

The fit chooses how many capsules a link gets by their volume, and volume can't see where the
bulk sits, so a link can still come out with fewer than suit it. **Capsules per link**, under
the fit, sets a link's count by hand, from 1 to 4, or leaves it on **Auto**. Fit Again after:
the panel asks for it. Links behind a fixed joint ride on the bone of the joint before them,
and are listed and counted with it: one count for their meshes together.

Capsules fitted by an earlier version of the fit, as in a file saved before it, are still
used as they are. The panel says "Fitted by an older version: Fit Again", which fits them
tighter.

They are kept on the rig and move with it, however the job plays, and scale with it: a robot
scaled unevenly gets the widest of its scales. After changing a link's meshes, faces, shape keys or tool,
the panel says so: **Fit Again**. The **×** beside it clears them, and with them all the
collision work Kinema does for the robot as it moves.

Capsules ride rigidly on their bones, so a link mesh that also moves on its bone, by a
constraint, a driver or keys, or changes shape as the job plays, by animated shape keys or
modifiers, can't be held by them. Nor can an attached collection with anything like that inside
it. **Fit Capsules** names it and fits nothing until what moves it is removed or applied.

**Obstacles are the objects in the scene's *Kinema Obstacles* collection,** directly in it:
not those in a collection inside it, which the panel counts. **Add Selected** puts the
selected objects there, and **Remove Selected** takes them out, leaving them in the scene. Each
scene has its own. The robot is never one, nor anything riding on it. Selecting them leaves the
robot in the panels: in Object Mode they keep the robot they last showed, marked *Not
selected*. Each is measured as one of three shapes, chosen in the panel:
- **Box**, the default: the object's own bounds, turned and placed as it stands. Tight for
  anything modelled along its own axes;
- **Sphere**: the smallest ball about its centre that holds all of it: a ball, stretched or
  not, is measured as the ball it is;
- **Floor**: everything below the plane through its origin, square to its Z axis.

A complicated fixture is a few of them. Obstacles are read on every frame, so one that moves
is measured where it is. What an obstacle follows by a constraint or a driver isn't watched,
though: move that, and neither the check nor Optimize Motion notices. Check Again after
([#94](https://github.com/ebgenius/kinema/issues/94)).

**The check** names, for each move, the link that came closest, to which obstacle, and on
which frame. A link reaching into an obstacle is flagged: "joint_3 hits Fixture by 40 mm at
frame 134". Where a capsule is further than its radius from where it was against an obstacle
on the frame before, either of them having moved, the check measures between the two frames
too, in up to 16 even parts: "joint_3 hits Plate by 12 mm between frames 40 and 41". That's
sampling, not a continuous check: a capsule moving more than 16 radii in a frame, or going
out and back within one, can still pass a thin obstacle unseen. The panel shows the closest
approach over the whole job, until the waypoints,
the obstacles, the capsules or where the robot stands change, when it asks for Check Again. **Show Collision
Shapes** draws the capsules and obstacles as the check sees them, red where they touch on the
current frame. That's the way to see a capsule that fits its link badly.

**Optimize Motion keeps clear of them.** With capsules fitted and obstacles in the scene, it
keeps every capsule at least **Keep Clear By** (10 mm unless changed) from every obstacle, on
every frame of the job. Both are set in the Collision panel. The waypoints and the lines stay where they
are, so the arm goes round an obstacle between them: higher, wider, or with its elbow another
way. An obstacle that moves is kept clear of where it is on each frame, and so is a robot
carried along.
- Where the job can't keep clear, the job wins. A waypoint, a line or the job's first or last
  frame inside an obstacle stays where it was taught, and the moves round it stay close to
  their limits; the Motion Check names the link, the obstacle and the frame. Pulling the arm
  out harder there would only drag the waypoints off and make the joints lurch.
- Where going round takes more than a move's frames allow at the robot's speeds, it comes
  as close to clear as the limits leave room for, and the Motion Check says where it still
  hits. Give the move more frames, or move the obstacle.
- **A hit it can't avoid changes the move's timing instead of its path.** Clearance is
  measured frame by frame, so where a link can't get clear, crossing in fewer frames is all
  that's left to lower it. The arm winds back first, crosses at full speed, and passes its
  next waypoint before coming back to it. A start or an end inside an obstacle does the same.
  - Where a solve leaves a capsule short of clear on a move, and no further out than it was
    taught, the job is solved once more, asking that capsule on that move to keep no further
    from that obstacle than it came closest. Its frames then cost nothing, so the move isn't
    hurried. That capsule may come as close anywhere on the move.
    - A lug on joint 1 swept through a post, solved once, wound the joint back 0.28 rad and
      crossed at 1.03 times its speed limit. Solved again, it keeps its taught path, at 0.43
      of the limit.
    - A waypoint taught 8.7 mm from a post, with Keep Clear By at 50 mm, is left at the same
      pace as with Avoid Obstacles off, where solved once the arm left it 3.5 times as fast.
    - The second solve takes as long as the first, and the report says when there was one.
  - Where the solve did bring that capsule further out somewhere on the move, nothing is
    conceded, and the frames next to a waypoint taught closer than Keep Clear By still
    hurry: on the KR120 job below, its last frame 57 mm from the cube, asking for 100 mm
    winds the base back 0.09 rad ([#98](https://github.com/ebgenius/kinema/issues/98)).
  - A capsule much bigger than its link does it: Show Collision Shapes shows one. On a KR120
    at 24 fps, a joint move swinging the tool 2.4 m past a 1.2 m cube went over it, but the
    turret's capsule, fitted as a cylinder with its ends rounded, reached 27 mm into the
    cube whenever the base faced it. The turret itself stood 337 mm clear. The base wound
    back 0.34 rad, crossed at its top speed and overshot by 0.27 rad, at any number of
    frames. Fitted as it is now, the turret's capsule stands 227 mm clear facing the cube.
    On that scene, with Keep Clear By at 10, 20, 30 and 50 mm, the same move came out 12, 22,
    32 and 51 mm clear, within every limit, with no wind-up and no overshoot. The base still
    holds back, then crosses faster than taught: at 50 mm a tenth of its swing comes 4
    frames later, and it peaks at 0.33 of its speed limit, against 0.29 as taught
    ([#98](https://github.com/ebgenius/kinema/issues/98)). Where a capsule still stands far
    out, give its link more: Capsules per link.
  - Where the hit is real, move the obstacle, or teach the waypoint clear of it.
  - Letting a move take longer to go round is [#93](https://github.com/ebgenius/kinema/issues/93).
- A solve that ends without its cost or its gradient having settled, having cut the cost by
  more than 1%, is solved again from where it stopped, up to four times. Whether the robot
  is clear yet doesn't enter into it.
- With obstacles, the job is solved two ways and the cheaper result kept: as above, and with
  the pull from deep inside an obstacle stronger at first. Solved one way only, a solve
  could settle with the arm deep in a crate, and a frame more or a waypoint 1 mm over could
  put it there. A solve with obstacles takes 7-10 s on the KR120 job above, where it took
  2-4 s before either of these.
- Optimize Motion measures the obstacles on whole frames only. A link could still pass
  through a thin obstacle between two frames of a fast move; the Motion Check, which
  measures between them too, then names the frames either side
  ([#99](https://github.com/ebgenius/kinema/issues/99)).
- **Avoid Obstacles**, beside it, turns it off. Capsules that no longer fit the meshes are
  refused while it's on: Fit Again first.
- On arm6, a post 72 mm into the forearm mid-move was gone round, 12 mm clear, within every
  limit, the waypoints within 0.001 mm. On a KR210 at 24 fps, a post the wrist swept
  through 26 frames before a linear move came out 3 mm in: going round in that time would
  have taken a joint near three times its acceleration limit. Solving its 400 frames took
  8.7 s on a desktop, against 3.0 s without obstacles. With a post across its line instead,
  330-420 mm into the arm, the waypoints stayed within 0.1 mm and the joints reached at most
  2.2 times an acceleration limit.

Self-collision is still to come.

## Choosing how the arm reaches

A six-axis arm can put its tool in one place up to eight different ways — elbow bent one way
or the other, wrist flipped, base swung round behind. Until now you got whichever one the
solver landed on from wherever the arm happened to be, which is how a shot ends up with the
elbow on the wrong side of the robot and no obvious way to fix it.

**Find Solutions**, in the IK panel, searches for the others. PyRoki is a nonlinear solver
rather than an analytic one, so it does not enumerate branches — it converges to whichever
is nearest its seed. So they are *found*: seed from configurations scattered across the joint
limits, solve each, keep the ones that reached the goal, fold away the duplicates. That makes
it a **lower bound** and the panel says so; more seeds may find more.

The ◀ ▶ arrows cycle through what turned up. Applying one only writes joint values, so
**Key All** keyframes it and it bakes like anything else — flipping the elbow between two
frames is two keyed poses, which is what a robot programmer would do anyway.

Solutions belong to one goal pose. Move the IK target and the quick bar names the one
picked, *Config 8*, without how many there were: they were found for a pose the target has
left. Clicking it searches again, and the arm stays where it is: a search selects the
configuration nearest the one the arm stands in. The arrows search again first, then step
from there.

**Auto Configuration**, in the quick bar and the IK panel's Configuration section, does
this for you while live IK tracks the target. A live solve that misses the target means one
of two things: the target is out of reach, or the arm can't get there from the
configuration it is in, because the wrist would have to flip or a joint is at its limit.
With it on, a miss runs a search, which tells the two apart. If it finds configurations
that reach the target, the arm switches to the one nearest where it was; if it finds none,
nothing changes. It searches 12 seeds, about 0.15 s on a six-axis arm, and after a search
that found nothing it waits until the target has moved on 1 cm or 5° and half a second has
gone by. It runs for viewport solves only: playback and renders play what was keyed.
Held joints stay where they stand in every solution, so on a robot with its rail held, every
alternative is an arm reaching from where the carriage already is.

On a KR120 at a working pose, this turns up the arm you are in plus the wrist-flipped one
— joints 1 to 3 identical, 4 to 6 turned through 180° — with the tool holding to 0.008 mm
across the switch. The other families a KR120 nominally has are ruled out by its own joint
limits at that pose, which is the correct answer rather than a shortfall of the search.

### Spare joints: rails you move by hand, and the elbow swivel

A robot with more joints than a tool pose needs can reach that pose a continuous family of
ways. What the spare joint *is* decides which control makes sense, so there are two.

**Held joints.** Most redundant robots in a cell are a six-axis arm on a rail or a gantry.
The extra axes are not something to solve for: an operator positions them, and the arm
reaches from wherever they stand. So a joint can be **held by hand**. Click the pin beside
its slider in the **Joints (FK)** panel, then move it with that slider or by dragging its
bone along its axis, and IK solves the rest of the chain with the tool held where it is. The
pin is keyframable, so a rail can be handed back to the solver part-way through a shot.

A held joint is solved where the rig shows it. Drag a rail past its limit and the arm
reaches from where the limit stops the carriage, while the channel keeps the value you
dragged, rather than planning around a carriage that isn't there.

**Add IK Target** pins rails for you. While a chain still has more than six unheld joints,
its prismatic joints are held, base first. So a six-axis arm on a rail gets its rail held and
nothing else, and an arm with nothing to spare, or whose spare joint is rotary, holds
nothing. A turntable can't be told apart from a seventh arm joint, so pin that one yourself.

**The elbow swivel.** On a seven-axis arm the spare freedom *is* the elbow: with the tool
held still, it can swing round the line from shoulder to wrist. **Add Elbow Swivel** puts a
ring on that line, and turning the ring swings the elbow round it. It's the same idea as the
control that steers a character's elbow once IK has placed the hand.

- **The ring rides the arm.** A hidden bone follows the shoulder-to-wrist line through
  Blender's own constraints, so the ring stays with the arm with IK off, during playback and
  at render.
- **Its one rotation channel is the elbow's angle.** It keys as a single curve, and the IK
  panel shows it as a **Swivel** slider with its own keyframe dot.
- **It aims at a point the elbow can actually reach:** a point on the circle the elbow
  sweeps, not wherever a bone happened to be dragged. On the `arm7` fixture, swinging the
  elbow through ±1.5 rad keeps the tool within 0.0003 mm and the elbow within 0.02° of the
  angle asked for, at every strength from 0.5 to 20. The free-floating elbow target this
  replaced pulled toward points off that circle and gave up tool accuracy instead: 0.55 mm at
  its default strength, 30.7 mm at 20.
- **It keeps up with the target.** The circle is built about where the wrist is going, not
  where it is, so the single live update a drag produces already puts the tool on its goal.
  Dragged past the arm's reach, the elbow stays on the side the ring put it instead of
  flipping over.
- **Adding one changes nothing.** The ring is turned onto the elbow before anything solves.
- **It doesn't vanish when a hold takes its freedom.** Pin a joint so that six or fewer are
  left to IK and the section stays, with the slider greyed, a line saying why, and
  **Remove** still there. Release the joint and the swivel works again.

Shoulder, elbow and wrist default to the second, middle and second-to-last joints. That is
exact for an arm with a spherical shoulder and wrist, like the `arm7` fixture or an iiwa, and
only approximate, and not yet measured, on an arm whose links are offset, like the Panda.
**Add Elbow Swivel** is offered only when more than six joints are left to IK, so a rail robot
with its rail held has no elbow to spare. The swivel also needs the PyRoki solver: the NumPy
fallback has no notion of a second goal, and the panel says so.

## External axes the description doesn't have

A robot often arrives as the arm alone, while the cell it works in has more: a track it
rides, a turntable under it, a positioner holding the part, a spindle on its flange.
**External Axes → Add External Axis…** adds one to the rig. The dialog asks for:

- **Preset:** Linear Track, Rotary Base, Positioner, Tool Spindle or Tool Slide. Each one
  just fills in the fields below, and any of them can still be changed.
- **Mount:**
  - *Under the Robot*: the robot rides it.
  - *On the Tool*: it rides the flange and carries the TCP.
  - *Standalone*: it stands beside the robot and carries nothing until you attach a part to
    it in the **Bones** panel.
- **Motion and direction:** linear or rotary, along or about ±X, ±Y or ±Z of the axis's base
  placement. A rotary axis can be continuous.
- **Limits and top speed.** The speed feeds the **Motion Limits** check like any other
  joint's.
- **External axis base:** where the axis sits, calculated from the current robot base, or
  from the current tool frame for an axis on the tool. Location, plus roll, pitch and yaw
  about fixed X, Y and Z, as in URDF.
- **External axis offset:** where what the axis carries sits on its moving part. That is the
  current robot base for an axis under the robot. For an axis on the tool it is the TCP,
  measured from the current TCP: with no offset the TCP stays where it is, however the base
  moves or turns the axis.

**The current robot base** includes any axes already under the robot. A new axis under the
robot goes in at the bottom of that stack, and its base is measured from just under the
lowest one's rail or turntable. Add an X track, then a Y track, and the Y track goes under
the X track's rail and carries it, as if the X track were already part of the robot.

The axis is an ordinary joint bone, built by the same code as an imported joint. It gets a
slider in **Joints (FK)**, keys, bake and a speed check, and it starts **held**, since it
is positioned by hand and the arm reaches from wherever it stands. Release the pin to let IK
move it too.

**See it before you add it.** While the dialog is open, the viewport shows the result and
follows every field you change:

- the rail and carriage, or base and plate, when **Placeholder** is ticked;
- the travel: the carriage outlined at both end stops, or an arc from stop to stop, with the
  limits written beside them;
- for an axis under the robot that moves it, a ghost of the robot and the axes it already
  rides, where they will stand;
- for an axis on the tool, the TCP where it will be, as a bright X/Y/Z triad, and where it
  is now, as a faint one. They part as you edit the offset.

All of it is drawn as the arm stands, so an axis on a posed tool shows on the tool.

The ghost is a coarse copy of the meshes, moved as one block. Nothing is solved, and nothing
is added to the scene until you press OK. Cancel leaves no trace. Clicking in the viewport
to look closer closes the dialog, but opening it again brings back everything you typed.

**No compile while you adjust it.** After OK, the **Adjust Last Operation** panel re-runs
the add on every change. A PyRoki compile is tens of seconds, so none happens there: the rig
solves on NumPy, and the panel says so, until you move on to something else or leave the
axis alone for ten seconds. Then PyRoki is compiled once, behind a wait cursor.

**Moving the robot, and putting it back.** With base and offset at zero nothing moves: the
axis slides in under the robot and its axes, or behind the TCP, exactly where they are. Give
them values and the robot ends up at base-then-offset on the carriage. Everything riding the
robot moves with it by one rigid transform, so the pose you had is still the pose you have:

- the robot's bones and its TCP;
- the axes already under it, and their placeholders;
- the IK target and the swivel;
- the base link's meshes;
- the waypoints.

**Remove** undoes all of it, stacked axes in any order included. Something you parented to
the axis bone by hand stays where it stands. An axis on the tool moves the TCP by its offset,
and removing the axis puts the TCP back with the offset it had.

**Joint indices follow names.** Blender orders bones by hierarchy, so an axis under the robot
becomes joint 0. Waypoints' joint vectors and the IK tip, keyframes included, are remapped
by name when an axis comes or goes.

**Placeholders are generated, not shipped.** A rail with a carriage, or a base with a
turning plate and a pointer tab, built from a few dozen vertices at the size the dialog
suggests from the robot's reach. They ride their bones, use one material you can restyle,
and leave with their axis. Turn **Placeholder** off if you have a model of the real thing;
attach it to the axis bone in the **Bones** panel.

**The solver sees the axis.** The rig is no longer the robot in its description, so PyRoki
is handed a model written from the rig's own bones instead of the file. Each joint bone
already records its type, limits, link and link frame, so the model is exact. It matches
the description to 1e-5 on the fixtures, and arm6 with a track added is joint-for-joint the
hand-written `rail6.urdf`. Mimic joints get no bone, so they are left out. They never move
the tool.

## Where the tool frame sits

The **Tool Centre Point** panel places the TCP on a joint bone and offsets it from there.
**Edit TCP…** opens a dialog for it that previews before anything moves:

- **Offset:** type a location and rotation, as the panel's fields take them.
- **3D Cursor:** the TCP goes where the cursor is. Snap the cursor to a vertex of the tool
  mesh first (Shift+S), and the TCP lands on that vertex.
- **Object Origin:** the TCP goes on an empty, or on the origin of the tool mesh itself.

The cursor and the object can give the orientation too, or leave it at the typed one. Either
way the dialog shows the offset it will store, measured from the mounting flange (see below)
where it stands now, so a TCP can be picked off a tool on a posed robot. While it is open, the viewport draws the
new TCP next to the current one. Clicking away closes the dialog, and opening it again brings
back what you had. The panel's fields and *Update TCP* still work as before.

**Changing the TCP doesn't move the robot.** Redefining the tool, as on a real controller,
leaves the joints where they are, and the IK target comes to the new TCP. A keyed IK target
goes back to its keys on the next frame, because an animated path is the tool's path, and
the new tool follows it.

**Changing the TCP doesn't recompile the solver.** PyRoki's compiled solver depends on the
robot and on the joint the TCP rides, not on the tool offset. Re-offsetting the TCP, or
removing and re-adding the IK target, reuses it. A TCP moved onto another joint needs a new
one, and it waits until you're done: live IK solves on NumPy meanwhile, and the panel says
so. It then compiles once, when you move on to something else or leave it for ten seconds.
A bake, Find Solutions, playback and rendering never wait. They use PyRoki, compiling it if
they must.

**Zero is the mounting flange.** The offset is measured from the flange the description
defines for that joint, the frame a tool's CAD is measured from. So a TCP taken off the tool's
drawing goes in unchanged, and `Z = 0.15` means 150 mm out of the flange face. The angles are
roll, pitch and yaw about fixed X, Y and Z, the convention URDF uses in `<origin rpy="...">`.

Industrial descriptions usually define the flange twice, as ROS-Industrial's REP-199 does:

- `tool0`, with Z out of the face;
- `flange`, with X out of it.

Kinema uses `tool0`. Where there's only a `flange`, it turns it so Z points out of the face,
as `tool0` would. A fresh import puts the TCP there, with zero offsets and the approach axis
out of the flange. The panel names the frame, for example *From the mounting flange (tool0)*.

A joint with no flange measures from its own **link frame**. The importer then puts the TCP
on the description's deepest link and seeds the distance to it into the offset, since fixed
joints get no bone and nothing else on the rig would show it. Rigs built before Kinema
recorded the flange keep measuring from the link frame too, so their offsets keep the meaning
they were typed with. Reimport to measure from the flange.

Bone axes are not tool axes and cannot be: a bone's +Y always runs head to tail. Kinema rides
the tool frame on the marker permuted — tool Z on the bone's Y — and reports the tool frame,
not the bone's, in the panel. The marker draws the approach axis long and arrowed so the
direction is readable at a glance.

## When the description moves

A rig keeps the path of the description it was imported from, and the PyRoki solver is
built from that file whenever it is needed. Where the description is in the .blend's folder
or under it, the path is also kept relative to the .blend when the file is saved, as Blender
keeps the paths of images: copy or move the project folder as a whole, to another place or
another machine, and the robot reads the description beside it. The relative path is tried
first, then the absolute one, so a .blend moved on its own still finds the file where it was.
A description anywhere else is kept by its absolute path only, as is every path while
Blender's *Default To: Relative Paths* preference, under *File Paths*, is off.

Move the description's folder, or open the .blend somewhere the file isn't, and Robot
Settings shows **Description file missing**, live IK falls back to NumPy, and Optimize Motion
says why it can't run.

**Locate Description…**, under that message, points the robot at the file again: a URDF,
xacro or MJCF, read as an import reads it, with the rig's xacro arguments, and
`package://` and meshes found from where the file is now. It is accepted only if it
describes the same robot: the same moving joints by name and type, each moving a link of
the same name, its frame in the same place and turned the same way with the robot at rest,
in the same chain. A different robot, or a changed one, is refused with what differs; a
changed robot is imported again. Position, speed and effort limits that differ are fine:
the rig keeps its own -- the position and speed limits the Motion Check and Optimize read,
and the effort shown with them -- and the report lists them. Nothing else in the file
changes, and the solver is rebuilt from the new file on the next solve, without reopening
the .blend.

A rig with an external axis needs none of this: the solver describes it from its own
bones. A rig linked from a library finds its description relative to the library's file,
and isn't offered Locate: open the library's file, locate the description there, and save it.

## Scripting Kinema

`kinema.api` is the part of Kinema that scripts and other add-ons can rely on.
Everything else is private and changes without notice. `api.VERSION` is
`(major, minor)`: the minor number grows when something is added, and the major
number changes when something changes or goes.

Blender extensions can't depend on one another, so another add-on finds Kinema at
runtime. Kinema's module name depends on the repository it was installed from, as
in `bl_ext.blender_org.kinema`, so look it up by its last part:

```python
import importlib

import bpy


def find_kinema():
    """Kinema's API, or None where Kinema isn't enabled."""
    for name in bpy.context.preferences.addons.keys():
        if name.rpartition(".")[2] == "kinema":
            return importlib.import_module(name + ".api")
    return None


kinema = find_kinema()
if kinema is not None and kinema.VERSION[0] == 1:
    rig = kinema.rigs()[0]
    print([joint.name for joint in kinema.joints(rig)])
    kinema.set_joint_values(rig, [0.0, -0.5, 1.0, 0.0, 0.5, 0.0])
    print(kinema.tool_frame(rig))  # 4x4, world space
```

It takes and gives Blender objects for rigs, NumPy arrays for joint values (in the
rig's joint order, in radians and metres) and for frames (4x4, world space,
metres), and small frozen dataclasses. A call that can't be done raises
`kinema.Error`, saying why. Nothing pushes an undo step.

| Function | |
| --- | --- |
| `rigs(scene=None)`, `is_rig(obj)` | The Kinema rigs in a scene, the current one by default |
| `joints(rig)` | Each moving joint: its name, bone, type, and the limits the rig holds |
| `joint_values(rig)`, `set_joint_values(rig, values)` | Where the joints stand. Setting them moves the IK goal to the tool, so live IK leaves them there |
| `tools(rig)`, `tool_frame(rig, tool=None)` | The tool centre point, the link it's measured from with its offset, and where it is now, its Z out of the tool |
| `solve(rig, pose, seed=None, tool=None)` | Joint values that stand the tool at a world-space pose, and whether they reach it, without moving the rig. Joints the rig holds stay at their seed |
| `capsules(rig)` | The capsules fitted to the rig, each on its bone |
| `obstacles(scene=None)`, `add_obstacle(obj, shape=None)`, `remove_obstacle(obj)` | What the robots are kept clear of: a box, a sphere or a floor |
| `clearance(rig, values=None)` | The closest approach to an obstacle, at the joint values given or where the rig stands, and which bone and obstacle |
| `reachable(rig, pose, seed=None, tool=None, keep_clear=0.0)` | Whether the tool can stand at a pose: reached, every joint in its range, and clear by `keep_clear` metres. `.ok` is all three |
| `waypoints(rig)`, `add_waypoint(rig, name=None, values=None, reference=None)`, `remove_waypoint(rig, name)` | Taught poses: taught at joint values without moving the rig, relative to an object or the robot's base |
| `motions(rig)`, `add_motion(rig, start, end, frame_start, frame_end, move="JOINT", name=None)`, `remove_motion(rig, name)` | Moves between waypoints, by name, joint or linear, over frames of their own |
| `add_ik_target(rig)` | The IK target linear moves are keyed on |
| `generate(rig)`, `check(rig)` | Generate Motion and Check Motion: whether it was done, the report, and how many moves were flagged |

A job taught, generated and checked from a script, on the arm6 test robot:

```python
import importlib

import bpy


def find_kinema():
    for name in bpy.context.preferences.addons.keys():
        if name.rpartition(".")[2] == "kinema":
            return importlib.import_module(name + ".api")
    return None


kinema = find_kinema()
rig = kinema.rigs()[0]
kinema.add_ik_target(rig)  # linear moves are keyed on it

home = kinema.add_waypoint(rig, "home", [0.0, -0.6, 1.2, 0.0, 0.6, 0.0])
above = home.pose.copy()
above[:3, 3] += (0.1, 0.15, -0.05)  # metres, in world space
reach = kinema.reachable(rig, above, seed=home.values)
assert reach.ok, reach
kinema.add_waypoint(rig, "above", reach.values)
down = above.copy()
down[2, 3] -= 0.1
pick = kinema.solve(rig, down, seed=reach.values)
assert pick.reached, pick
kinema.add_waypoint(rig, "pick", pick.values)

kinema.add_motion(rig, "home", "above", 1, 25)
kinema.add_motion(rig, "above", "pick", 25, 49, move="LINEAR")
kinema.add_motion(rig, "pick", "home", 49, 73)
report = kinema.generate(rig)
assert report.done and report.flagged == 0, report.message
print(kinema.check(rig).message)
```


## Requirements

Blender **5.2 LTS or newer** (embeds CPython 3.13). Everything else ships with the add-on.

## Development

Kinema is a [uv] project. The dev virtualenv pins Python 3.13 to match Blender's embedded
interpreter, which is what makes IDE debugging work without a Blender-specific extension.

```bash
uv sync --group dev                        # dev venv (Python 3.13)
uv run python tools/vendor.py              # vendor pyroki + jaxls source
uv run python tools/fetch_wheels.py        # download the wheel payload (~340 MB, 3 platforms)
uv run python tools/dev.py link            # live-link the add-on into Blender
uv run python tools/dev.py run             # launch Blender with Kinema enabled
```

Set `KINEMA_BLENDER` if Blender is not on `PATH` or you have several builds installed:

```bash
export KINEMA_BLENDER="/path/to/blender"
```

### Debugging from any IDE

`tools/dev.py debug` starts Blender with a [debugpy] DAP server. Attach from anything that
speaks DAP — VS Code, PyCharm, neovim-dap, Helix. No Blender IDE plugin required.

```bash
uv run python tools/dev.py debug --port 5678
```

VS Code `.vscode/launch.json`:

```json
{ "type": "debugpy", "request": "attach",
  "connect": { "host": "127.0.0.1", "port": 5678 } }
```

PyCharm: *Run → Attach to Process*, or a Python Debug Server on port 5678.

### Tests

```bash
uv run pytest                              # parsers, no Blender needed
uv run python tools/dev.py test            # in-Blender suite (real bpy)
```

The in-Blender suite runs in several Blender processes at once, each with a share of the
tests: half the logical cores, at most 8, and no more than the memory free when it starts
holds at 6 GB a process, but always at least one. On 20 cores and 64 GB it has measured 5
to 8, by what else was open, and the suite took 3½ to 5 minutes, where one process took 21. `-j N` sets the
number. `-j 1` runs it in one process, as before, its output as it comes, unless times are
being recorded. The shares are balanced by each test's last time, in `tests/durations.json`:
`dev.py test --record-durations` measures them again, and after a run of the whole suite,
with nothing ignored and every process reporting, drops the times of tests no longer there. A run in several processes, or one recording
times, keeps each process's log and prints their failures and counts at the end.

Run the non-Blender tests with UTF-8 mode enabled (`PYTHONUTF8=1`) on Windows. Blender runs
its interpreter in UTF-8 mode already, but the system default is cp1252, and some ROS xacro
files contain non-ASCII bytes that fail to decode under it.

### Building

```bash
uv run python tools/vendor.py              # vendored solver source, at the pinned commits
uv run python tools/fetch_wheels.py        # all three platforms (~340 MB)
uv run python tools/dev.py validate
uv run python tools/dev.py build           # per-platform zips into dist/
```

`vendor/` and `wheels/` are both gitignored, so a fresh clone has neither and the first
two commands are not optional. They are one-time setup rather than per-build steps —
re-run `vendor.py` only when its pins move, and `fetch_wheels.py` only when `uv.lock` does.
`dev.py build` refuses to run if either directory is missing, stale against its pin, or
holds one package at two versions; `vendor.py --check` reports the vendored state on its
own, offline.

The wheels are downloaded at exactly the versions `uv.lock` pins, so the zip carries what
the dev venv and the tests ran against. To move a package, run
`uv lock --upgrade-package <name>` and fetch again. A unit test fails if the manifest and
the lock disagree.

`--split-platforms` is the default and is not optional in practice: a combined zip would
carry three copies of `jaxlib` and exceed the extensions platform's size limit. Current
output, all comfortably under the ~200 MB ceiling:

| Platform | Zip |
|---|---|
| `linux-x64` | 136.5 MB |
| `windows-x64` | 116.6 MB |
| `macos-arm64` | 99.8 MB |

For 0.6.0, the Windows zip was installed into a clean Blender 5.2.1 profile
(`BLENDER_USER_RESOURCES` pointed at an empty directory) and checked:

- every bundled dependency loads from the zip's own wheels, and IK solves on PyRoki;
- a KUKA KR10 R1100-2 xacro imports through rospkg, with its TCP on `tool0`, and PyRoki
  solves it to 0.00 mm;
- MJCF and COLLADA files import, and Load Joint Limits reads a yaml;
- Blender reports no extension policy warnings, and no thread is started;
- a baked .blend plays exactly as baked in a Blender with no Kinema installed at all.

[uv]: https://docs.astral.sh/uv/
[debugpy]: https://github.com/microsoft/debugpy

## How dependencies ship

Blender extensions may not install packages at runtime, so everything is bundled:

| Dependency | How | Why |
|---|---|---|
| jax, jaxlib, scipy, lxml, … | bundled wheels (cp313) | normal PyPI packages |
| **pyroki** | vendored source | not on PyPI; `version = "0.0.0"`, git-install only |
| **jaxls** | vendored source | not on PyPI — the PyPI name `jaxls` is an unrelated "JAX Language Server" package |
| numpy | *not bundled* | Blender ships NumPy 2.3.4; a second copy risks an ABI clash |

Both vendored projects are MIT licensed; see `LICENSES/`. `tools/vendor.py` keeps the
vendored trees byte-identical to upstream except for dropping PyRoki's `viewer/`
subpackage, which depends on `viser` (a web 3D visualiser — pointless inside Blender).

An import runs as one blocking call behind a wait cursor. Parsing used to run on a worker
thread, with the build spread across modal timer ticks so Esc could cancel it, but Blender's
extension guidelines do not allow an extension to start threads.

### The robot catalogue

Kinema downloads nothing. It ships a catalogue of the 186 robots in
[`robot_descriptions`](https://github.com/robot-descriptions/robot_descriptions.py) as
static JSON, and *Browse Robot Catalog* turns a search into a `git clone` command on your
clipboard — with the pinned revision and, crucially, the path of the description file
*inside* that repository. `mujoco_menagerie` alone holds 2466 files and backs 49 of the
robots, so "clone this and find it yourself" would not be much of an answer.

`tools/build_catalog.py` generates `src/kinema/catalog/robots.json`. Resolving those inner
paths is the interesting part: a description module builds them with `os.path.join` over
whatever its cloning function returned, so importing it with that function stubbed to a
sentinel string yields every path relative to the repository root, offline. One of the 186
(`eve_r3_description`) rewrites its URDF at import and cannot be probed this way.

`src/kinema/catalog/curation.json` sits alongside it, hand-maintained and *subtractive*: an
entry named there is marked a duplicate, broken or partial and hidden from the picker until
**Show all variants** is ticked. Everything not named there shows normally.

## Known limitations

- **Ball joints are rejected.** A 3-DoF spherical joint has no honest single-axis bone
  equivalent. One catalog robot (Cassie) is affected; everything else parses.
- **PyRoki reads URDF only.** MJCF rigs are bridged by rendering the parsed kinematic
  tree back out as a minimal URDF, which works but is one more moving part.
- **Windows long paths.** The bundled wheels unpack to deeply nested directories —
  `jax/_src/internal_test_util/export_back_compat_test_data/…`. On a Windows profile
  without long-path support the install fails partway with `WinError 206`. Enable
  `HKLM\SYSTEM\CurrentControlSet\Control\FileSystem\LongPathsEnabled`, or keep
  Blender's config directory shallow.
- **A robot's first IK solve compiles.** JAX JITs the solver on first use, which takes a few
  seconds, more for a bigger robot or on a slower machine. It is paid up front when the IK
  target is created, behind a wait cursor. The compiled solver is kept in Kinema's own user
  folder, and a later Blender session loads it instead of compiling again, for as long as it
  is kept:
  - **Keep Compiled Solvers** in the add-on preferences turns keeping them off;
  - **Clear** empties the folder;
  - the folder is trimmed back to 256 MB, oldest first, at the start of each session;
  - an update that changes the solver's code compiles it afresh.

  Adding an **Elbow Swivel** pays one more compile, because an elbow goal makes a different
  problem; pinning a joint pays none. Warm solves are ~5–20 ms.
- **Imports block.** A big robot freezes Blender until it has finished loading, with no
  progress and no cancel. Extensions may not start threads, which is what kept the viewport
  alive before.
- **MuJoCo's OBJ meshes print MTL errors** on import. MJCF carries its own colours, so
  the missing .mtl files are harmless console noise from Blender's OBJ importer.

## Contributing

Bug reports, suggestions and pull requests from forks are welcome. See
[CONTRIBUTING.md](CONTRIBUTING.md) for how to report a bug and how pull requests are
reviewed.

## License

GPL-3.0-or-later (the add-on links `bpy`). Vendored PyRoki and jaxls remain MIT.
