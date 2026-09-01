# Your first robot

Five minutes, start to finish: pick a robot from the catalog, look at what you got, and
move it.

We will use a **UR5e** — a six-jointed industrial arm from Universal Robots. It is the
"default cube" of robot arms: small enough to understand, real enough to be useful.

## 1. Open the Kinema tab

In the 3D viewport, press <kbd>N</kbd> to open the sidebar, then click the **Kinema** tab.

With nothing selected you will see two buttons and a note saying no robot is selected.
That is the starting state.

![Screenshot: the Kinema panel in its empty state.](../assets/images/kinema_panel_empty.png){ .screenshot }

## 2. Import from the catalog

Click **Import from Catalog…**.

A picker opens listing 186 robots — arms, humanoids, quadrupeds, drones, grippers — with
the maker and joint count for each. Type `ur5e` to filter, select it, and confirm.

!!! info "The first import downloads"
    Kinema ships the *catalog* (names, makers, joint counts) but not the robots
    themselves — that would be gigabytes. The first time you import a given robot it is
    downloaded and cached; after that it loads offline. See
    [Robot catalog](../reference/catalog.md).

![Screenshot: the catalog picker filtered to ur5e.](../assets/images/kinema_catalog_picker.png){ .screenshot }

## 3. Look at what you got

One armature, named after the robot, with its meshes parented to it. The Kinema panel now
shows the robot's name and **6 DoF** — six degrees of freedom, meaning six independently
movable joints. ([What's a DoF?](../concepts/links-and-joints.md))

Drop into **Pose Mode** and open the bone collections in the Properties editor
(Object Data → Bone Collections). You will find four:

| Collection | What is in it | Visible? |
|---|---|---|
| `Kinema/FK` | One bone per joint — your controls | Yes |
| `Kinema/TCP` | The tool centre point marker | Yes |
| `Kinema/IK` | The IK target, once you add one | **Hidden** |
| `Kinema/Mechanism` | Internal bones that make the rig work | **Hidden** |

!!! tip "Two collections start hidden on purpose"
    `Kinema/IK` and `Kinema/Mechanism` are hidden so that a fresh rig shows you controls
    and nothing else. If you were expecting more bones than you can see, they are there —
    just tucked away. Toggle their visibility any time.

## 4. Move it

In the **Joints (FK)** panel there is one slider per joint, labelled with the joint's
real name from the manufacturer's data, and each one is clamped to that joint's true
range of motion — you cannot drive the elbow through the forearm.

Drag a few. The arm moves.

![Screenshot: the Joints (FK) panel, one slider per joint.](../assets/images/kinema_joints_sliders.png){ .screenshot }

Two buttons sit at the top of that panel:

- **Rest Pose** — everything back to zero, the robot's home configuration
- **Key All** — a keyframe on every joint at the current frame

This is forward kinematics: you set joint angles, the robot's shape follows. It is exactly
how posing an ordinary Blender armature works.

## 5. Now do it the other way round

Open the **Inverse Kinematics** panel and click **Add IK Target**.

You get a control at the robot's tool tip. Grab it and move it — the whole arm
reconfigures to follow, respecting every joint limit. Keyframe that control and you have
animated the robot by animating the thing you actually care about: where the tool goes.

The panel reports how long each solve took, typically a few milliseconds.

![Screenshot: an IK target being dragged, with the "Last solve" readout visible.](../assets/images/kinema_ik_solve.png){ .screenshot }

!!! warning "The first solve takes about 15 seconds"
    Creating the IK target compiles the solver for this specific robot, behind a wait
    cursor. It happens once per robot, never again. Later solves are milliseconds.

## Where to go next

- [Pose a robot by hand](../tutorials/pose-fk.md) — FK in depth, and when it is the right
  tool
- [Animate with an IK target](../tutorials/animate-ik.md) — the real animation workflow
- [Bake and hand off](../tutorials/bake.md) — make the `.blend` work on machines without
  Kinema
- [Import your own robot](../tutorials/import-your-own.md) — when the robot is not in the
  catalog
