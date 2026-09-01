# Links, joints and DoF

Robotics has its own words for things you already understand. Here they are, translated.

## Links and joints

A robot is described as a chain of **links** connected by **joints**.

- A **link** is a rigid piece of the machine — a forearm, a base, a gripper finger. It
  never bends or changes shape. In Blender terms, a link is a mesh.
- A **joint** is the connection between two links, and the thing that moves. In Blender
  terms, a joint is a bone.

That mapping is the whole trick, and it is worth stating plainly because it is the
opposite of the intuition most people arrive with:

!!! info "Bones are joints, not limbs"
    In a character rig, a bone *is* the upper arm. In a robot rig, a bone is the **elbow**
    — the rotating connection — and the forearm mesh is parented to it.

    So a six-jointed arm has six control bones, and the meshes hang off them. If you go
    looking for a bone shaped like the forearm, you will not find one.

## Degrees of freedom

A **degree of freedom** (DoF) is one independent way the robot can move. Kinema shows this
count in the main panel: a UR5e reads **6 DoF**.

For nearly every robot you will meet, DoF is simply the number of movable joints. Six
joints, six DoF, six sliders in the [Joints (FK)](../tutorials/pose-fk.md) panel.

Six is not an arbitrary number. It takes exactly six to place a tool at an arbitrary
position *and* an arbitrary orientation in space — three for position (X, Y, Z), three for
orientation (roll, pitch, yaw). That is why industrial arms are overwhelmingly 6-DoF.

Arms with **seven** joints are called redundant: they have a spare, so they can reach the
same tool pose in infinitely many configurations. That is useful for reaching around
obstacles, and it means the solver has a choice to make — which is a large part of why a
good IK solver matters.

## Joint types

| Type | Motion | Real example |
|---|---|---|
| **Revolute** | Rotates, with limits | An elbow that stops at 150° |
| **Continuous** | Rotates without limits | A wheel, a wrist that spins freely |
| **Prismatic** | Slides along a line | A linear rail, a gripper finger |
| **Fixed** | Does not move | A bolted-on bracket or sensor mount |
| **Ball** | Rotates on three axes | Not supported — see below |

Fixed joints still matter even though nothing moves: they carry the offsets that place one
link relative to another. Kinema folds them into the rig's structure rather than giving
you a useless slider.

**Ball joints are rejected.** A 3-DoF spherical joint cannot be represented honestly as
one single-axis bone, and faking it would give you a control that lies about what the
machine does. Exactly one catalog robot is affected.

## One bone, one axis

Every joint bone in a Kinema rig rotates on exactly one axis: the bone's local Y is
aligned to the real joint axis from the robot's description file.

This is a deliberate constraint and it is what makes the rig usable:

- One slider per joint means something — it is *the* angle of that joint
- The limits are real limits, not approximations of a 3-axis rotation
- The rig cannot be posed into a shape the real machine cannot achieve
- The values you read off correspond to numbers a robot engineer would recognise

The cost is that you cannot freely rotate a joint bone on its other two axes. That is not
a limitation of the rig — it is a fact about the robot.

## Multi-turn joints

Some joints spin past 360°, and how far round they have gone is real information: a wrist
at +400° is not in the same state as one at +40°, even though the robot looks identical.
Unwinding matters when the tool is trailing a cable.

Kinema tracks this rather than wrapping angles into a single turn. The joint value keeps
counting.

## The root

Every Kinema rig has a root bone that the whole chain hangs from. Move it and the entire
robot moves — that is how you place the machine in your scene.

It lives in the `Kinema/FK` collection alongside the joint controls, and the
[IK target](../tutorials/animate-ik.md) is parented to it. That parenting is what stops
the IK goal from moving with the arm it is driving.

## Bone collections

A Kinema rig sorts its bones into four collections:

| Collection | Contains | Visible on a new rig |
|---|---|---|
| `Kinema/FK` | Joint controls and the root | Yes |
| `Kinema/TCP` | The tool centre point marker | Yes |
| `Kinema/IK` | The IK target | No — until you add one |
| `Kinema/Mechanism` | Internal bones the rig needs to work | No |

`Kinema/Mechanism` stays hidden because those bones are not controls. Unhiding it is
useful for understanding how the rig is built, and unhelpful while animating.

## Attaching things to a link

A robot on its own is not a working cell. The gripper, the tool, the cable harness running
down the forearm — those are yours to add, and the [**Bones**
panel](../reference/sidebar.md#bones) is where they go: pick an object or a collection in a
bone's row and a copy rides that link.

It is a **linked copy**. It shares mesh and materials with what you picked, so the same
harness model can dress six links and still be edited in one place. What it does not share
is the transform: that belongs to the copy, and it is what positions it on the bone. So
scaling the original does not resize the attachment — scale the attachment.

### Why the offset is measured from the bone's head

This is the one place Blender's own behaviour and a robot's are at odds, and it is worth
knowing which you are looking at.

Blender parents to a bone's **tail**. That is sensible for a character, where a bone *is* a
limb and its far end is where the next thing goes. On a robot rig it is nowhere useful: a
Kinema bone sits at the **joint**, aligned to the axis it turns about, and its length is a
display choice, not a measurement. Parenting a gripper the ordinary way would put it a
handspan past the joint, and every offset typed afterwards would be measured from a frame
that means nothing.

So Kinema cancels the tail offset. An attachment's plain location, rotation and scale *is*
its offset from the joint — type it in the panel, drag it with <kbd>G</kbd>, keyframe it like
any other object channel.

!!! tip "A moved link mesh can be put back"
    The robot's own meshes are locked, precisely because a stray drag would otherwise be
    unrecoverable. If one does end up adrift, **Reset Meshes** in the Joints panel returns
    them; attachments are left alone.
