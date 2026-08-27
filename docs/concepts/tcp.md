# The tool centre point

The **tool centre point** — TCP — is the spot on the robot that actually does the work.
The tip of the welding torch. The centre of the gripper's grasp. The point of the pen.

It is the single most important frame on the machine, because every task is defined in
terms of it. "Put the weld here" means "put the TCP here". Nobody specifies a robot task
in terms of where its elbow goes.

## Why it needs to exist separately

A robot's last joint is somewhere inside the wrist. The thing you care about is several
centimetres further out, at the end of whatever tool is bolted on — and that offset
changes every time the tool changes.

So the TCP is a marker: a frame that rides on the end of the chain at a defined offset,
representing the working point rather than the last mechanical joint.

In Kinema it is a bone in the `Kinema/TCP` collection, created automatically on import
unless you turn **Create TCP** off.

> 📷 *Screenshot: the TCP marker at the end of a robot arm.*

## What it is used for

**It is what IK aims.** When you [add an IK target](../tutorials/animate-ik.md), the
target is created at the TCP, and the solver's job is to put the TCP where the target is.
No TCP means nothing to aim, and Add IK Target refuses.

**It is what you measure from.** The Tool Centre Point panel reports its position live —
useful when you need the tool at a specific coordinate rather than somewhere that merely
looks right.

**It is what you parent to.** Attach a torch mesh, a particle emitter, a light or a camera
to the TCP bone and it follows the working point exactly.

## Orientation matters too

The TCP has a direction, not just a position. Kinema aligns it so its **+Z points along
the tool's approach direction** — the way the tool is "looking", the direction it would
travel to touch something.

This follows the convention used throughout robotics, so a TCP frame in Blender means the
same thing it means in the robot's own documentation.

It matters in practice because IK solves for orientation as well as position. Rotating the
IK target rotates the tool. If you are animating a gripper approaching an object, the
approach *direction* is usually as important as the point.

## Moving the TCP

The default TCP sits at the end of the chain, which is right for a bare robot and wrong as
soon as a real tool is attached — you want the working point of *that tool*.

To move it: enter Pose Mode, select the bone the tool is mounted on, and click **Move TCP
to Active Bone** in the Tool Centre Point panel.

If the rig has no TCP at all, the same button reads **Create TCP from Active Bone**.

!!! warning "Move the TCP before adding an IK target"
    The IK target is created at the TCP's location. Moving the TCP afterwards leaves the
    target somewhere that is no longer the working point.

    If you have already added the target, use **Snap to Tool** to bring it back into
    agreement.

!!! tip "Select a joint bone, not the TCP itself"
    Kinema will refuse and tell you if you try to place the TCP on itself.

## The workflow

For a robot with a tool on the end:

1. Import the robot
2. Import or model the tool, and parent it to the last joint bone
3. Select the bone the tool is mounted on
4. **Move TCP to Active Bone**
5. Add an IK target — it appears at the corrected point
6. Animate

Getting this order right saves re-doing step 5.
