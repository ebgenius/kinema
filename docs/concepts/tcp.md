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

**It is what IK aims at by default.** When you [add an IK target](../tutorials/animate-ik.md),
the target is created at the TCP, and the solver's job is to put the TCP where the target is.
No TCP means nothing to aim, and Add IK Target refuses.

Only by default, though: the **Bones** panel can point the solver at any joint bone instead,
and that choice is keyframable. See
[aiming somewhere else](../tutorials/animate-ik.md#aiming-at-a-different-bone).

**It is what you measure from.** The Tool Centre Point panel reports its position and its
orientation live — useful when you need the tool at a specific pose rather than somewhere
that merely looks right.

**It is what you hang a tool on.** Pick the TCP's row in the **Bones** panel and attach a
torch mesh, an emitter, a light or a camera; it rides the working point exactly, with an
offset you can type. See
[attaching things to a link](links-and-joints.md#attaching-things-to-a-link).

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

The Tool Centre Point panel takes a **Parent Bone** and a **Tool Offset**: pick the joint the
tool is bolted to, type where the working point sits relative to it, and press **Update TCP**.
No mode change, and nothing to move by hand.

Only joint bones can host the TCP. They are the ones carrying a link frame for the offset to
be measured in — and it keeps the marker off the IK control, which would otherwise leave it
riding the goal it is supposed to define.

### The offset is measured from the flange

Not from the bone. A joint bone's axes are aligned to its **joint axis**, because that is
what makes it a one-degree-of-freedom control; the *link* frame — the flange face, with its
Z pointing out of it — is a different frame, and it is the one a tool is specified against.

So `Z = 0.15` means 150 mm out of the flange, not 150 mm up. The three angles are roll, pitch
and yaw about fixed X, Y and Z, which is what URDF's `<origin rpy="…">` means, so a tool
transform copied out of a description can be typed in unchanged.

!!! info "A fresh import already has an offset, and that is correct"
    The tool frame is the description's deepest link, which normally sits behind one or more
    **fixed** joints from the last actuated one. Fixed joints get no bone, so nothing else on
    the rig shows that distance — the offset field is where it becomes visible. **Reset**
    clears it to zero, which puts the TCP on the flange itself rather than back where the
    importer had it.

!!! warning "Place the TCP before adding an IK target"
    The IK target is created at the TCP's location. Moving the TCP afterwards leaves the
    target somewhere that is no longer the working point.

    If you have already added the target, use **Snap to Tool** to bring it back into
    agreement.

**Move TCP to Active Bone** is still there for the older habit: it places the TCP on
whichever bone is active in Pose or Edit mode. It needs one, and says so if there is none.

## The workflow

For a robot with a tool on the end:

1. Import the robot
2. Attach the tool in the [**Bones** panel](../reference/sidebar.md#bones), on the bone it
   is bolted to
3. Set **Parent Bone** to that same bone
4. Type the **Tool Offset** — where the working point sits relative to the flange
5. **Update TCP**
6. Add an IK target — it appears at the corrected point
7. Animate

Getting this order right saves re-doing step 6.
