# Install

## Requirements

Blender **5.2 LTS or newer**. Nothing else — the solver, the robot catalog and every
file-format reader ship inside the add-on, and none of it is downloaded at runtime. Kinema
asks for one permission, **files**, to read the robot descriptions and meshes you point it
at.

!!! warning "Downloads are per-platform"
    Kinema bundles a compiled maths library, so there is a **separate build for each
    operating system**. Downloading the wrong one gives you an add-on that installs and
    then cannot solve. Check the filename before you drag it in.

| Your machine | File | Size |
|---|---|---|
| Windows (Intel/AMD 64-bit) | `kinema-0.3.0-windows_x64.zip` | ~117 MB |
| Linux (Intel/AMD 64-bit) | `kinema-0.3.0-linux_x64.zip` | ~136 MB |
| macOS (Apple Silicon) | `kinema-0.3.0-macos_arm64.zip` | ~100 MB |

Intel Macs are not currently built. If you need one, open an issue.

## Install the extension

Download the zip for your platform from the
[releases page](https://github.com/ebgenius/kinema/releases), then:

=== "Drag and drop"

    Drag the zip file from your file manager straight into the Blender window. Blender
    recognises an extension zip and opens its install dialog. Confirm, and you are done.

    This is the fastest route and works on all three platforms.

=== "Windows"

    1. **Edit → Preferences → Get Extensions**
    2. Click the dropdown arrow at the top right → **Install from Disk…**
    3. Select `kinema-0.3.0-windows_x64.zip`
    4. Kinema appears in the add-on list, already enabled

    !!! danger "Enable long paths first"
        The bundled maths library unpacks into very deeply nested folders. On a Windows
        profile without long-path support the install fails partway through with
        `WinError 206`. See [Troubleshooting](../troubleshooting.md#windows-winerror-206)
        for the one-line fix.

=== "macOS"

    1. **Blender → Settings → Get Extensions**
    2. Click the dropdown arrow at the top right → **Install from Disk…**
    3. Select `kinema-0.3.0-macos_arm64.zip`
    4. Kinema appears in the add-on list, already enabled

=== "Linux"

    1. **Edit → Preferences → Get Extensions**
    2. Click the dropdown arrow at the top right → **Install from Disk…**
    3. Select `kinema-0.3.0-linux_x64.zip`
    4. Kinema appears in the add-on list, already enabled

## Check it worked

Press <kbd>N</kbd> in the 3D viewport to open the sidebar. You should see a **Kinema**
tab down the right-hand edge.

Open it, expand the **Solver** panel at the bottom, and look for:

```
✓ PyRoki ready
```

That is the full solver, loaded and ready.

If it says **PyRoki unavailable** with an error underneath, the add-on still works — it
falls back to a simpler solver — but you are missing the good one. That usually means a
platform mismatch in the download. See
[Solver unavailable](../troubleshooting.md#solver-unavailable).

!!! tip "The first solve pauses. Once."
    The solver compiles itself the first time it runs on a given robot, which takes
    roughly 15 seconds behind a wait cursor. Every solve after that is a few milliseconds.
    This is normal and is explained in [Forward and inverse kinematics](../concepts/ik.md).

## Next

You have a working install. Now [load a robot and pose it](first-robot.md).

