# A small kinema_catalog with its descriptions, for importing from a Catalog Folder (#105)

Laid out as a clone: `catalog.json` at the top, and each entry's repository under
`src/`, as kinema_catalog keeps its submodules.

- `fixture_arm` includes a package from another repository, `common_repo`. It
  imports only when the whole of `src/` is searched.
- `fixture_named` needs the xacro arguments its entry names.
- `fixture_tool`, an end effector, names its mesh relative to its repository's
  top, which its `mesh_base` says.
- `fixture_absent`'s repository isn't here: it stands for a submodule not yet
  cloned.

The tests copy it to a temporary directory and add the `.git` a clone and its
submodules have, which git won't keep in a repository.

The descriptions are copies of the ones in `../ros_ws`.
