# A small kinema_catalog, for the Catalog Folder tests (#105)

- `clone/` is laid out as a clone: one `catalog.json` at the top.
- `zips/` is laid out as family release zips unpacked side by side: one
  `catalog.json` in each directory.

The entries are copied from kinema_catalog's `catalog.json`, except
`gripper_2f`, which stands in for an end effector. The description files they
name aren't here.

`catalog.schema.json` is a copy of kinema_catalog's. `test_catalog_folder.py`
holds Kinema's field checks to it, so refresh it when the schema changes.
