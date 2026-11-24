"""Edit Genotype / plate / extra labels for one analyzed plant slot."""

import os
import shutil

from .fileUtilities import (
    convertFromPathSafe,
    convertToPathSafe,
    get_latest_result_dir,
    list_result_dirs,
    load_result_metadata,
)
from .metadata_schema import dump_json, load_json
from .remap_identifiers import patch_metadata_dict, remove_empty_dirs


class EditPlantLabelsError(Exception):
    """Raised when a per-plant label edit cannot be applied safely."""


def plant_slot_parts(plant_slot):
    """Return (analysis_root, experiment_readable, rpi, cam, plant) from a slot path."""
    plant_slot = os.path.abspath(plant_slot)
    cam_dir = os.path.dirname(plant_slot)
    rpi_dir = os.path.dirname(cam_dir)
    exp_dir = os.path.dirname(rpi_dir)
    analysis_root = os.path.dirname(exp_dir)
    if os.path.basename(analysis_root) != "Analysis":
        raise EditPlantLabelsError(
            "This path is not a plant slot under Analysis/<identifier>/…."
        )
    return (
        analysis_root,
        convertFromPathSafe(os.path.basename(exp_dir)),
        os.path.basename(rpi_dir),
        os.path.basename(cam_dir),
        os.path.basename(plant_slot),
    )


def current_plant_labels(plant_slot):
    """
    Labels and hardware identity for the edit dialog.

    Returns dict with experiment, plate_condition, extra_variable,
    rpi, cam, plant, and result_dir (latest Results_* or "").
    """
    plant_slot = os.path.abspath(plant_slot)
    if not os.path.isdir(plant_slot):
        raise EditPlantLabelsError("Plant folder does not exist.")

    _analysis, experiment_from_path, rpi, cam, plant = plant_slot_parts(plant_slot)
    result_dir = get_latest_result_dir(plant_slot) or ""
    if not result_dir:
        raise EditPlantLabelsError(
            "This plant has no Results_* folder yet. Analyze it before editing labels."
        )

    meta = load_result_metadata(result_dir)
    experiment = meta.get("Experiment")
    if experiment is None or str(experiment).strip() == "":
        experiment = experiment_from_path
    else:
        experiment = str(experiment)

    return {
        "experiment": experiment,
        "plate_condition": "" if meta.get("PlateCondition") is None else str(meta.get("PlateCondition")),
        "extra_variable": "" if meta.get("ExtraVariable") is None else str(meta.get("ExtraVariable")),
        "rpi": rpi,
        "cam": cam,
        "plant": plant,
        "result_dir": result_dir,
        "plant_slot": plant_slot,
    }


def _strip_label(value):
    if value is None:
        return ""
    return str(value).strip()


def _patch_factor_fields(meta, experiment, plate_condition, extra_variable, old_experiment=None):
    """Update label fields (and Experiment aliases/paths when the genotype changes)."""
    old_name = old_experiment if old_experiment is not None else meta.get("Experiment")
    old_safe = convertToPathSafe(old_name) if old_name is not None else None
    new_safe = convertToPathSafe(experiment)
    patch_metadata_dict(meta, experiment, old_name, old_safe, new_safe)
    meta["PlateCondition"] = plate_condition
    meta["ExtraVariable"] = extra_variable
    # plate/extra aliases used by older configs
    if "plateConditionName" in meta:
        meta["plateConditionName"] = plate_condition
    if "extraField" in meta:
        meta["extraField"] = extra_variable


def _update_all_results_metadata(
    plant_slot, experiment, plate_condition, extra_variable, old_experiment=None,
):
    updated = 0
    for result_dir in list_result_dirs(plant_slot):
        meta_path = os.path.join(result_dir, "metadata.json")
        if not os.path.isfile(meta_path):
            continue
        try:
            meta = load_json(meta_path)
        except (OSError, ValueError):
            continue
        _patch_factor_fields(
            meta, experiment, plate_condition, extra_variable, old_experiment=old_experiment,
        )
        dump_json(meta_path, meta)
        updated += 1
    if updated == 0:
        raise EditPlantLabelsError(
            "No Results_*/metadata.json files found under this plant."
        )


def apply_plant_label_edit(
    plant_slot, *, experiment, plate_condition, extra_variable,
):
    """
    Apply label edits to one plant slot.

    Returns the (possibly moved) absolute plant slot path.
    """
    plant_slot = os.path.abspath(plant_slot)
    current = current_plant_labels(plant_slot)

    new_experiment = _strip_label(experiment)
    new_plate = _strip_label(plate_condition)
    new_extra = _strip_label(extra_variable)
    if not new_experiment:
        raise EditPlantLabelsError("Genotype / identifier cannot be empty.")

    old_experiment = current["experiment"]
    old_plate = _strip_label(current["plate_condition"])
    old_extra = _strip_label(current["extra_variable"])

    if (
        new_experiment == old_experiment
        and new_plate == old_plate
        and new_extra == old_extra
    ):
        return plant_slot

    analysis_root, _old_exp, rpi, cam, plant = plant_slot_parts(plant_slot)

    if new_experiment == old_experiment:
        _update_all_results_metadata(
            plant_slot, new_experiment, new_plate, new_extra, old_experiment=old_experiment,
        )
        return plant_slot

    dest_slot = os.path.join(
        analysis_root, convertToPathSafe(new_experiment), rpi, cam, plant,
    )
    if os.path.exists(dest_slot):
        raise EditPlantLabelsError(
            f"Cannot move plant: destination already exists:\n{dest_slot}"
        )

    old_parents = []
    parent = os.path.dirname(plant_slot)
    while os.path.basename(parent) != "Analysis" and parent != analysis_root:
        old_parents.append(parent)
        next_parent = os.path.dirname(parent)
        if next_parent == parent:
            break
        parent = next_parent

    os.makedirs(os.path.dirname(dest_slot), exist_ok=True)
    shutil.move(plant_slot, dest_slot)
    for folder in old_parents:
        remove_empty_dirs(folder)

    _update_all_results_metadata(
        dest_slot, new_experiment, new_plate, new_extra, old_experiment=old_experiment,
    )
    return dest_slot
