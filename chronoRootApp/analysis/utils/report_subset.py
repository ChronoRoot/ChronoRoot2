"""Which plants belong in one named report folder.

A subset stores the genotype, plate-condition, and extra-variable values to
keep. A missing list means every value of that factor. The report script reads
the saved subset by name and does not change the project's default folder.
"""

import os

from analysis.time_windows import report_folder_name
from analysis.utils.fileUtilities import (
    convertFromPathSafe,
    load_result_metadata,
    normalize_factor_value,
)

FULL_REPORT_NAME = "Report"


def _name_list(value):
    """A list of factor values, or None when every value is included."""
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        return None
    return [str(item) for item in value]


def normalize_subset(entry):
    """One subset record, or None when the entry is not usable."""
    if not isinstance(entry, dict):
        return None
    name = report_folder_name({"reportFolderName": entry.get("name") or ""})
    return {
        "name": name,
        "experiments": _name_list(entry.get("experiments")),
        "plates": _name_list(entry.get("plates")),
        "extras": _name_list(entry.get("extras")),
    }


def normalize_subset_list(raw):
    """Drop broken entries and keep the first record for each folder name."""
    if not isinstance(raw, list):
        return []
    subsets = []
    seen = set()
    for entry in raw:
        subset = normalize_subset(entry)
        if subset is None or subset["name"] in seen:
            continue
        seen.add(subset["name"])
        subsets.append(subset)
    return subsets


def find_subset(conf, name):
    """The saved subset with this folder name, or None."""
    wanted = report_folder_name({"reportFolderName": name or ""})
    for subset in normalize_subset_list((conf or {}).get("reportSubsets")):
        if subset["name"] == wanted:
            return subset
    return None


def apply_subset(conf, name):
    """Point this run at the subset folder without editing the saved config.

    An unknown name still writes that folder and includes every plant, which
    is the same as the full report under a different directory.
    """
    conf = dict(conf or {})
    if not name:
        conf.pop("reportInclude", None)
        return conf
    folder = report_folder_name({"reportFolderName": name})
    conf["reportFolderName"] = folder
    subset = find_subset(conf, folder)
    if subset is None:
        conf.pop("reportInclude", None)
        return conf
    conf["reportInclude"] = {
        "experiments": subset["experiments"],
        "plates": subset["plates"],
        "extras": subset["extras"],
    }
    return conf


def _value_kept(allowed, value, normalize=False):
    if allowed is None:
        return True
    if normalize:
        allowed_values = {normalize_factor_value(item) for item in allowed}
        return normalize_factor_value(value) in allowed_values
    return str(value) in {str(item) for item in allowed}


def plant_in_report(conf, experiment, plate, extra):
    """True when this plant's factors are part of the active subset."""
    include = (conf or {}).get("reportInclude")
    if not include:
        return True
    if not _value_kept(include.get("experiments"), experiment):
        return False
    if not _value_kept(include.get("plates"), plate, normalize=True):
        return False
    if not _value_kept(include.get("extras"), extra, normalize=True):
        return False
    return True


def experiment_in_report(conf, experiment):
    """True when this genotype is part of the active subset."""
    include = (conf or {}).get("reportInclude")
    if not include:
        return True
    return _value_kept(include.get("experiments"), experiment)


def experiment_name_from_result(result_dir):
    """Genotype folder sitting under Analysis/, as shown in the report."""
    parts = os.path.abspath(result_dir or "").split(os.sep)
    if "Analysis" not in parts:
        return ""
    index = parts.index("Analysis")
    if index + 1 >= len(parts):
        return ""
    return convertFromPathSafe(parts[index + 1])


def subset_description(subset):
    """One sentence naming who this report compares."""
    subset = subset or {}
    experiments = subset.get("experiments")
    plates = subset.get("plates")
    extras = subset.get("extras")
    if experiments is None and plates is None and extras is None:
        return "Compares all genotypes."

    def phrase(values):
        values = [str(value) for value in values]
        if len(values) == 1:
            return values[0]
        if len(values) == 2:
            return f"{values[0]} and {values[1]}"
        return ", ".join(values[:-1]) + ", and " + values[-1]

    parts = []
    if experiments is None:
        parts.append("all genotypes")
    else:
        parts.append(phrase(experiments))
    if plates is not None:
        parts.append("plate " + phrase(plates))
    if extras is not None:
        parts.append(phrase(extras))
    return "Compares " + ", ".join(parts) + "."


def write_comparison_note(conf):
    """Record the comparison in the report folder."""
    from analysis.utils.report_paths import report_root

    include = (conf or {}).get("reportInclude")
    if include:
        text = subset_description({
            "experiments": include.get("experiments"),
            "plates": include.get("plates"),
            "extras": include.get("extras"),
        })
    else:
        text = "Compares all genotypes."
    folder = report_root(conf)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "Comparison.txt"), "w", encoding="utf-8") as handle:
        handle.write(text + "\n")


def result_dir_in_report(conf, result_dir):
    """True when the latest result folder belongs in the active subset."""
    if not (conf or {}).get("reportInclude"):
        return True
    meta = load_result_metadata(result_dir) or {}
    experiment = str(meta.get("Experiment") or "").strip() or experiment_name_from_result(result_dir)
    return plant_in_report(
        conf,
        experiment,
        meta.get("PlateCondition", ""),
        meta.get("ExtraVariable", ""),
    )
