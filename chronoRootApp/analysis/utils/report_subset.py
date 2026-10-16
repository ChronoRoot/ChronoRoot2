"""Which plants belong in one report folder.

The folder on disk is the record. subset.json says who was included.
report_status.txt says whether generation finished.
"""

import json
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


SUBSET_FILE = "subset.json"
STATUS_FILE = "report_status.txt"
STATUS_INCOMPLETE = "incomplete"
STATUS_COMPLETE = "complete"
STATUS_RUNNING = "running"


def process_is_running(pid):
    """True when this process still exists."""
    try:
        os.kill(int(pid), 0)
    except (ProcessLookupError, ValueError, TypeError):
        return False
    except PermissionError:
        return True
    return True


def running_status(pid):
    return f"{STATUS_RUNNING} {int(pid)}"


def read_report_status(folder):
    """Return complete, running for the status file in this folder."""
    path = os.path.join(folder, STATUS_FILE)
    if not os.path.isfile(path):
        return False, False
    try:
        with open(path, encoding="utf-8") as handle:
            text = handle.read().strip()
    except OSError:
        return False, False
    if text == STATUS_COMPLETE:
        return True, False
    if text.startswith(STATUS_RUNNING + " "):
        return False, process_is_running(text.split()[-1])
    return False, False


def write_subset_file(folder, subset):
    """Store this run's filter inside the report folder."""
    os.makedirs(folder, exist_ok=True)
    payload = {
        "experiments": subset.get("experiments"),
        "plates": subset.get("plates"),
        "extras": subset.get("extras"),
    }
    with open(os.path.join(folder, SUBSET_FILE), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def read_subset_file(folder):
    path = os.path.join(folder, SUBSET_FILE)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(loaded, dict):
        return None
    return {
        "experiments": _name_list(loaded.get("experiments")),
        "plates": _name_list(loaded.get("plates")),
        "extras": _name_list(loaded.get("extras")),
    }


def mark_report_status(folder, status):
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, STATUS_FILE), "w", encoding="utf-8") as handle:
        handle.write(status + "\n")


def mark_report_running(folder, pid):
    mark_report_status(folder, f"{STATUS_RUNNING} {int(pid)}")


def _report_process_alive(pid):
    """True when this pid is still the report script, not a reused process."""
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    cmdline_path = os.path.join("/proc", str(int(pid)), "cmdline")
    try:
        with open(cmdline_path, "rb") as handle:
            command = handle.read().replace(b"\x00", b" ").decode("utf-8", "replace")
    except OSError:
        return False
    return "3_generateReport.py" in command


def read_report_state(folder):
    """Return complete, running, or incomplete for one report folder."""
    path = os.path.join(folder, STATUS_FILE)
    if not os.path.isfile(path):
        temporal = os.path.join(folder, "data", "Temporal_Data.csv")
        if os.path.isfile(temporal):
            return "complete"
        return "incomplete"
    try:
        with open(path, encoding="utf-8") as handle:
            text = handle.read().strip()
    except OSError:
        return "incomplete"
    if text == STATUS_COMPLETE:
        return "complete"
    if text.startswith(STATUS_RUNNING):
        parts = text.split()
        pid = parts[1] if len(parts) > 1 else ""
        if _report_process_alive(pid):
            return "running"
    return "incomplete"


def list_reports(project_dir):
    """Report folders that exist. A missing folder is not a report."""
    from analysis.utils.report_utils import natural_key as natural_keys

    if not project_dir or not os.path.isdir(project_dir):
        return []
    try:
        entries = sorted(os.listdir(project_dir), key=natural_keys)
    except OSError:
        return []
    reports = []
    for entry in entries:
        folder = os.path.join(project_dir, entry)
        if not os.path.isdir(folder):
            continue
        status_path = os.path.join(folder, STATUS_FILE)
        temporal = os.path.join(folder, "data", "Temporal_Data.csv")
        subset_path = os.path.join(folder, SUBSET_FILE)
        note_path = os.path.join(folder, "Comparison.txt")
        has_status = os.path.isfile(status_path)
        if not has_status and not os.path.isfile(temporal) and not os.path.isfile(subset_path):
            continue
        complete, running = read_report_status(folder)
        if not has_status and not os.path.isfile(temporal):
            complete, running = False, False
        elif not has_status and os.path.isfile(temporal):
            complete, running = True, False
        description = ""
        if os.path.isfile(note_path):
            try:
                with open(note_path, encoding="utf-8") as handle:
                    description = handle.read().strip()
            except OSError:
                description = ""
        include = read_subset_file(folder) or {}
        if not description:
            description = subset_description(include) if include else "Compares all genotypes."
        if running:
            description = description.rstrip(".") + ". Running."
        elif not complete:
            description = description.rstrip(".") + ". Not finished. Generate again."
        reports.append({
            "name": entry,
            "description": description,
            "complete": complete,
            "running": running,
            "experiments": include.get("experiments"),
            "plates": include.get("plates"),
            "extras": include.get("extras"),
        })
    return reports


def apply_subset(conf, name):
    """Point this run at a report folder and read its filter from that folder."""
    conf = dict(conf or {})
    if not name:
        conf.pop("reportInclude", None)
        return conf
    folder_name = report_folder_name({"reportFolderName": name})
    conf["reportFolderName"] = folder_name
    folder = os.path.join(conf.get("MainFolder") or "", folder_name)
    subset = read_subset_file(folder)
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
        if not values:
            return ""
        if len(values) == 1:
            return values[0]
        if len(values) == 2:
            return f"{values[0]} and {values[1]}"
        return ", ".join(values[:-1]) + ", and " + values[-1]

    if experiments is not None and not experiments:
        return "No genotypes selected."
    if plates is not None and not plates:
        return "No plate condition selected."
    if extras is not None and not extras:
        return "No extra variable selected."

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
