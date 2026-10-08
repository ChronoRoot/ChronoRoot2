"""Project and global configuration persistence for the ChronoRoot GUI."""

import os

from analysis.time_windows import parse_hour_list
from analysis.utils.metadata_schema import (
    dump_json,
    load_json,
    strip_roi_keys,
    widget_value,
)

APP_NAME = "chronoroot"
PROJECT_CONFIG_NAME = "project_config.json"
GLOBAL_CONFIG_DIR = os.path.expanduser(f"~/.config/{APP_NAME}")
GLOBAL_CONFIG_FILE = os.path.join(GLOBAL_CONFIG_DIR, "mainInterfaceConfig.json")

# Old widget names still accepted on load. Never written.
LEGACY_CONFIG_ALIASES = {
    "projectField_2": "reportProjectField",
    "processingLimitField_3": "reportProcessingLimitField",
    "captureIntervalField_3": "reportCaptureIntervalField",
    "PostProcessButton2": "reportPostProcessButton",
    "loadLastConfig2": "reportLoadConfigButton",
    "saveButton_2": "reportSaveConfigButton",
    "loadProject_2": "reportSelectProjectButton",
}

os.makedirs(GLOBAL_CONFIG_DIR, exist_ok=True)


def _apply_snapshot_hours_field(field, data, specific_key, legacy_days_key):
    if specific_key in data:
        hours = parse_hour_list(data.get(specific_key))
        field.setText(','.join(str(h) for h in hours))
        return
    if data.get('snapshotHours') not in (None, ''):
        hours = parse_hour_list(data.get('snapshotHours'))
        if hours:
            field.setText(','.join(str(h) for h in hours))
            return
    if legacy_days_key in data and data.get(legacy_days_key) not in (None,):
        hours = parse_hour_list(data.get(legacy_days_key))
        field.setText(','.join(str(h) for h in hours) if hours else str(data.get(legacy_days_key)))


def _numeric_or_text(text):
    if text == "":
        return ""
    if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
        return int(text)
    try:
        return float(text)
    except ValueError:
        return text


class ConfigStore:
  def config_value(self, data, key, default=None):
    value = widget_value(data, key, None)
    if value is not None:
      return value
    for legacy_key, new_key in LEGACY_CONFIG_ALIASES.items():
      if new_key == key and legacy_key in data:
        return data[legacy_key]
    return default

  def build_payload(self, host):
    data = {
        "Experiment": host.plantIdentifier.text(),
        "MainFolder": host.projectField.text(),
        "Images": host.videoField.text(),
        "rpi": host.rpiField.text(),
        "cam": host.cameraField.text(),
        "plant": host.plantField.text(),
        "PlateCondition": host.plateConditionName.text(),
        "ExtraVariable": host.extraField.text(),
        "processingLimit": _numeric_or_text(host.processingLimitField.text()),
        "timeStep": _numeric_or_text(host.captureIntervalField.text()),
        "saveImages": host.saveImagesButton.isChecked(),
        "videoHasQR": host.videoHasQRbutton.isChecked(),
        "knownDistance": host.knownDistanceField.text(),
        "pixelDistance": host.pixelDistanceField.text(),
        "emergenceDistance": _numeric_or_text(host.analysisEmergenceDistanceField.text()
                                                or host.reportEmergenceDistanceField.text()),
        "saveImagesConvex": host.saveImagesConvex.isChecked(),
        "doConvex": host.doConvex.isChecked(),
        "doFourier": host.doFourier.isChecked(),
        "doLateralAngles": host.doLateralAngles.isChecked(),
        "doFPCA": host.doFPCA.isChecked(),
        "normFPCA": host.normFPCA.isChecked(),
        "averagePerPlantStats": host.averagePerPlantStats.isChecked(),
        "measureRelativeToInitial": host.measureRelativeToInitial.isChecked(),
        "statsByGenotype": host.statsByGenotype.isChecked(),
        "statsGenotypeByPlate": host.statsGenotypeByPlate.isChecked(),
        "statsGenotypeByExtra": host.statsGenotypeByExtra.isChecked(),
        "statsByPlateCondition": host.statsByPlateCondition.isChecked(),
        "statsByExtraVariable": host.statsByExtraVariable.isChecked(),
        "statsPlateWithinGenotype": host.statsPlateWithinGenotype.isChecked(),
        "statsExtraWithinGenotype": host.statsExtraWithinGenotype.isChecked(),
        "everyXhourField": _numeric_or_text(host.everyXhourField.text()),
        "everyXhourFieldFourier": _numeric_or_text(host.everyXhourFieldFourier.text()),
        "everyXhourFieldAngles": _numeric_or_text(host.everyXhourFieldAngles.text()),
        "numComponentsFPCAField": _numeric_or_text(host.numComponentsFPCAField.text()),
        "genotypeAxisLabel": host.reportGenotypeAxisLabelField.text(),
        "plateConditionAxisLabel": host.reportPlateConditionAxisLabelField.text(),
        "extraVariableLabel": host.reportExtraVariableAxisLabelField.text(),
        "snapshotHoursConvex": parse_hour_list(host.daysConvexField.text()),
        "snapshotHoursAngles": parse_hour_list(host.daysAnglesField.text()),
    }
    if getattr(host, 'advanced_group', None) is not None:
        data['advancedComparisonModes'] = host.advanced_group.isChecked()
    for key in (
        'timeSyncMode', 'timeDurationHours', 'reportFolderName',
        'figureClockTicks', 'showFigureClockTicks', 'timeGroups', 'timePeriodSources',
    ):
      if hasattr(host, key):
        data[key] = getattr(host, key)
    return strip_roi_keys(data)

  def apply_payload(self, host, data):
    data = data or {}
    text_fields = [
        (host.rpiField, 'rpiField'),
        (host.cameraField, 'cameraField'),
        (host.plantField, 'plantField'),
        (host.processingLimitField, 'processingLimitField'),
        (host.plateConditionName, 'plateConditionName'),
        (host.extraField, 'extraField'),
        (host.reportProjectField, 'reportProjectField'),
        (host.reportProcessingLimitField, 'reportProcessingLimitField'),
        (host.reportEmergenceDistanceField, 'emergenceDistanceField'),
        (host.analysisEmergenceDistanceField, 'analysisEmergenceDistanceField'),
        (host.captureIntervalField, 'captureIntervalField'),
        (host.reportCaptureIntervalField, 'reportCaptureIntervalField'),
        (host.everyXhourField, 'everyXhourField'),
        (host.everyXhourFieldFourier, 'everyXhourFieldFourier'),
        (host.everyXhourFieldAngles, 'everyXhourFieldAngles'),
        (host.numComponentsFPCAField, 'numComponentsFPCAField'),
        (host.reportGenotypeAxisLabelField, 'reportGenotypeAxisLabelField'),
        (host.reportPlateConditionAxisLabelField, 'reportPlateConditionAxisLabelField'),
        (host.reportExtraVariableAxisLabelField, 'reportExtraVariableAxisLabelField'),
        (host.plantIdentifier, 'plantIdentifier'),
        (host.videoField, 'videoField'),
        (host.projectField, 'projectField'),
    ]
    for field, object_name in text_fields:
      val = self.config_value(data, object_name)
      if val is not None:
        field.setText(str(val))

    for field in [
        host.saveImagesButton,
        host.videoHasQRbutton,
        host.saveImagesConvex,
        host.doConvex,
        host.doFourier,
        host.doLateralAngles,
        host.doFPCA,
        host.normFPCA,
        host.averagePerPlantStats,
        host.measureRelativeToInitial,
        host.statsByGenotype,
        host.statsGenotypeByPlate,
        host.statsGenotypeByExtra,
        host.statsByPlateCondition,
        host.statsByExtraVariable,
        host.statsPlateWithinGenotype,
        host.statsExtraWithinGenotype,
    ]:
      val = self.config_value(data, field.objectName())
      if val is not None:
        field.setChecked(bool(val))

    if getattr(host, 'advanced_group', None) is not None:
      val = self.config_value(data, 'advancedComparisonModes')
      if val is not None:
        host.advanced_group.setChecked(bool(val))

    known = self.config_value(data, 'knownDistance')
    if known is not None:
      host.knownDistanceField.setText(str(known))
    pixel = self.config_value(data, 'pixelDistance')
    if pixel is not None:
      host.pixelDistanceField.setText(str(pixel))
    _apply_snapshot_hours_field(
        host.daysConvexField, data,
        'snapshotHoursConvex', 'daysConvexHull',
    )
    _apply_snapshot_hours_field(
        host.daysAnglesField, data,
        'snapshotHoursAngles', 'daysAngles',
    )
    for key, default in (
        ('reportFolderName', 'Report'),
        ('snapshotHours', None),
        ('snapshotHoursConvex', None),
        ('snapshotHoursAngles', None),
    ):
      if key in data:
        setattr(host, key, data[key])
      elif not hasattr(host, key):
        setattr(host, key, default)
    # Period keys are always replaced. Leaving them in place when a file
    # omitted them kept the previous project's windows on the window.
    self.apply_analysis_period(host, data)
    if hasattr(host, '_sync_snapshot_hour_fields'):
      host._sync_snapshot_hour_fields()

  def apply_analysis_period(self, host, data):
    """Copy one project's analysis period onto the window.

    The period is the sync mode, duration, clock ticks, time groups, and the
    acquisition fingerprint (timePeriodSources). It is stored on the window and
    also inside each project's project_config.json. Callers must pass that
    project's file, or an empty dict when the project has no saved period.

    A missing key is a reset, not "keep whatever was already there". The old
    loader only wrote these attributes the first time, so after a project
    change the previous groups were still what Set analysis period displayed.
    """
    data = data if isinstance(data, dict) else {}

    mode = data.get('timeSyncMode') if 'timeSyncMode' in data else 'clock'
    if mode not in ('clock', 'anchor'):
      mode = 'clock'
    host.timeSyncMode = mode

    duration = data.get('timeDurationHours') if 'timeDurationHours' in data else None
    if duration == '':
      duration = None
    host.timeDurationHours = duration

    ticks = data.get('figureClockTicks') if 'figureClockTicks' in data else ['00:00']
    if isinstance(ticks, str):
      ticks = [part.strip() for part in ticks.replace(';', ',').split(',') if part.strip()]
    elif isinstance(ticks, (list, tuple)):
      ticks = [str(part) for part in ticks]
    else:
      ticks = ['00:00']
    host.figureClockTicks = ticks or ['00:00']

    groups = data.get('timeGroups') if 'timeGroups' in data else []
    host.timeGroups = list(groups) if isinstance(groups, list) else []

    sources = data.get('timePeriodSources') if 'timePeriodSources' in data else []
    if isinstance(sources, (list, tuple)):
      host.timePeriodSources = [str(item) for item in sources]
    else:
      host.timePeriodSources = []

    if 'showFigureClockTicks' in data and data.get('showFigureClockTicks') is not None:
      host.showFigureClockTicks = bool(data.get('showFigureClockTicks'))
    else:
      host.showFigureClockTicks = mode != 'anchor'

  def resolve_config_path(self, host):
    project_cfg = os.path.join(host.projectField.text(), PROJECT_CONFIG_NAME)
    if os.path.exists(project_cfg):
      return project_cfg
    if os.path.exists(GLOBAL_CONFIG_FILE):
      return GLOBAL_CONFIG_FILE
    return None

  def save(self, host):
    data = self.build_payload(host)
    try:
      dump_json(GLOBAL_CONFIG_FILE, data)
    except Exception as e:
      print(f"Error saving global config: {e}")

    project_path = host.projectField.text()
    if project_path and os.path.isdir(project_path):
      try:
        dump_json(os.path.join(project_path, PROJECT_CONFIG_NAME), data)
      except Exception as e:
        print(f"Error saving project config: {e}")

  def load(self, host):
    json_path = self.resolve_config_path(host)
    if not json_path:
      return
    try:
      self.apply_payload(host, load_json(json_path))
    except Exception as e:
      print(f"Error loading config: {e}")

  def apply_file(self, host, json_path):
    self.apply_payload(host, load_json(json_path))
