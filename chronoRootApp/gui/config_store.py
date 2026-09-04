"""Project and global configuration persistence for the ChronoRoot GUI."""

import json
import os

from analysis.time_windows import parse_hour_list

APP_NAME = "chronoroot"
PROJECT_CONFIG_NAME = "project_config.json"
GLOBAL_CONFIG_DIR = os.path.expanduser(f"~/.config/{APP_NAME}")
GLOBAL_CONFIG_FILE = os.path.join(GLOBAL_CONFIG_DIR, "mainInterfaceConfig.json")

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


class ConfigStore:
  def config_value(self, data, key, default=None):
    if key in data:
      return data[key]
    for legacy_key, new_key in LEGACY_CONFIG_ALIASES.items():
      if new_key == key and legacy_key in data:
        return data[legacy_key]
    return default

  def build_payload(self, host):
    data = {}

    for field in [
        host.processingLimitField,
        host.reportProcessingLimitField,
        host.reportEmergenceDistanceField,
        host.captureIntervalField,
        host.everyXhourField,
        host.everyXhourFieldFourier,
        host.everyXhourFieldAngles,
        host.numComponentsFPCAField,
    ]:
      if field.text().isdigit():
        data[field.objectName()] = int(field.text())
      if field.text() == "":
        data[field.objectName()] = ""

    data.update({
        field.objectName(): field.text()
        for field in [
            host.plantIdentifier,
            host.videoField,
            host.projectField,
            host.plateConditionName,
            host.extraField,
            host.everyXhourField,
            host.everyXhourFieldFourier,
            host.everyXhourFieldAngles,
            host.numComponentsFPCAField,
            host.reportGenotypeAxisLabelField,
            host.reportPlateConditionAxisLabelField,
            host.reportExtraVariableAxisLabelField,
        ]
    })
    data.update({
        field.objectName(): field.isChecked()
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
            host.statsByGenotype,
            host.statsGenotypeByPlate,
            host.statsGenotypeByExtra,
            host.statsByPlateCondition,
            host.statsByExtraVariable,
            host.statsPlateWithinGenotype,
            host.statsExtraWithinGenotype,
        ]
    })

    data["daysConvexHull"] = host.daysConvexField.text()
    data["daysAngles"] = host.daysAnglesField.text()
    data["snapshotHoursConvex"] = parse_hour_list(host.daysConvexField.text())
    data["snapshotHoursAngles"] = parse_hour_list(host.daysAnglesField.text())
    for key in (
        'timeSyncMode', 'timeDurationHours', 'reportFolderName',
        'figureClockTicks', 'showFigureClockTicks', 'timeGroups', 'timePeriodSources',
    ):
      if hasattr(host, key):
        data[key] = getattr(host, key)

    data["rpi"] = host.rpiField.text()
    data["rpiField"] = host.rpiField.text()
    data["cam"] = host.cameraField.text()
    data["cameraField"] = host.cameraField.text()
    data["plant"] = host.plantField.text()
    data["plantField"] = host.plantField.text()
    data["Experiment"] = data["plantIdentifier"]
    data["PlateCondition"] = data["plateConditionName"]
    data["ExtraVariable"] = data["extraField"]
    data["genotypeAxisLabel"] = data.get("reportGenotypeAxisLabelField", "Genotype")
    data["plateConditionAxisLabel"] = data.get("reportPlateConditionAxisLabelField", "Plate condition")
    data["extraVariableLabel"] = data.get("reportExtraVariableAxisLabelField", "Run")
    data["Images"] = data["videoField"]
    data["processingLimit"] = data["processingLimitField"]
    data["timeStep"] = data["captureIntervalField"]
    data["MainFolder"] = data["projectField"]
    data["saveImages"] = data["saveImagesButton"]
    data["videoHasQR"] = data["videoHasQRbutton"]
    data["emergenceDistance"] = data["emergenceDistanceField"]

    for legacy_key, new_key in LEGACY_CONFIG_ALIASES.items():
      if new_key in data:
        data[legacy_key] = data[new_key]

    if data["processingLimit"] != "":
      data["Limit"] = int(data["processingLimit"] * 24 * 60 / int(data["timeStep"]))
    else:
      data["Limit"] = 0

    data["knownDistance"] = host.knownDistanceField.text()
    data["pixelDistance"] = host.pixelDistanceField.text()
    return data

  def apply_payload(self, host, data):
    for field in [
        host.rpiField,
        host.cameraField,
        host.plantField,
        host.processingLimitField,
        host.plateConditionName,
        host.extraField,
        host.reportProjectField,
        host.reportProcessingLimitField,
        host.reportEmergenceDistanceField,
        host.captureIntervalField,
        host.reportCaptureIntervalField,
        host.everyXhourField,
        host.everyXhourFieldFourier,
        host.everyXhourFieldAngles,
        host.numComponentsFPCAField,
        host.reportGenotypeAxisLabelField,
        host.reportPlateConditionAxisLabelField,
        host.reportExtraVariableAxisLabelField,
    ]:
      val = self.config_value(data, field.objectName())
      if val is not None:
        field.setText(str(val))
      elif field.objectName() == 'reportGenotypeAxisLabelField' and 'genotypeAxisLabel' in data:
        field.setText(str(data['genotypeAxisLabel']))
      elif field.objectName() == 'reportPlateConditionAxisLabelField' and 'plateConditionAxisLabel' in data:
        field.setText(str(data['plateConditionAxisLabel']))
      elif field.objectName() == 'reportExtraVariableAxisLabelField' and 'extraVariableLabel' in data:
        field.setText(str(data['extraVariableLabel']))

    for field in [
        host.plantIdentifier,
        host.videoField,
        host.projectField
    ]:
      val = self.config_value(data, field.objectName())
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
        host.statsByGenotype,
        host.statsGenotypeByPlate,
        host.statsGenotypeByExtra,
        host.statsByPlateCondition,
        host.statsByExtraVariable,
        host.statsPlateWithinGenotype,
        host.statsExtraWithinGenotype,
    ]:
      if field.objectName() in data:
        field.setChecked(data[field.objectName()])

    if "knownDistance" in data:
      host.knownDistanceField.setText(str(data["knownDistance"]))
    if "pixelDistance" in data:
      host.pixelDistanceField.setText(str(data["pixelDistance"]))
    _apply_snapshot_hours_field(
        host.daysConvexField, data,
        'snapshotHoursConvex', 'daysConvexHull',
    )
    _apply_snapshot_hours_field(
        host.daysAnglesField, data,
        'snapshotHoursAngles', 'daysAngles',
    )
    for key, default in (
        ('timeSyncMode', 'clock'),
        ('timeDurationHours', None),
        ('reportFolderName', 'Report'),
        ('figureClockTicks', ['00:00']),
        ('showFigureClockTicks', None),
        ('snapshotHours', None),
        ('snapshotHoursConvex', None),
        ('snapshotHoursAngles', None),
        ('timeGroups', []),
        ('timePeriodSources', []),
    ):
      if key in data:
        setattr(host, key, data[key])
      elif key == 'timePeriodSources':
        setattr(host, key, [])
      elif key == 'showFigureClockTicks':
        mode = getattr(host, 'timeSyncMode', 'clock')
        setattr(host, key, mode != 'anchor')
      elif not hasattr(host, key):
        setattr(host, key, default)
    if hasattr(host, '_sync_snapshot_hour_fields'):
      host._sync_snapshot_hour_fields()

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
      with open(GLOBAL_CONFIG_FILE, "w") as f:
        json.dump(data, f, indent=4)
    except Exception as e:
      print(f"Error saving global config: {e}")

    project_path = host.projectField.text()
    if project_path and os.path.isdir(project_path):
      try:
        proj_cfg_path = os.path.join(project_path, PROJECT_CONFIG_NAME)
        with open(proj_cfg_path, "w") as f:
          json.dump(data, f, indent=4)
      except Exception as e:
        print(f"Error saving project config: {e}")

  def load(self, host):
    json_path = self.resolve_config_path(host)
    if not json_path:
      return
    try:
      with open(json_path, "r") as f:
        data = json.load(f)
      self.apply_payload(host, data)
    except Exception as e:
      print(f"Error loading config: {e}")

  def apply_file(self, host, json_path):
    with open(json_path, "r") as f:
      data = json.load(f)
    self.apply_payload(host, data)
