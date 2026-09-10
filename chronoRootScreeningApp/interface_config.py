"""Save/load Analysis tab configuration (session and process configs)."""

import json
import os
from typing import Any, Dict, List, Tuple

CONFIG_KIND = "chrono_root_screening_setup"
CONFIG_VERSION = 1
SETUP_FILE_FILTER = 'Setup files (*config*.json)'

KNOWN_WRONG_FILES = {
    'metadata.json': (
        'This file tracks analysis progress (status, start time). '
        'It is not an analysis setup file.'
    ),
    'group_info.json': (
        'This file stores group results from a completed run. '
        'It cannot restore the Analysis tab.'
    ),
    'group_rois.json': (
        'This file stores ROI rectangles only. '
        'Use Process to set ROIs again, or load a setup config instead.'
    ),
}

GENERIC_SETUP_FAILURE = (
    'This is not an analysis setup file.\n\n'
    "Use a file with 'config' in the name, such as:\n"
    '• process_config.json (in your analysis folder, created when you Process)\n'
    '• A file you exported with Export Setup to File\n\n'
    'Do not use: metadata.json, group_info.json, or group_rois.json.'
)


def is_setup_filename(path: str) -> bool:
    return 'config' in os.path.basename(path).lower()


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ('true', '1', 'yes')
    return bool(value)


def _looks_like_group_rois(data: Any) -> bool:
    if not isinstance(data, dict) or not data:
        return False
    if 'config_version' in data or 'config_kind' in data:
        return False
    return all(
        isinstance(v, (list, tuple)) and len(v) == 4
        for v in data.values()
    )


def _looks_like_metadata(data: Any) -> bool:
    if not isinstance(data, dict):
        return False
    if data.get('kind') == 'screening_job':
        return True
    return 'status' in data and 'start_time' in data


def validate_setup_config(data: Any, path: str) -> Tuple[bool, str]:
    basename = os.path.basename(path).lower()
    if basename in KNOWN_WRONG_FILES:
        return False, KNOWN_WRONG_FILES[basename]

    if not isinstance(data, dict):
        return False, GENERIC_SETUP_FAILURE

    if data.get('config_kind') == CONFIG_KIND:
        return True, ''

    if data.get('config_version') is not None:
        has_paths = bool(data.get('user_video_path') or data.get('video_dir'))
        has_ids = bool(data.get('project_dir') or data.get('analysis_id'))
        if has_paths and has_ids:
            return True, ''

    if _looks_like_metadata(data):
        return False, KNOWN_WRONG_FILES['metadata.json']

    if _looks_like_group_rois(data):
        return False, KNOWN_WRONG_FILES['group_rois.json']

    if isinstance(data, dict) and 'group_names' in data and 'seed_counts' in data:
        if 'config_version' not in data and 'config_kind' not in data:
            return False, KNOWN_WRONG_FILES['group_info.json']

    return False, GENERIC_SETUP_FAILURE


def build_interface_config(analysis_tab) -> Dict[str, Any]:
    """Serialize Analysis tab state. Never includes group_rois."""
    groups = []
    for entry in analysis_tab.group_entries:
        seed_count = entry.get_seed_count()
        groups.append({
            'name': entry.name_edit.text().strip(),
            'seed_count': seed_count if seed_count is not None else '',
        })

    payload = {
        'config_kind': CONFIG_KIND,
        'config_version': CONFIG_VERSION,
        'user_video_path': analysis_tab.video_path_edit.text(),
        'project_dir': analysis_tab.proj_dir_edit.text(),
        'analysis_id': analysis_tab.identifier_edit.text().strip(),
        'PlateCondition': analysis_tab.plateConditionName.text(),
        'ExtraVariable': analysis_tab.extraField.text(),
        'timeStep': analysis_tab.time_delta_edit.text(),
        'add_time': analysis_tab.add_time_edit.text(),
        'germination_time_cut': analysis_tab.germination_time_edit.text(),
        'videoHasQR': analysis_tab.qr_checkbox.isChecked(),
        'knownDistance': analysis_tab.known_dist_edit.text(),
        'pixelDistance': analysis_tab.pixel_dist_edit.text(),
        'germination_analysis': analysis_tab.germination_checkbox.isChecked(),
        'plant_growth_analysis': analysis_tab.plant_growth_checkbox.isChecked(),
        'show_tracking': analysis_tab.show_tracking_checkbox.isChecked(),
        'germination_each_video': analysis_tab.store_each_video_checkbox.isChecked(),
        'do_fpca': analysis_tab.fpca_checkbox.isChecked(),
        'fpca_components': analysis_tab.fpca_components_edit.text(),
        'normalize_fpca': analysis_tab.fpca_normalize_checkbox.isChecked(),
        'groups': groups,
    }
    if hasattr(analysis_tab, 'current_rpi_cam'):
        rpi, cam = analysis_tab.current_rpi_cam()
        payload['rpi'] = rpi
        payload['cam'] = cam
    if hasattr(analysis_tab, 'averagePerPlantStats'):
        payload['averagePerPlantStats'] = analysis_tab.averagePerPlantStats.isChecked()
        payload['measureRelativeToInitial'] = analysis_tab.measureRelativeToInitial.isChecked()
        payload['everyXhourField'] = analysis_tab.everyXhourField.text()
        payload['everyXhourFieldFourier'] = analysis_tab.everyXhourFieldFourier.text()
        payload['everyXhourFieldAngles'] = analysis_tab.everyXhourFieldAngles.text()
        payload['doFourier'] = analysis_tab.doFourier.isChecked()
        payload['genotypeAxisLabel'] = analysis_tab.genotypeAxisLabelField.text()
        payload['plateConditionAxisLabel'] = analysis_tab.plateConditionAxisLabelField.text()
        payload['extraVariableLabel'] = analysis_tab.extraVariableLabelField.text()
        payload['timeSyncMode'] = getattr(analysis_tab, 'timeSyncMode', 'clock')
        payload['timeDurationHours'] = getattr(analysis_tab, 'timeDurationHours', None)
        payload['reportFolderName'] = getattr(analysis_tab, 'reportFolderName', 'Report')
        payload['figureClockTicks'] = getattr(analysis_tab, 'figureClockTicks', ['00:00'])
        payload['showFigureClockTicks'] = getattr(analysis_tab, 'showFigureClockTicks', True)
        payload['timeGroups'] = getattr(analysis_tab, 'timeGroups', [])
        payload['timePeriodSources'] = getattr(analysis_tab, 'timePeriodSources', [])
        for name in (
            'statsByGenotype', 'statsGenotypeByPlate', 'statsGenotypeByExtra',
            'statsByPlateCondition', 'statsByExtraVariable',
            'statsPlateWithinGenotype', 'statsExtraWithinGenotype',
            'measureHypocotyl', 'measureMainRoot', 'measureTotalRoot',
            'measureArea', 'measureDenseRoot', 'measureGermination',
        ):
            widget = getattr(analysis_tab, name, None)
            if widget is not None:
                payload[name] = widget.isChecked()
        if getattr(analysis_tab, 'advanced_group', None) is not None:
            payload['advancedComparisonModes'] = analysis_tab.advanced_group.isChecked()
    return payload


def _rebuild_groups(analysis_tab, groups: List[Dict[str, Any]]) -> None:
    scroll = getattr(analysis_tab, 'group_scroll_widget', None)
    if scroll is not None:
        scroll.setUpdatesEnabled(False)

    try:
        while analysis_tab.group_entries:
            entry = analysis_tab.group_entries.pop()
            analysis_tab.group_layout.removeWidget(entry)
            entry.hide()
            entry.setParent(None)
            entry.deleteLater()

        if not groups:
            groups = [{'name': '', 'seed_count': ''}]

        for group in groups:
            analysis_tab._append_group_entry()
            entry = analysis_tab.group_entries[-1]
            entry.name_edit.setText(str(group.get('name', '')))
            seed_count = group.get('seed_count', '')
            if seed_count not in (None, ''):
                entry.seed_count_edit.setText(str(seed_count))
    finally:
        if scroll is not None:
            scroll.setUpdatesEnabled(True)


def _set_checkbox_from_keys(widget, data: Dict[str, Any], keys: tuple) -> None:
    for key in keys:
        if key in data:
            widget.setChecked(_as_bool(data[key]))
            return


def apply_interface_config(analysis_tab, data: Dict[str, Any]) -> None:
    """Apply saved session or process config fields to the Analysis tab (ignores group_rois)."""
    analysis_tab._loading_config = True
    signal_widgets = (
        analysis_tab.qr_checkbox,
        analysis_tab.plant_growth_checkbox,
    )
    for widget in signal_widgets:
        widget.blockSignals(True)

    try:
        video_path = data.get('user_video_path') or data.get('video_dir', '')
        if video_path is not None:
            analysis_tab.video_path_edit.setText(str(video_path))

        project_dir = data.get('project_dir', '')
        if project_dir is not None:
            analysis_tab.proj_dir_edit.setText(str(project_dir))

        analysis_id = data.get('analysis_id', data.get('identifier', ''))
        if analysis_id is not None:
            analysis_tab.identifier_edit.setText(str(analysis_id))

        if hasattr(analysis_tab, 'plateConditionName'):
            plate = data.get('PlateCondition', data.get('plateConditionName', ''))
            extra = data.get('ExtraVariable', data.get('extraField', ''))
            analysis_tab.plateConditionName.setText(str(plate or ''))
            analysis_tab.extraField.setText(str(extra or ''))

        for widget, keys, default in (
            (analysis_tab.time_delta_edit, ('timeStep', 'time_delta'), ''),
            (analysis_tab.add_time_edit, ('add_time',), ''),
            (analysis_tab.known_dist_edit, ('knownDistance', 'known_distance'), ''),
            (analysis_tab.pixel_dist_edit, ('pixelDistance', 'pixel_distance'), ''),
            (analysis_tab.fpca_components_edit, ('fpca_components',), '2'),
        ):
            value = None
            for key in keys:
                if key in data and data[key] is not None:
                    value = data[key]
                    break
            if value is not None:
                widget.setText(str(value))
            elif keys[0] not in data and default:
                widget.setText(str(default))

        if 'germination_time_cut' in data or 'germination_time' in data:
            germ_time = data.get('germination_time_cut', data.get('germination_time', ''))
            if germ_time is not None:
                analysis_tab.germination_time_edit.setText(str(germ_time))

        _set_checkbox_from_keys(analysis_tab.qr_checkbox, data, ('videoHasQR', 'has_qr'))
        _set_checkbox_from_keys(analysis_tab.germination_checkbox, data, ('germination_analysis', 'do_germination'))
        _set_checkbox_from_keys(analysis_tab.plant_growth_checkbox, data, ('plant_growth_analysis', 'do_plant_growth'))
        _set_checkbox_from_keys(analysis_tab.show_tracking_checkbox, data, ('show_tracking',))
        _set_checkbox_from_keys(
            analysis_tab.store_each_video_checkbox,
            data,
            ('germination_each_video', 'germination-each-video'),
        )
        _set_checkbox_from_keys(analysis_tab.fpca_checkbox, data, ('do_fpca',))
        _set_checkbox_from_keys(analysis_tab.fpca_normalize_checkbox, data, ('normalize_fpca',))

        if hasattr(analysis_tab, 'averagePerPlantStats'):
            _set_checkbox_from_keys(analysis_tab.averagePerPlantStats, data, ('averagePerPlantStats',))
            if getattr(analysis_tab, 'measureRelativeToInitial', None) is not None:
                _set_checkbox_from_keys(
                    analysis_tab.measureRelativeToInitial, data, ('measureRelativeToInitial',),
                )
            _set_checkbox_from_keys(analysis_tab.doFourier, data, ('doFourier',))
            for widget, key in (
                (analysis_tab.everyXhourField, 'everyXhourField'),
                (analysis_tab.everyXhourFieldFourier, 'everyXhourFieldFourier'),
                (analysis_tab.everyXhourFieldAngles, 'everyXhourFieldAngles'),
                (analysis_tab.genotypeAxisLabelField, 'genotypeAxisLabel'),
                (analysis_tab.plateConditionAxisLabelField, 'plateConditionAxisLabel'),
                (analysis_tab.extraVariableLabelField, 'extraVariableLabel'),
            ):
                if key in data and data[key] is not None:
                    widget.setText(str(data[key]))
            for name in (
                'statsByGenotype', 'statsGenotypeByPlate', 'statsGenotypeByExtra',
                'statsByPlateCondition', 'statsByExtraVariable',
                'statsPlateWithinGenotype', 'statsExtraWithinGenotype',
                'measureHypocotyl', 'measureMainRoot', 'measureTotalRoot',
                'measureArea', 'measureDenseRoot', 'measureGermination',
            ):
                widget = getattr(analysis_tab, name, None)
                if widget is not None:
                    _set_checkbox_from_keys(widget, data, (name,))
            if getattr(analysis_tab, 'advanced_group', None) is not None:
                _set_checkbox_from_keys(
                    analysis_tab.advanced_group, data, ('advancedComparisonModes',),
                )
            if 'timeSyncMode' in data:
                analysis_tab.timeSyncMode = data.get('timeSyncMode') or 'clock'
            if 'timeDurationHours' in data:
                analysis_tab.timeDurationHours = data.get('timeDurationHours')
            if 'reportFolderName' in data:
                analysis_tab.reportFolderName = data.get('reportFolderName') or 'Report'
            if 'figureClockTicks' in data:
                analysis_tab.figureClockTicks = data.get('figureClockTicks') or ['00:00']
            if 'showFigureClockTicks' in data:
                analysis_tab.showFigureClockTicks = bool(data.get('showFigureClockTicks'))
            else:
                mode = getattr(analysis_tab, 'timeSyncMode', 'clock')
                analysis_tab.showFigureClockTicks = mode != 'anchor'
            if 'timeGroups' in data:
                analysis_tab.timeGroups = data.get('timeGroups') or []
            if 'timePeriodSources' in data:
                analysis_tab.timePeriodSources = data.get('timePeriodSources') or []
            else:
                analysis_tab.timePeriodSources = []

        groups = data.get('groups')
        if groups is None and 'group_names' in data:
            seed_counts = data.get('seed_counts', [])
            groups = []
            for i, name in enumerate(data['group_names']):
                count = seed_counts[i] if i < len(seed_counts) else ''
                if count == 0:
                    count = ''
                groups.append({'name': name, 'seed_count': count})
        if groups is not None:
            _rebuild_groups(analysis_tab, groups)

        analysis_tab.toggle_calibration_mode()
        analysis_tab.toggle_plant_growth_options()
        analysis_tab.on_project_dir_changed()
        if hasattr(analysis_tab, '_remember_auto_identifier'):
            analysis_tab._remember_auto_identifier()
    finally:
        for widget in signal_widgets:
            widget.blockSignals(False)
        analysis_tab._loading_config = False


def save_interface_config(analysis_tab, path: str) -> bool:
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(build_interface_config(analysis_tab), f, indent=4)
        return True
    except OSError:
        return False


def load_interface_config(analysis_tab, path: str) -> Tuple[bool, str]:
    if not os.path.exists(path):
        return False, f'File not found:\n{path}'

    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        return False, f'Could not read the file (invalid JSON):\n{e}'

    ok, error = validate_setup_config(data, path)
    if not ok:
        return False, error

    try:
        apply_interface_config(analysis_tab, data)
    except (TypeError, KeyError, AttributeError) as e:
        return False, f'The file looks like a setup file but could not be applied:\n{e}'

    return True, ''
