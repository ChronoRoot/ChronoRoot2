"""Parse ChronoRoot robot video paths into rpi / camera identifiers."""

import os
import re


def parse_robot_video_path(video_dir):
    """Return (rpi, cam) from a robot folder path, or (None, None).

    rpi is the digits after 'rpi' in any path segment.
    cam is the last folder, with a leading 'cam_' stripped.
    """
    if not video_dir:
        return None, None
    parts = [p for p in os.path.normpath(str(video_dir)).split(os.sep) if p]
    if not parts:
        return None, None

    last = parts[-1]
    cam = last[4:] if last.lower().startswith('cam_') else last

    rpi = None
    for part in parts:
        match = re.search(r'rpi(\d+)', part, re.IGNORECASE)
        if match:
            rpi = match.group(1)

    if not cam:
        cam = None
    return rpi, cam


def rpi_cam_from_identifier(analysis_id):
    """Parse identifier 'rpi3_cam_0' into ('3', '0')."""
    if not analysis_id:
        return None, None
    match = re.match(r'^rpi(\d+)_cam_(.+)$', str(analysis_id), re.IGNORECASE)
    if match:
        return match.group(1), match.group(2)
    return None, None


def identifier_suffix(extra):
    """Filesystem-safe extra token for analysis identifiers."""
    text = re.sub(r'[^A-Za-z0-9_]+', '_', str(extra or '').strip()).strip('_')
    return text


def identifier_from_rpi_cam(rpi, cam, extra=None):
    base = f'rpi{rpi}_cam_{cam}'
    suffix = identifier_suffix(extra)
    if suffix:
        return f'{base}_{suffix}'
    return base


def _clean_id(value):
    if value is None:
        return ''
    text = str(value).strip()
    if text.lower() in ('', 'nan', 'none'):
        return ''
    return text


def resolve_rpi_cam(rpi=None, cam=None, video_dir=None, analysis_id=None):
    """Fill rpi/cam from explicit values, video path, then analysis identifier."""
    rpi = _clean_id(rpi)
    cam = _clean_id(cam)
    if cam.lower().startswith('cam_'):
        cam = cam[4:]
    if rpi and cam:
        return rpi, cam

    parsed_rpi, parsed_cam = parse_robot_video_path(video_dir)
    rpi = rpi or (parsed_rpi or '')
    cam = cam or (parsed_cam or '')
    if rpi and cam:
        return rpi, cam

    parsed_rpi, parsed_cam = rpi_cam_from_identifier(analysis_id)
    rpi = rpi or (parsed_rpi or '')
    cam = cam or (parsed_cam or '')
    if rpi and cam:
        return rpi, cam

    fallback_rpi = rpi or (str(analysis_id).strip() if analysis_id else 'unspecified')
    fallback_cam = cam or '0'
    return fallback_rpi, fallback_cam


def cam_folder_name(cam):
    text = str(cam).strip()
    if not text:
        text = '0'
    if text.lower().startswith('cam_'):
        return text
    return f'cam_{text}'
