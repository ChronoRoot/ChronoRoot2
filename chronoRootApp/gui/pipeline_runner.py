"""Launch ChronoRoot analysis pipeline scripts as subprocesses."""

import os
import subprocess

PROJECT_CONFIG_NAME = "project_config.json"


def project_config_path(project_dir):
  return os.path.join(project_dir, PROJECT_CONFIG_NAME)


def run_analysis(project_dir):
  return subprocess.Popen([
      "python", "1_analysis.py",
      "--config", project_config_path(project_dir),
  ])


def run_analysis_config(config_path):
  return subprocess.Popen([
      "python", "1_analysis.py",
      "--config", config_path,
  ])


def run_pending_analysis(config_path):
  """Launch analysis for a pending config path written by the GUI."""
  return run_analysis_config(config_path)


def run_analysis_rerun(metadata_path):
  return subprocess.Popen([
      "python", "1_analysis.py",
      "--config", metadata_path,
      "--rerun",
  ])


def run_postprocess(project_dir):
  return subprocess.Popen([
      "python", "2_postprocess.py",
      "--config", project_config_path(project_dir),
  ])


def run_report(project_dir, subset_name=""):
  command = [
      "python", "3_generateReport.py",
      "--config", project_config_path(project_dir),
  ]
  if subset_name:
    command.extend(["--subset", subset_name])
  return subprocess.Popen(command)
