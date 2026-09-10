"""Launch ChronoRoot analysis pipeline scripts as subprocesses."""

import os
import subprocess

from analysis.utils.metadata_schema import pending_analysis_path

PROJECT_CONFIG_NAME = "project_config.json"


def project_config_path(project_dir):
  return os.path.join(project_dir, PROJECT_CONFIG_NAME)


def run_analysis(project_dir):
  subprocess.Popen([
      "python", "1_analysis.py",
      "--config", project_config_path(project_dir),
  ])


def run_analysis_config(config_path):
  subprocess.Popen([
      "python", "1_analysis.py",
      "--config", config_path,
  ])


def run_pending_analysis(project_dir):
  subprocess.Popen([
      "python", "1_analysis.py",
      "--config", pending_analysis_path(project_dir),
  ])


def run_analysis_rerun(metadata_path):
  subprocess.Popen([
      "python", "1_analysis.py",
      "--config", metadata_path,
      "--rerun",
  ])


def run_postprocess(project_dir):
  subprocess.Popen([
      "python", "2_postprocess.py",
      "--config", project_config_path(project_dir),
  ])


def run_report(project_dir):
  subprocess.Popen([
      "python", "3_generateReport.py",
      "--config", project_config_path(project_dir),
  ])
