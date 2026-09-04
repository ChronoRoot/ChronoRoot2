"""Shared UI error handling and subprocess launching for the screening app."""

import os
import sys
from typing import Callable, List, Optional

from PyQt5.QtCore import QObject, QProcess, QProcessEnvironment
from PyQt5.QtGui import QTextCursor
from PyQt5.QtWidgets import (
    QDialog, QDialogButtonBox, QMessageBox, QTextEdit, QVBoxLayout, QWidget,
)


def show_warning(parent: Optional[QWidget], title: str, message: str) -> None:
    QMessageBox.warning(parent, title, message)


def show_critical(parent: Optional[QWidget], title: str, message: str) -> None:
    QMessageBox.critical(parent, title, message)


def show_information(parent: Optional[QWidget], title: str, message: str) -> None:
    QMessageBox.information(parent, title, message)


class WorkerLogDialog(QDialog):
    """Non-modal console that streams a worker's stdout/stderr."""

    def __init__(self, title: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._base_title = title
        self.setWindowTitle(title)
        self.setModal(False)
        self.resize(780, 460)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setLineWrapMode(QTextEdit.NoWrap)

        self._finished = False
        self.close_btn = QDialogButtonBox(QDialogButtonBox.Close)
        self.close_btn.button(QDialogButtonBox.Close).setEnabled(False)
        self.close_btn.rejected.connect(self.close)

        layout = QVBoxLayout(self)
        layout.addWidget(self.log)
        layout.addWidget(self.close_btn)

    def closeEvent(self, event):
        if not self._finished:
            event.ignore()
            return
        super().closeEvent(event)

    def append_text(self, text: str) -> None:
        self.log.moveCursor(QTextCursor.End)
        self.log.insertPlainText(text)
        self.log.moveCursor(QTextCursor.End)

    def mark_finished(self, success: bool) -> None:
        self._finished = True
        suffix = 'finished' if success else 'failed'
        self.setWindowTitle(f'{self._base_title} — {suffix}')
        self.close_btn.button(QDialogButtonBox.Close).setEnabled(True)


class WorkerLauncher(QObject):
    """Launch a worker process without shell=True and report failures via popups."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        working_directory: Optional[str] = None,
    ):
        super().__init__(parent)
        self._dialog_parent = parent
        self.working_directory = working_directory or os.path.dirname(os.path.abspath(__file__))
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.setWorkingDirectory(self.working_directory)
        env = QProcessEnvironment.systemEnvironment()
        env.insert('PYTHONUNBUFFERED', '1')
        self.process.setProcessEnvironment(env)
        self._output_chunks: List[bytes] = []
        self._on_success: Optional[Callable[[], None]] = None
        self._success_title: Optional[str] = None
        self._success_message: Optional[str] = None
        self._error_title = "Processing Error"
        self._log_dialog: Optional[WorkerLogDialog] = None
        self.process.readyReadStandardOutput.connect(self._forward_output)
        self.process.finished.connect(self._on_finished)

    def _decode_bytes(self, data) -> str:
        return bytes(data).decode('utf-8', errors='replace')

    def _forward_output(self) -> None:
        data = self.process.readAllStandardOutput()
        if not data:
            return
        raw = bytes(data)
        self._output_chunks.append(raw)
        text = self._decode_bytes(raw)
        sys.stdout.write(text)
        sys.stdout.flush()
        if self._log_dialog is not None:
            self._log_dialog.append_text(text)

    def _on_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        leftover = self.process.readAllStandardOutput()
        if leftover:
            raw = bytes(leftover)
            self._output_chunks.append(raw)
            text = self._decode_bytes(raw)
            sys.stdout.write(text)
            sys.stdout.flush()
            if self._log_dialog is not None:
                self._log_dialog.append_text(text)

        success = exit_code == 0 and exit_status == QProcess.NormalExit
        if self._log_dialog is not None:
            self._log_dialog.mark_finished(success)

        if success:
            if self._on_success:
                self._on_success()
            elif self._success_title and self._success_message:
                show_information(self._dialog_parent, self._success_title, self._success_message)
            return

        details = b"".join(self._output_chunks).decode("utf-8", errors="replace").strip()
        details = details or f"Process exited with code {exit_code}."
        if len(details) > 2000:
            details = "...\n" + details[-2000:]
        show_critical(self._dialog_parent, self._error_title, details)

    def start(
        self,
        args: List[str],
        *,
        started_title: Optional[str] = None,
        started_message: Optional[str] = None,
        success_title: Optional[str] = None,
        success_message: Optional[str] = None,
        error_title: str = "Processing Error",
        on_success: Optional[Callable[[], None]] = None,
        log_dialog: Optional[WorkerLogDialog] = None,
    ) -> bool:
        if self.process.state() != QProcess.NotRunning:
            show_warning(self._dialog_parent, "Busy", "A background task is already running.")
            return False

        self._output_chunks = []
        self._on_success = on_success
        self._success_title = success_title
        self._success_message = success_message
        self._error_title = error_title
        self._log_dialog = log_dialog

        program = args[0]
        program_args = args[1:]
        if program == "python":
            program = sys.executable
            program_args = args[1:]

        self.process.start(program, program_args)
        if not self.process.waitForStarted(5000):
            show_critical(
                self._dialog_parent,
                error_title,
                f"Failed to start process:\n{self.process.errorString()}",
            )
            return False

        if log_dialog is not None:
            log_dialog.show()
            log_dialog.raise_()
        if started_title and started_message:
            show_information(self._dialog_parent, started_title, started_message)
        return True

    def is_running(self) -> bool:
        return self.process.state() != QProcess.NotRunning


def launch_worker(
    args: List[str],
    parent: Optional[QWidget] = None,
    *,
    started_title: Optional[str] = None,
    started_message: Optional[str] = None,
    success_title: Optional[str] = None,
    success_message: Optional[str] = None,
    error_title: str = "Processing Error",
    working_directory: Optional[str] = None,
    on_success: Optional[Callable[[], None]] = None,
    log_dialog: Optional[WorkerLogDialog] = None,
) -> WorkerLauncher:
    launcher = WorkerLauncher(parent=parent, working_directory=working_directory)
    launcher.start(
        args,
        started_title=started_title,
        started_message=started_message,
        success_title=success_title,
        success_message=success_message,
        error_title=error_title,
        on_success=on_success,
        log_dialog=log_dialog,
    )
    return launcher
