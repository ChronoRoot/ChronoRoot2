"""Run disk and compute work off the Qt GUI thread.

The window stays responsive because the worker thread does the slow call while
a nested event loop keeps painting. A loading dialog appears only when the
call is still running after a short delay.
"""

import sys

from PyQt5 import QtCore, QtWidgets


_SHOW_AFTER_MS = 300
_kept_alive = []


def _retire_thread(thread, worker):
    """Stop a worker thread and delete it only after it has finished.

    Destroying a QThread while its run() is still on the stack aborts the
    process from C++. Python must keep the wrapper alive until then, because
    deleteLater does not.
    """
    if thread is None:
        return
    _kept_alive.append(thread)
    if worker is not None:
        _kept_alive.append(worker)
    thread.quit()
    thread.wait()

    def release():
        if thread.isRunning():
            thread.quit()
            thread.wait()
        if thread.isRunning():
            QtCore.QTimer.singleShot(50, release)
            return
        if worker is not None:
            worker.deleteLater()
        thread.deleteLater()

        def forget():
            for obj in (worker, thread):
                if obj is not None and obj in _kept_alive:
                    _kept_alive.remove(obj)
        QtCore.QTimer.singleShot(0, forget)

    if thread.isRunning():
        thread.finished.connect(release)
    else:
        QtCore.QTimer.singleShot(0, release)


class _Job(QtCore.QObject):
    done = QtCore.pyqtSignal()

    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.result = None
        self.error = None

    @QtCore.pyqtSlot()
    def run(self):
        try:
            self.result = self.fn()
        except Exception:
            self.error = sys.exc_info()
        # Back to the GUI thread before the signal, so the caller can delete
        # this object after the nested event loop returns.
        app = QtCore.QCoreApplication.instance()
        if app is not None:
            self.moveToThread(app.thread())
        self.done.emit()


def run_off_main_thread(parent, message, fn, then=None):
    """Run fn() on a QThread and return its result on the GUI thread.

    message is the loading dialog title. None runs quietly: no dialog and the
    window stays enabled. then(result), when given, runs on the GUI thread
    before the dialog closes, so a long table fill can keep the dialog up.

    Exceptions raised by fn or by then are raised again here.
    """
    depth = getattr(run_off_main_thread, "depth", 0) + 1
    run_off_main_thread.depth = depth
    disabled_here = False
    dialog = None
    thread = None
    worker = None
    try:
        if message and parent is not None and depth == 1:
            parent.setEnabled(False)
            disabled_here = True

        worker = _Job(fn)
        thread = QtCore.QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        loop = QtCore.QEventLoop()
        state = {"done": False}

        def finish():
            if state["done"]:
                return
            if worker.error is None and then is not None:
                try:
                    then(worker.result)
                except Exception:
                    worker.error = sys.exc_info()
            state["done"] = True
            loop.quit()

        worker.done.connect(finish)

        if message:
            dialog = QtWidgets.QDialog(None)
            dialog.setWindowTitle(message)
            dialog.setWindowModality(QtCore.Qt.ApplicationModal)
            dialog.setWindowFlags(
                QtCore.Qt.Dialog
                | QtCore.Qt.WindowTitleHint
                | QtCore.Qt.CustomizeWindowHint
            )
            dialog.setMinimumWidth(360)
            layout = QtWidgets.QVBoxLayout(dialog)
            label = QtWidgets.QLabel(message)
            label.setWordWrap(True)
            layout.addWidget(label)
            bar = QtWidgets.QProgressBar()
            bar.setRange(0, 0)
            bar.setTextVisible(False)
            layout.addWidget(bar)

            timer = QtCore.QTimer(dialog)
            timer.setSingleShot(True)

            def show_dialog():
                if state["done"]:
                    return
                dialog.adjustSize()
                if parent is not None and parent.isVisible():
                    center = parent.frameGeometry().center()
                    dialog.move(
                        center.x() - dialog.width() // 2,
                        center.y() - dialog.height() // 2,
                    )
                dialog.show()
                dialog.raise_()
                dialog.activateWindow()

            timer.timeout.connect(show_dialog)
            timer.start(_SHOW_AFTER_MS)

        thread.start()
        loop.exec_()
        error = worker.error
        result = worker.result
    finally:
        run_off_main_thread.depth = depth - 1
        if dialog is not None:
            dialog.close()
            dialog.deleteLater()
        _retire_thread(thread, worker)
        if disabled_here and parent is not None:
            parent.setEnabled(True)
            parent.activateWindow()

    if error is not None:
        _exc_type, exc, tb = error
        raise exc.with_traceback(tb)
    return result
