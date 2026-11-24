"""Keep a Qt window open when a slot raises, and show the error.

PyQt treats an exception that leaves a button, timer, or signal handler as
fatal and aborts the process. Installing this hook makes PyQt report the
exception here instead, whichever way the window was started.
"""

import sys
import threading
import traceback

_installed = False
_showing = False


def install_gui_exception_hook():
    """Replace the process exception hooks. Safe to call more than once."""
    global _installed
    if _installed:
        return
    _installed = True
    sys.excepthook = _report
    if hasattr(threading, "excepthook"):
        def _thread_report(args):
            _report(args.exc_type, args.exc_value, args.exc_traceback)
        threading.excepthook = _thread_report


def _report(exc_type, exc_value, exc_tb):
    if exc_type is not None and issubclass(exc_type, (KeyboardInterrupt, SystemExit)):
        sys.__excepthook__(exc_type, exc_value, exc_tb)
        return

    detail = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    sys.__excepthook__(exc_type, exc_value, exc_tb)
    _show_dialog(exc_value, detail)


def _show_dialog(exc_value, detail):
    global _showing
    if _showing:
        return

    from PyQt5 import QtCore, QtWidgets

    app = QtWidgets.QApplication.instance()
    if app is None:
        return
    if QtCore.QThread.currentThread() is not app.thread():
        QtCore.QTimer.singleShot(0, app, lambda: _show_dialog(exc_value, detail))
        return

    summary = str(exc_value).strip() or exc_value.__class__.__name__
    _showing = True
    try:
        box = QtWidgets.QMessageBox()
        box.setIcon(QtWidgets.QMessageBox.Critical)
        box.setWindowTitle("ChronoRoot")
        box.setText("Something went wrong. The window stayed open.")
        box.setInformativeText(summary)
        box.setDetailedText(detail)
        box.exec_()
    except Exception:
        print(detail, file=sys.stderr)
    finally:
        _showing = False
