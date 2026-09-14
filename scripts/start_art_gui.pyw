"""Double-click on Windows with the project's Python dependencies installed."""
import sys
import traceback

from PySide6.QtWidgets import QApplication, QMessageBox

from art_gui import main


def show_unexpected(kind, error, trace):
    # pythonw has no stderr, so an exception raised in a Qt slot would otherwise vanish.
    if QApplication.instance():
        QMessageBox.critical(None, '予期しないエラー', ''.join(traceback.format_exception(kind, error, trace))[-3000:])


sys.excepthook = show_unexpected
main()
