import sys
from PyQt5.QtWidgets import QApplication
from src.annotation_app.media_editor import MediaEditor


def main():
    app = QApplication(sys.argv)
    window = MediaEditor()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
