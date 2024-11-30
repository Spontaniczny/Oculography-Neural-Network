import sys
from PyQt5.QtWidgets import QApplication
from video_editor import VideoEditor

def main():
    app = QApplication(sys.argv)
    window = VideoEditor()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
