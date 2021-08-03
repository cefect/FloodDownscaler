'''
Created on Jul. 20, 2021

@author: cefect

PyQt helpers
'''
from PyQt5 import QtCore, QtGui

from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QLineEdit,\
     QMessageBox, QDesktopWidget, QLabel

from PyQt5.QtCore import pyqtSlot
 
import sys, time

 
        
        
class Window(QMainWindow):

    def __init__(self, title='title'):
        super().__init__()
        self.title = title
        self.left = 10
        self.top = 10
        self.width = 400
        self.height = 140
        

        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
    
        # Create textbox
        self.t = QLabel(self)
        
    def center_window(self):
        
        #=======================================================================
        # build a proxy recvtangle
        #=======================================================================
        qtRectangle = self.frameGeometry()
        centerPoint = QDesktopWidget().availableGeometry().center()
        qtRectangle.moveCenter(centerPoint)
        self.move(qtRectangle.topLeft())
        
        #=======================================================================
        # move the rectangle
        #=======================================================================
        centerPoint = QDesktopWidget().availableGeometry().center()
        qtRectangle.moveCenter(centerPoint)
        
        #=======================================================================
        # match the window
        #=======================================================================
        self.move(qtRectangle.topLeft())
        
    def log(self, text):
        
        old_text = self.text()
        
        self.setText(old_text+'/n' + text)
        
        
        
    #===========================================================================
    #     # Create a button in the window
    #     self.button = QPushButton('Show text', self)
    #     self.button.move(20,80)
    #     
    #     # connect button to function on_click
    #     self.button.clicked.connect(self.on_click)
    #     self.show()
    # 
    # @pyqtSlot()
    # def on_click(self):
    #     textboxValue = self.textbox.text()
    #     QMessageBox.question(self, 
    #                          'Message - pythonspot.com',
    #                           "You typed: " + textboxValue, 
    #                           QMessageBox.Ok,
    #                            QMessageBox.Ok)
    #     
    #     self.textbox.setText("")
    #===========================================================================
 
        
        
def openWindow():
    QApp = QApplication(sys.argv) #initlize a QT appliaction (inplace of Qgis)
    
    
    app = Window()
    app.center_window()
    
    
    app.t.setText('test')
    app.show()
    
 
    time.sleep(1)
    
    app.t.setText('test2')
    app.show()
    
    time.sleep(1)
 
    sys.exit(QApp.exec_()) #wrap
 
        
    
    
if __name__ == '__main__':

    openWindow()
    #window2()
    

    
    print('finished')
    