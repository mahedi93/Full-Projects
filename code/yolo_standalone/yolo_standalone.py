
import os
#import sys
import math
import torch
from ultralytics import YOLO
#from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import cv2

#Library for scenarios
#from __future__ import print_function
#from builtins import input
import cv2 as cv
import numpy as np
#import argparse


# Data default dir
DIR_DATA = '..\..\data'

class SignParam:

  # Constructor
  def __init__(self):
    """

    F: File
    O: Object
    C: class
    """
    self._UUID = ''
    self._sIdx = ''
    self._sLen = ''
    self._time = ''
    self._fImg = ''
    self._fIdx = ''
    self._cObj = ''
    self._cNam = ''
    self._cCon = ''
    self._tcBr = '' # Test case Brightness
    self._tcCo = '' # Test case Contrast
    self._tcUs = '' # Test case Up Size
    self._tcDs = '' # Test case Down Size
    self._tcBa = '' # Test case camBarrel
    self._tcPc = '' # Test case camPinCushion
  @property
  def Time(self):
      """ init time property."""
      return self._time

  @Time.setter
  def Time(self, value):
      """ set time property."""
      self._time = value

  @Time.deleter
  def Time(self):
      """ delete time property."""
      del self._time

  def dataHeadCSVShort(self):
    """
    """
    data  = 'Time' + ',';
    data += 'fIdx' + ',';
    data += 'sIdx' + ',';
    data += 'sLen' + ',';
    data += 'fImg' + ',';
    data += 'cObj' + ',';
    data += 'cNam' + ',';
    data += 'cCon' + ',';
    data += 'tcBr' + ',';
    data += 'tcCo' + ',';
    data += 'tcUs' + ',';
    data += 'tcDs' ;
    
    return data;

  def dataInCSV(self):

    """
    """ 
    data =  str(self._time) + ',';
    data += str(self._fIdx) + ',';
    data += str(self._sIdx) + ',';
    data += str(self._sLen) + ',';
    data += str(self._fImg) + ',';
    data += str(self._cObj) + ',';
    data += str(self._cNam) + ',';
    data += str(self._cCon) + ',';
    data += str(self._tcBr) + ',';
    data += str(self._tcCo) + ',';
    data += str(self._tcUs) + ',';
    data += str(self._tcDs) ;
    
    return data;


class SignDetect:

  ## list of images to process
  #lImgList = []

  ## yolo model name (*.pt)
  #sPtName = ''

  ## yolo model name (*.pt)
  #ptModel = ''

  ## output file name
  #sReportFile = 'out-report.txt'

  ## handler to report file
  #hReport = ''

  ## Report file array
  #lReportData = []

  def __init__(self):
    self.lImgList = []
    self.setReportName = 'out_report_resizeddown.csv'
    self.sPtName = "yolov8m.pt" 
    self.ptModel = ''
    self.hReport = ''
    self.lReportData = []

  # handler to Yolo model
  #hModel = ;
  #def __init__(self):

  def addImgList(self, imgDir):
    """
    """
    self.imgDir = imgDir
    self.lImgList = os.listdir(imgDir)

  def yoloModel(self, ptFile):
    """
    """
   
    self.ptModel = YOLO(ptFile)
    

  def extractParams(self, lPredicted, fIdx):
    """
    """
    # change final attribute to desired box format
    # imgsCount = len(lPredicted)
    # signsCount = len(lPredicted[0].boxes)

    # Loop through the predictions
    # since we are providing one image at a time, it will loop once
    # i.e. lPredicted array length is 1
    for iImgPredict, hImgPredict in enumerate(lPredicted):
      # Loop through the boxes, i.e. detected objects/signs
      # iBox: Index of the box object
      # hBox: Handler to the box object
      for iBox, hBox in enumerate(hImgPredict.boxes):

        # get new signParam object
        param = SignParam()

        # frame index
        param._fIdx = str(fIdx)

        # path of image file, store base filename only
        param._fImg = os.path.basename(hImgPredict.path)

        # Number of sign object predicted
        param._sLen = str(len(hImgPredict.boxes))

        # index within the prediction
        param._sIdx = str(iBox)

        # class index
        iClsIdx = int(hBox.cls[0].item())

        # store class index
        param._cObj = str(iClsIdx)
        
        # get respective class name from its parent names list 
        param._cNam = hImgPredict.names[iClsIdx]

        # value obtained: 0.4668, multiple by 100 -> 46.68, 
        # ceil to 47 and 
        # normalize by / 100 -> 0.46
        param._cCon = math.ceil(hBox.conf[0] * 100) / 100

      

        # store the values
        self.lReportData.append(param)

  def execDetectByList(self):
    """
    """
    fIdx = 0;
    for img in self.lImgList:
      fIdx += 1
      imgPath = self.imgDir + '/' + img;
      predictOut = self.ptModel(imgPath)
      self.extractParams(predictOut, str(fIdx))


  
    

  def setReportName(self, reportName):
    """
    """
    sReportFilename = reportName;

  def writeReportHead(self):
    """
    """
    print("TODO Write report file")

  def writeReportBody(self):
    """
    """
    print("TODO Write report file and close")
    if self.hReport == '':
      self.hReport = open(self.setReportName, "w")

      # get object to know its header
      param = SignParam()
      rowHead = param.dataHeadCSVShort()
      print("%s,%s" %('#', rowHead) , file=self.hReport)

    # TODO: Handler error conditions

    rowIdx = 0
    # loop through the list and save/write
    for idxRow in self.lReportData:
      rowData = idxRow.dataInCSV()
      # Row index
      rowIdx += 1
      # write to file
      print("%d,%s" %(rowIdx, rowData) , file=self.hReport)

    # close file
    self.hReport.close()


                        
def run_model(signDetectYoloModel):
  """
  """
  yModel = signDetectYoloModel;

  return 0;



# Main function
def main():
  """
  Yolo mode main function

  arguments: TBD
  """

  appSD = SignDetect() 

  # get list of images to process
  appSD.addImgList(DIR_DATA + '\\imgs')

  appSD.yoloModel(DIR_DATA + '\\pt\\(1-4best).pt')



  # start detection on the given list of image
  appSD.execDetectByList()

  #appSD.execDetectByListForBrightness()


  # write report file
  appSD.writeReportBody()

if __name__ == "__main__":
    main()