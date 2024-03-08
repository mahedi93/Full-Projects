
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

# Constant
ERR_NO_DATA = -1;

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
    self.sPtFilename = ptFile; 
    self.ptModel = YOLO(ptFile)
    

  def extractParams(self, lPredicted, imgPath, fIdx, beta, alpha, downSize, upSize):
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

          # Test case params++++++++
          # frame index
        param._tcBr = str(beta)
          # frame index
        param._tcCo = str(alpha)
          # frame index
        param._tcUs = str(downSize)
          # frame index
        param._tcDs = str(upSize)

        # store the values
        self.lReportData.append(param)

  def execDetectByList(self):
    """
    """
    fIdx = 0;
    for img in self.lImgList:
      fIdx += 1
      imgPath = self.imgDir + '/' + img;
      #predictOut = self.ptModel.predict(imgPath)
      predictOut = self.ptModel(imgPath)
      self.extractParams(predictOut, str(fIdx))


  def execDetectByListForBrightness(self):
      """
      alpha = 1.0 # 1- 3 Simple contrast control
      beta = 0    # 0 -100 Simple brightness control
      """
      fIdx = 0;
      print(self.lImgList)
      for img in self.lImgList:
        fIdx += 1
        imgPath = self.imgDir + '/' + img;
        print(imgPath)
        for v in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        #for v in [0, 5, 10, 15, 20, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 85, 90, 95, 100]:
          image = cv.imread(imgPath)
          new_image = np.zeros(image.shape, image.dtype)
          alpha=1
          new_image = cv.convertScaleAbs(image, alpha=alpha, beta=v)
          predictOut = self.ptModel.predict(new_image)
          self.extractParams(predictOut, str(fIdx), imgPath, v, alpha,-1,-1)
    


  def execDetectByListForContrast(self):
      """
      alpha = 1.0 # 1- 3 Simple contrast control
      beta = 0    # 0 -100 Simple brightness control
      """
      fIdx = 0;
      print(self.lImgList)
      for img in self.lImgList:
        fIdx += 1
        imgPath = self.imgDir + '/' + img;
        print(imgPath)
        for a in [1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.20, 2.40, 2.60, 2.8, 3.0]:
          image = cv.imread(imgPath)
          new_image = np.zeros(image.shape, image.dtype)
          beta = 0
          new_image = cv.convertScaleAbs(image, alpha=a, beta=beta)
          predictOut = self.ptModel.predict(new_image)
          self.extractParams(predictOut, str(fIdx), imgPath, beta, a,-1,-1)
  

  def execDetectByListforResizedDown(self):
      
      fIdx = 0;
      print(self.lImgList)
      for img in self.lImgList:
        fIdx += 1
        imgPath = self.imgDir + '/' + img;
        #print(imgPath)
        image = cv.imread(imgPath)
        down_width = 300
        down_height = 200
        down_points = (down_width, down_height)
        resized_down = cv2.resize(image, down_points, interpolation= cv2.INTER_LINEAR)
        predictOut = self.ptModel.predict(resized_down)
        self.extractParams(predictOut, str(fIdx), imgPath,resized_down,-1,-1,-1)

  def execDetectByListforResizedUp(self):
      
      fIdx = 0;
      print(self.lImgList)
      for img in self.lImgList:
        fIdx += 1
        imgPath = self.imgDir + '/' + img;
        print(imgPath)
        image = cv.imread(imgPath)
        up_width = 600
        up_height = 400
        up_points = (up_width, up_height)
        resized_up = cv2.resize(image, up_points, interpolation= cv2.INTER_LINEAR)
        predictOut = self.ptModel.predict(resized_up)
        self.extractParams(predictOut, str(fIdx), imgPath, resized_up,-1,-1,-1,-1,-1)



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
      #self.hReport.write("%s;%s;%s;%s\n" %("Frame number", "Class Id", "Class Name", "Confidence"))

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

# run_yolo
def run_app(imgDir):
  """
  Run custom yolo mode

  :param imgDir: path of directory that contains images to process.
  :return: 0 success, else failure
  """ 
  directory = imgDir
  items = os.listdir(directory)
  print(items)
  
  # if not images than return -1 -> Not image data found
  if (items.len() == 0):
    return -1;

  model = YOLO("yolov8m.pt")
  
  #results = model.predict(source="0", show=True)
  #results = 
  #model.predict(source="0")
  #a =  model.predict(source="0", stream=True)
  a =  model.predict(['sample.jpg', 'people.webp'])
  
  f = open("detect-out-realtime1.csv", "a")
  # Header
  f.write("%s;%s;%s;%s\n" %("Frame number", "Class Id", "Class Name", "Confidence"))
  
  for idx, a1 in enumerate(a[0].boxes.xywhn): # change final attribute to desired box format
    cls = int(a[0].boxes.cls[idx].item())
    #print(cls)
    #print(a[0].names[cls])
    confidence = math.ceil(a[0].boxes.conf[0])
    print("%s;%s;%s;%s" %(a[0].path, cls, a[0].names[cls], confidence))
    print("%s;%s;%s;%s" %(a[0].path, cls, a[0].names[cls], confidence), file=f)
    #f.write("%s;%s;%s\n" %(a[0].path, cls, a[0].names[cls]))
  
  print('Closing output file')
  f.close()
  print('Program ended....')
  return 0


# Main function
def main():
  """
  Yolo mode main function

  arguments: TBD
  """
  imgDir = "../../data/img1";

  # Release mode
  #try:
  #  run_yolo(imgDir)
  #except WindowsError as e:
  #  print(e)
  #except:
  #  print("Handling other exceptions")

  # Debug mode
  #run_app(imgDir)

  appSD = SignDetect() 

  # get list of images to process
  appSD.addImgList(DIR_DATA + '\\imgs')

  # get the hander to the signdetect yolo model
  #appSD.yoloModel(DIR_DATA + '\\pt\\sign_detect_1_0.pt')
  appSD.yoloModel(DIR_DATA + '\\pt\\(1-4best).pt')



  # start detection on the given list of image
  #appSD.execDetectByList()

  #appSD.execDetectByListForBrightness()
  #appSD.execDetectByListForContrast()


  appSD.execDetectByListforResizedDown()


  #appSD.execDetectByListforResizedUp()

  # write report file
  appSD.writeReportBody()

  # TODO
  # 1. Ground truth
  # 2. Scenarios, 
  # 3. ...

  # set report file name
  #appSD.setReportName('yolov8m.pt')



if __name__ == "__main__":
    main()