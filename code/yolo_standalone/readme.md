## Yolo standalone detection

### General setup
* The code repository is at github of ultralytics,[^1]
* The yolov8m.pt(https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt) is sample model

### Output
* detect-out.csv is generated
* Snip of content:
  ```csv
  Frame number;Class Id;Class Name
  sample.jpg;16;dog
  sample.jpg;15;cat
  ```

### logs
* pkgs.txt: 
  * Contains package installed
  * Exe added the Python virtual env folder

[^1]: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics): is a cutting-edge, state-of-the-art (SOTA) model that builds upon the success of previous YOLO versions 