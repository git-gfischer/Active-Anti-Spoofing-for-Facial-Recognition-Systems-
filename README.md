# Active Anti-Spoofing System 

https://github.com/git-gfischer/Facial_Recognition_Active_Anti_Spoofing/assets/39131927/6b6ae104-5da8-49da-a712-bf412a05981a

This repo presents anti-spoofing method that captures two face pictures with a single RGB camera at different time and position. SORT (Simple Online Real-time Tracking) tracks every face in frame in order to make the system more secure and robust. This repo uses two independent geometric approach to calculate metrics for classifying a real face from a spoofed one. <br>

ORB is basically a fusion of FAST keypoint detector and BRIEF descriptor with many modifications to enhance the performance. First it use FAST to find keypoints, then apply Harris corner measure to find top N points among them. The Keypoints from each image are compared using KNN (K Nearst Neighbors) algorithm and gernerates a similarity score. <br>

## Depedencies
```bash
   pip3 install -r requirements.txt
   cd retinaface
   make
```

## Run
To run demo using ORB with webcam:          ``` python3 main.py  ```  <br>
To run demo using ORB with any camera:      ``` python3 main.py --source [SOURCE] ``` <br>


## References
ORB:      https://ieeexplore.ieee.org/document/6126544 <br>
SORT:     https://arxiv.org/abs/1602.00763 <br>
KNN:      https://sci-hub.se/10.1109/ICCS45141.2019.9065747 <br>
Harris corner: http://www.bmva.org/bmvc/1988/avc-88-023.pdf <br>
Retina Face:   https://github.com/deepinsight/insightface <br>
