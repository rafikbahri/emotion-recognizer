# Emotion Recognizer using Python and OpenCV
This project creates a classification model in Machine Learning capable of recognizing human facial emotions.

We consider 8 emotions {0=neutral, 1=anger, 2=contempt, 3=disgust, 4=fear, 5=happy, 6=sadness, 7=surprise} and we use [CK+ Dataset](http://www.consortium.ri.cmu.edu/ckagree/)

## Prerequisites
1. [CK+ Dataset](http://www.consortium.ri.cmu.edu/ckagree/)
2. [Python 2.7](https://www.python.org/download/releases/2.7/)
3. [OpenCV 3, Follow this tuto for Ubuntu](https://www.learnopencv.com/install-opencv3-on-ubuntu/)

## Installation
1. Clone this repo
`git clone https://github.com/rafikbahri/emotion-recognizer.git`

2. Put the images from the dataset in the source_images/ folder.
   Put the emotions from the dataset in the source_emotions/ folder.
   Put your test images in prepared-users-inputs/ folder.
  
3. Organize the dataset in folders with emotions names 
`python dataset_org.py`
   
   Extract faces from the images
`python extract_faces.py`

   Clean up neutral folder by deleting redondant images
`python cleanup_neutral.py`

   Start training and classification script, then test on your own images (or dataset)
`python classi.py`


# References
1. van Gent, P. (2016). Emotion Recognition With Python, OpenCV and a Face Dataset. A tech blog about fun things with Python and embedded electronics. Retrieved from: [here](http://www.paulvangent.com/2016/04/01/emotion-recognition-with-python-opencv-and-a-face-dataset/)
2. Kanade, T., Cohn, J. F., & Tian, Y. (2000). Comprehensive database for facial expression analysis. Proceedings of the Fourth IEEE International Conference on Automatic Face and Gesture Recognition (FG'00), Grenoble, France, 46-53.
3. Lucey, P., Cohn, J. F., Kanade, T., Saragih, J., Ambadar, Z., & Matthews, I. (2010). The Extended Cohn-Kanade Dataset (CK+): A complete expression dataset for action unit and emotion-specified expression. Proceedings of the Third International Workshop on CVPR for Human Communicative Behavior Analysis (CVPR4HB 2010), San Francisco, USA, 94-101.

