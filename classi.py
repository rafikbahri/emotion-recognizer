import cv2
import glob
import random
import numpy as np
import os.path
# HAAR Face Classifiers

faceDet = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
faceDet2 = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
faceDet3 = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
faceDet4 = cv2.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")

#Change
emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] #Emotion list
#To:
#emotions = ["neutral", "anger", "disgust", "happy", "surprise"]
fishface = cv2.face.FisherFaceRecognizer_create() #Initialize fisher face classifier

data = {}

def get_files(emotion): #Define function to get file list, randomly shuffle it and split 80/20
    files = glob.glob("dataset/%s/*" %emotion)
    random.shuffle(files)
    training = files[:int(len(files)*0.8)] #get first 80% of file list
    prediction = files[-int(len(files)*0.2):] #get last 20% of file list
    return training, prediction

def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    print "Random shuffling the dataset.."
    print "Making sets..."
    for emotion in emotions:
        training, prediction = get_files(emotion)
        #Append data to training and prediction list, and generate labels 0-7
        for item in training:
            image = cv2.imread(item) #open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
            training_data.append(gray) #append image array to training data list
            training_labels.append(emotions.index(emotion))
    
        for item in prediction: #repeat above process for prediction set
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            prediction_data.append(gray)
            prediction_labels.append(emotions.index(emotion))

    return training_data, training_labels, prediction_data, prediction_labels

def run_recognizer():
    training_data, training_labels, prediction_data, prediction_labels = make_sets()
    
    print "Training fisher face classifier with 80percent of the dataset"
    print "size of training set is:", len(training_labels), "images"
    fishface.train(training_data, np.asarray(training_labels))

    print "Predicting classification set 20percent of the dataset"
    cnt = 0
    correct = 0
    incorrect = 0
    for image in prediction_data:
        pred, conf = fishface.predict(image)
        if pred == prediction_labels[cnt]:
            correct += 1
            cnt += 1
        else:
            cv2.imwrite("difficult/%s_%s_%s.jpg" %(emotions[prediction_labels[cnt]], emotions[pred], cnt), image)
            incorrect += 1
            cnt += 1
    return ((100*correct)/(correct + incorrect))



    

def predict_image():
    metascore=[]
    for i in range(0,2):
        print "Iteration number: %d"%i
        correct = run_recognizer()
        print "Got ", correct, " percent correct!\n"
        metascore.append(correct)
    print "\n\nend score:", np.mean(metascore), "percent correct!\n"

    print "Now predicting given images.."
    resp=raw_input("\033[94m Give image path: (Q to quit)\n")
    while resp != "Q":
        if os.path.exists(resp):
            path_to_prepared_img=prepare_img(resp)
            prep_img=cv2.imread(path_to_prepared_img,0)
            cv2.imshow(path_to_prepared_img,prep_img)            
            pred=fishface.predict(prep_img)    
            print "\033[92m %s is %s\n"%(resp,emotions[pred[0]])
            cv2.waitKey(0)
            resp=raw_input("\033[94m Give image path: (Q to quit)\n")
        else :
            resp=raw_input("Give a valid path for the image\n")
    print "\033[0m-- END -- "



def prepare_img(im_path):
    input_path=glob.glob(im_path)[0]
    img=cv2.imread(input_path,0)
    gray=img

    #Detect face using 4 different classifiers
    face = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    face2 = faceDet2.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    face3 = faceDet3.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    face4 = faceDet4.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)

    #Go over detected faces, stop at first detected face, return empty if no face.
    if len(face) == 1:
        facefeatures = face
    elif len(face2) == 1:
        facefeatures == face2
    elif len(face3) == 1:
        facefeatures = face3
    elif len(face4) == 1:
        facefeatures = face4
    else:
        facefeatures = ""
    #Cut and save face
    for (x, y, w, h) in facefeatures: #get coordinates and size of rectangle containing faceinputs
        gray = gray[y:y+h, x:x+w] #Cut the frame to size
        prepared_path=""
        try:
            out = cv2.resize(gray, (350, 350)) #Resize face so all images have same size
            prepared_path="prepared-users-inputs/%s_prep.jpg"%input_path[8:-4]
            cv2.imwrite(prepared_path, out) #Write image
        except:
            pass
    
    return prepared_path


# #Now run it
# metascore = []
# for i in range(0,5):
#     print "Iteration number: %d"%i
#     correct = run_recognizer()
#     print "Got", correct, "percent correct!"
#     metascore.append(correct)
# print "\n\nend score:", np.mean(metascore), "percent correct!"

predict_image()

