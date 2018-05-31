##
# dataset_org.py organizes our dataset of images contained in source_images
# by putting the neutral image (soruce_images/part/session[0], eq to the first image in the folder) in sorted_set/neutral 
# and the last image in the folder witch is the emotion (source_images/part/session[-1]) and puts it in the corresponding folder  
# in the sorted_set, this means  sorted_set/emotions[emotion]


import glob
from shutil import copyfile

emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]
participants = glob.glob("source_emotions/*") #Returns a list of all folders with participant numbers")


for x in participants:
    part = "%s" %x[-4:] #store current participant number
    for sessions in glob.glob("%s/*" %x): #Store list of sessions for current participant
        for files in glob.glob("%s/*" %sessions):
            current_session = sessions[-3:]
            #current_session = files
            file = open(files, 'r')

            emotion = int(float(file.readline())) #emotions are encoded as a float, readline as float, then convert to integer.

            tmp=glob.glob("source_images/%s/%s/*" %(part, current_session)) #do same for neutral image
            tmp.sort()
            sourcefile_neutral = tmp[0]            
            print "Neutral Image %s"%sourcefile_neutral
            sourcefile_emotion=tmp[-1]
            print "Emotion Image %s"%sourcefile_emotion
            
            dest_neut = "sorted_set/neutral/%s" %sourcefile_neutral[-21:] #Generate path to put neutral image
            print dest_neut 
            dest_emot = "sorted_set/%s/%s" %(emotions[emotion], sourcefile_emotion[-21:]) #Do same for emotion containing image

            copyfile(sourcefile_neutral, dest_neut) #Copy file
            copyfile(sourcefile_emotion, dest_emot) #Copy file
