import cv2
from time import time
import os
from os import listdir
from PIL import Image
from random import randint


################################################### OPTIONS ############################################################
# Should the images be saved? if yes, where to save it: train or val?
train = input("Is it training set? (Y/N): ")
# the below input was made so that the person could try changing the code without saving the images if needed
save = input("Do you want to save the images? (Y/N): ")
# which class ?
images_cls = input("Enter class of images (rock/paper/scissors/none/all): ")

######################################### SETTING NO. OF IMAGES ALREADY SAVED ##########################################
# initialize the no. of images in each class folder
rock, paper, scissors, none = 0, 0, 0, 0

dataset_type = "train" if train == "Y" or train == "y" else "val"

if images_cls == "all":
    # Set no. of images in each class folder to the respective variables so that
    # when the images are saved, the images can be correctly named
    for folder in listdir(f'my-images/{dataset_type}'):
        imgs = listdir(f'my-images/{dataset_type}/{folder}')
        nums = []
        # loop through all images in the class folder
        for name in imgs:
            # obtain the image number
            second_part = name.split('-')[-1]
            num = int(second_part.split('.')[0])
            # append it to the list nums
            nums.append(num)
        # set the no. of saved images of class to the maximum number in the list nums
        # which contains the image numbers of the images in the current class folder
        if nums != []:
            if folder == 'c0':
                rock = max(nums)
            elif folder == 'c1':
                paper = max(nums)
            elif folder == 'c2':
                scissors = max(nums)
            else:
                none = max(nums)
else:
    folder = "c0" if images_cls == "rock" else "c1" if images_cls == "paper" else "c2" if images_cls == "scissors" else "c3"
    num_of_images = len(listdir(f'my-images/{dataset_type}/{folder}'))

# coordinates of the vertices of rectangle in the image
x1, y1, x2, y2 = 30, 70, 240, 280
BORDER_THICKNESS = 2

################################################ IMAGE CAPTURING #######################################################
# set video capturing and initialize time
video = cv2.VideoCapture(0)
previous = time()
delta = 0
i = 0

print("Start...")
first = True
while True:
    # update previous time to current and add up the difference
    current = time()
    delta += current - previous
    previous = current
    # obtain frame
    check, frame = video.read()
    # draw the rectangle
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color=(0, 255, 0),
        thickness=BORDER_THICKNESS
    )

    cv2.imshow("Webcam", frame)

    # if this is the first frame being shown, wait for 2 seconds so that person could settle
    # and also bring the opencv window to the front
    if first:
        target_delta = 3
        if images_cls=="all":
            print("rock")
        first = False

    # if time interval > 2
    if delta > target_delta:
        target_delta = 1
        # cut out the rectangle part of the image and convert it to RGB format
        img = frame[y1:y2, x1:x2, ::-1]
        img = cv2.cvtColor(img, code=cv2.COLOR_BGR2RGB)
        # if not c3 class
        ####################################### IMAGES OF ALL CLASSES #################################################
        if images_cls == "all":
            if i == 3:
                i = 0
            cls = "rock" if i == 0 else "paper" if i == 1 else "scissors"
            if cls == "rock":
                rock += 1
                num = rock
            elif cls == "paper":
                paper += 1
                num = paper
            else:
                scissors += 1
                num = scissors
            folder = f"c{i}"
            # if the image should be saved
            if save == "Y" or save == "y":
                cv2.imwrite(f"my-images/{dataset_type}/c{i}/{cls}-{num}.jpg", img)
            print(rock + paper + scissors, "images saved...")
            # print the next hand gesture to be shown by the person
            print("rock" if i == 3 else "paper" if i ==
                  0 else "scissors" if i == 1 else "rock")
            i += 1
        ################################### IMAGES OF ONE OF THE CLASSES ##############################################
        else:
            if save == "Y" or save == "y":
                cv2.imwrite(
                    f"my-images/{dataset_type}/{folder}/{images_cls}-{num_of_images}.jpg", img)
                num_of_images += 1
                print(num_of_images, "saved")

        # reset time difference to 0
        delta = 0

    # press 'q' to exit
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

video.release()
