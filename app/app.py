# imports
import cv2
from random import randint
import PIL.Image as Image
from time import time
from torch import load, device, argmax
from torchvision import transforms
from utils import load_comp_images, overlay_image
from utils import display_score, result, display_result

# TRANSFORMS AND LOADING MODE
img_transform = transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
model = load('final-model.pt', map_location=device('cpu'))
model.eval()

# CONSTANTS
X_OFFSET, Y_OFFSET = 30, 30
COMP_X1, COMP_Y1, COMP_X2, COMP_Y2 = 30, 70, 240, 280
PLAYER_X1, PLAYER_Y1, PLAYER_X2, PLAYER_Y2 = 400, 70, 610, 280
BORDER_THICKNESS = 2

# loading images and initializing a random image to be displayed
comp_images = load_comp_images(
    'comp-images', (COMP_X1, COMP_Y1, COMP_X2, COMP_Y2),
    (X_OFFSET, Y_OFFSET), "png")

# video object to capture image using webcam
video = cv2.VideoCapture(0)

# the time difference used to predict the hand gesture
global_previous_time = time()
global_delta = 0

# the time difference used to determine whether to show play result or not
play_previous = time()
play_delta = 0

# booleans used to decide what to display
show_comp_img = False
make_prediction = True
display_play_result = False

#########################################################################################################################
play = True
# the first opening screen
while True:
    check, frame = video.read()  # returns ret and the frame
    frame = cv2.flip(frame, 1)

    cv2.putText(frame, "PRESS 'p' to continue and 'q' to quit!",
                org=(10, 50), fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=1, color=(23, 5, 72), thickness=2)
    cv2.imshow('Image', frame)
    key = cv2.waitKey(1)
    # quit if 'q' is pressed
    if key == ord('q'):
        play = False
        break
    # play if 'p' is pressed
    if key == ord('p'):
        break
##########################################################################################################################
comp_choices = ['rock', 'paper', 'scissors']
player_score, comp_score = 0, 0

if play:
    while True:
        if player_score == 5 or comp_score == 5:
            break
        # update global time
        global_current = time()
        global_delta += global_current-global_previous_time
        global_previous_time = global_current

        # update play time
        play_current = time()
        play_delta += play_current-play_previous
        play_previous = play_current

        # reads the frame from video stream
        check, frame = video.read()
        frame = cv2.flip(frame, 1)
        # drawing player's box
        cv2.rectangle(
            frame, (PLAYER_X1, PLAYER_Y1), (PLAYER_X2, PLAYER_Y2),
            color=(255, 255, 255),
            thickness=BORDER_THICKNESS
        )
        # drawing computer's box
        cv2.rectangle(
            frame,
            (COMP_X1, COMP_Y1),
            (COMP_X2, COMP_Y2),
            color=(255, 255, 255),
            thickness=BORDER_THICKNESS
        )

        # make prediction in 1 second interval
        if global_delta > 1:
            if make_prediction:
                # MODEL PREDICTION
                img = Image.fromarray(frame)
                # crop out player's box
                cropped = img.crop(
                    (PLAYER_X1, PLAYER_Y1, PLAYER_X2, PLAYER_Y2))
                img = img_transform(cropped).view(1, 3, 224, 224)
                # convert into correct input format to be passed into the model
                probs = model(img)
                pred = argmax(probs).item()
                pred_cls = "rock" if pred == 0 else "paper" if pred == 1 else "scissors" if pred == 2 else "none"
                # if the player is showing a hand gesture
                if pred_cls != "none":
                    # choosing a random image as computer's choice
                    comp_choice = comp_choices[randint(0, 2)]
                    comp_image = comp_images[comp_choices.index(comp_choice)]

                    play_result = result(pred_cls, comp_choice)
                    show_comp_img = True
                    # update player score
                    if play_result == 1:
                        player_score += 1
                    elif play_result == -1:
                        comp_score += 1
                    make_prediction = False
                    display_play_result = True
                    play_delta = 0
                else:
                    show_comp_img = False

            global_delta = 0

        # hide computer's image one minute before next play starts
        if play_delta > 1:
            show_comp_img = False
            # if 2 seconds has passed, start next prediction and hide play result
            if play_delta > 2:
                make_prediction = True
                display_play_result = False
                play_delta = 0

        # display if player has won or not
        if display_play_result:
            display_result(frame, play_result, pred_cls,
                           comp_choice,
                           (PLAYER_X1, PLAYER_Y1, PLAYER_X2, PLAYER_Y2),
                           (COMP_X1, COMP_Y1, COMP_X2, COMP_Y2))
        else:
            cv2.putText(frame, "PLAY !", org=(250, 400),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX,
                        fontScale=1.3, color=(255, 255, 255), thickness=2)

            display_score(frame, player_score, comp_score)

        key = cv2.waitKey(1)

        if show_comp_img:
            overlay_image(frame, comp_image,
                          (COMP_X1, COMP_Y1, COMP_X2, COMP_Y2),
                          (X_OFFSET, Y_OFFSET))

        
        if key == 82:
            PLAYER_Y1, PLAYER_Y2 = PLAYER_Y1-10, PLAYER_Y2-10

        if key == 84:
            PLAYER_Y1, PLAYER_Y2 = PLAYER_Y1+10, PLAYER_Y2+10
        
        if key == 81:
           PLAYER_X1, PLAYER_X2 = PLAYER_X1-10, PLAYER_X2-10

        if key == 83:
            PLAYER_X1, PLAYER_X2 = PLAYER_X1+10, PLAYER_X2+10

        # quit if 'q' is pressed
        if key == ord('q'):
            break

        cv2.imshow("Image", frame)
    ##################################################################################################################################
    while True:
        check, frame = video.read()
        frame = cv2.flip(frame, 1)

        cv2.putText(frame, "GAME OVER",
                    org=(50, 20), fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=0.8, color=(0, 0, 0), thickness=2)
        cv2.imshow('Image', frame)
        key = cv2.waitKey(1)

        if key == ord('q'):
            break
video.release()
