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
    '../comp-images', (COMP_X1, COMP_Y1, COMP_X2, COMP_Y2),
    (X_OFFSET, Y_OFFSET), "png")

video = cv2.VideoCapture(0)

global_previous_time = time()
global_delta = 0

play_previous = time()
play_delta = 0

prev_pred_cls = "none"
show_comp_img = False
make_prediction = True
display_play_result = False

comp_choices = ['rock', 'paper', 'scissors']
player_score, comp_score = 0, 0
play = True
while True:
    ret, frame = video.read()  # returns ret and the frame
    frame = cv2.flip(frame, 1)

    cv2.putText(frame, "Press 'p' to continue and 'q' to quit!", org=(100, 400),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=0.8, color=(255, 255, 255), thickness=2)
    cv2.imshow('Image', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        play = False
        break
    if key == ord('p'):
        break

if play:
    while True:
        global_current = time()
        global_delta += global_current-global_previous_time
        global_previous_time = global_current

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

        if global_delta > 1:
            if make_prediction:
                # MODEL PREDICTION
                # rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                img = Image.fromarray(frame)
                cropped = img.crop(
                    (PLAYER_X1, PLAYER_Y1, PLAYER_X2, PLAYER_Y2))
                img = img_transform(cropped).view(1, 3, 224, 224)
                probs = model(img)
                pred = argmax(probs).item()
                pred_cls = "rock" if pred == 0 else "paper" if pred == 1 else "scissors" if pred == 2 else "none"
                comp_choice = comp_choices[randint(0, 2)]
                comp_image = comp_images[comp_choices.index(comp_choice)]
                if pred_cls != "none":
                    comp_choice = comp_choices[randint(0, 2)]
                    comp_img = comp_images[comp_choices.index(comp_choice)]
                    play_result = result(pred_cls, comp_choice)
                    show_comp_img = True
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

        if play_delta > 1:
            show_comp_img = False
            if play_delta > 2:
                make_prediction = True
                display_play_result = False
                play_delta = 0

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
            overlay_image(frame, comp_img,
                          (COMP_X1, COMP_Y1, COMP_X2, COMP_Y2),
                          (X_OFFSET, Y_OFFSET))

            # quit if 'q' is pressed
        if key == ord('q'):
            break

        cv2.imshow("Image", frame)

video.release()
