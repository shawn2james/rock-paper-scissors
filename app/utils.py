import cv2


def load_comp_images(root, coordinates, offsets, ext="jpg"):
    """
    loads the already saved rock-paper-scissors images and resizes them
    :param root: root directory that contains the images rock.png, paper.png and scissors.png
    :param coordinates: a sequence (x1, y1, x2, y2) that are the coordinates of the box in which
                        these images are being displayed when playing
    :param offsets: padding to be given to the image with respect to the rectangular box
    :param ext: extension of the images
    :return: list of images arrays of rock-paper-scissors images
    """

    # extracting coordinates and offsets
    x1, y1, x2, y2 = coordinates
    x_offset, y_offset = offsets

    # reading in the images
    rock = cv2.imread(f"{root}/rock.{ext}")
    paper = cv2.imread(f"{root}/paper.{ext}")
    scissors = cv2.imread(f"{root}/scissors.{ext}")

    # setting width and height for resizing the images by applying offsets
    width, height = (x2 - x_offset) - \
        (x1 + x_offset), (y2 - y_offset) - (y1 + y_offset)

    # resizing images with the above width and height
    rock = cv2.resize(rock, (width, height), interpolation=cv2.INTER_AREA)
    paper = cv2.resize(paper, (width, height), interpolation=cv2.INTER_AREA)
    scissors = cv2.resize(scissors, (width, height),
                          interpolation=cv2.INTER_AREA)

    return [rock, paper, scissors]


def overlay_image(l_img, s_img, coordinates, offsets):
    """
    places a smaller image over a larger image given the coordinates and offsets to be applied
    :param l_img: the larger image over which the smaller image is to be placed
    :param s_img: the smaller image to be placed over the larger image
    :param coordinates: a squence of coordinates (x1, y1, x2, y2) at which the smaller image is to be placed with
                        respect ot the larger image
    :param offsets: padding to be given to the smaller image with respect to the box formed by the given coordinates
    :return: None (applies the overlay inplace)
    """
    # extracting coordinates and offsets
    x1, y1, x2, y2 = coordinates
    x_offset, y_offset = offsets
    # obtaining the alpha channels of s_img and l_img
    alpha_s = s_img[:, :, 2] / 255.0
    alpha_l = 1.0 - alpha_s

    # applying offsets to coordinates
    x1, y1, x2, y2 = x1 + x_offset, y1 + y_offset, x2 - x_offset, y2 - y_offset
    # placing s_img over l_img
    for c in range(0, 3):
        l_img[y1:y2, x1:x2, c] = (alpha_s * s_img[:, :, c] +
                                  alpha_l * l_img[y1:y2, x1:x2, c])


def result(player_choice, comp_choice):
    """
    returns the update to be given to player score
    :param player_choice: the player's choice => rock/paper/scissors
    :param comp_choice: the computer's choice => rock/paper/scissors
    :return: -1, 0 or 1 depending on the user and computer choices
    """
    if player_choice == comp_choice:
        return 0
    elif player_choice == "rock":
        if comp_choice == "paper":
            return -1
        else:
            return 1
    elif player_choice == "paper":
        if comp_choice == "rock":
            return 1
        else:
            return -1
    elif player_choice == "scissors":
        if comp_choice == "rock":
            return -1
        else:
            return 1
    

def display_result(frame, result, player_choice, comp_choice, player_rect, comp_rect):
    PLAYER_X1, PLAYER_Y1, PLAYER_X2, PLAYER_Y2 = player_rect
    COMP_X1, COMP_Y1, COMP_X2, COMP_Y2 = comp_rect

    cv2.putText(frame, player_choice, org=(PLAYER_X1+50, PLAYER_Y2+40),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=0.9, color=(0, 0, 0), thickness=2)
    cv2.putText(frame, comp_choice, org=(COMP_X1+50, COMP_Y2+40),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=0.9, color=(0, 0, 0), thickness=2)

    ORG, FONTFACE, FONTSCALE, THICKNESS = (250, 400), cv2.FONT_HERSHEY_COMPLEX, 1.3, 2

    if result == 0:
        display_text, color = "DRAW...", (255, 255, 255)
    elif result == 1:
        display_text, color = "YOU WIN...", (0, 255, 0)
    else:
        display_text, color = "YOU LOSE...", (0, 0, 255)

    cv2.putText(frame, text=display_text, org=ORG, fontFace=FONTFACE,
                fontScale=FONTSCALE, color=color, thickness=THICKNESS)

    comp_color = (0, 255, 0) if color == (0, 0, 255) else (
        0, 0, 255) if color == (0, 255, 0) else (255, 255, 255)
    cv2.rectangle(frame, (PLAYER_X1, PLAYER_Y1),
                    (PLAYER_X2, PLAYER_Y2), color=color, thickness=2)
    cv2.rectangle(frame, (COMP_X1, COMP_Y1), (COMP_X2, COMP_Y2),
                    color=comp_color, thickness=2)


def display_score(frame, player_score, comp_score, score_title_pos=(560, 20)):
    """
    displays score in the frame

    Args:
        frame (ndarray): the frame on which score is to be displayed
        player_score (int): player's score
        comp_score (int): computer's score
        score_title_pos (tuple): the coorodinates at which the score should be displayed in the frame. Defaults to (560, 20).
    """
    score_title_x, score_title_y = score_title_pos
    player_score_text = f'PLAYER: {player_score}'
    comp_score_text = f'COMPUTER: {comp_score}'

    cv2.putText(frame, 'SCORE:', org=(480, 20), fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=0.7, color=(0, 0, 255), thickness=2)
    cv2.putText(frame, player_score_text, org=(score_title_x-150, score_title_y+20),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=0.5, color=(0, 0, 255), thickness=2)
    cv2.putText(frame, comp_score_text, org=(score_title_x-50, score_title_y+20),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=0.5, color=(0, 0, 255), thickness=2)