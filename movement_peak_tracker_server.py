import cv2
from statistics import median, mean
import time

def main():
    input_number = select_input()
    #input_number = 1
    #show_video(input_number)
    show_movement_tracking(input_number)

#display movement tracking on a video using opencv
def show_movement_tracking(video_number):
    frame_count = 0
    #array of 100 frame times
    frame_times = [1] * 100
    # Open the Video
    video = cv2.VideoCapture(video_number)

    # read the first frame  of the video as the initial background image
    ret, Prev_frame = video.read()


    # get the start time of each frame
    start_time = time.time()
    while (video.isOpened()):
        new_time = time.time()
        time_diff = new_time - start_time
        frame_times[frame_count % 100] = time_diff
        #calculate the average frame time
        fps = 1 / mean(frame_times)
        start_time = new_time

        frame_count += 1

        ##capture frame by frame
        ret, Current_frame = video.read()

        # Find the absolute difference between the pixels of the prev_frame and current_frame
        # absdiff() will extract just the pixels of the objects that are moving between the two frames
        frame_diff = cv2.absdiff(Current_frame, Prev_frame)
        motion = 0

        # applying Gray scale by converting the images from color to grayscale,
        # This will reduce noise
        gray = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)

        # image smoothing also called blurring is applied using gauisian Blur
        # convert the gray to Gausioan blur to detect motion

        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)[1]

        # fill the gaps by dialiting the image
        # Dilation is applied to binary images.
        dilate = cv2.dilate(thresh, None, iterations=4)

        # Finding contour of moving object
        # Contours can be explained simply as a curve joining all the continuous points (along the boundary),
        # having same color or intensity.
        # For better accuracy, use binary images. So before finding contours, we apply threshold aletrnatives
        (contours, _) = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # calculate the median x and y position using the contours
        x_list = []
        y_list = []
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            x_list.append(x)
            y_list.append(y)
            x_list.append(x + w)
            y_list.append(y + h)



        # loop over the contours
        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)
            #if cv2.contourArea(cnt) > 700 and (x <= 840) and (y >= 150 and y <= 350):
            cv2.rectangle(Prev_frame, (x, y), (x + w, y + h), (255, 255, 0), 3)
            motion += 1
            cv2.putText(Prev_frame, f'person {motion} area {cv2.contourArea(cnt)}', (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        if len(x_list) > 0 and len(y_list) > 0:
            y_center = int(sum(y_list) / len(y_list))
            x_center = int(sum(x_list) / len(x_list))

            x_median = int(median(x_list))
            y_median = int(median(y_list))

            # put a small blue recangle at position y_center and x_center
            cv2.rectangle(Prev_frame, (x_center - 10, y_center - 10), (x_center + 10, y_center + 10), (255, 0, 0), 2)

            #put a small red recangle at position x_median and y_median
            cv2.rectangle(Prev_frame, (x_median - 10, y_median - 10), (x_median + 10, y_median + 10), (0, 0, 255), 2)


        # print the text of "start_time" on lower left corner of the frame
        cv2.putText(Prev_frame, f'fps:{fps}', (10, Prev_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Display the resulting frame
        cv2.imshow("feed", Prev_frame)
        #cv2.imwrite("frame%d.jpg" % frame_count, Prev_frame)

        Prev_frame = Current_frame

        if ret == False:
            break
        if cv2.waitKey(50) == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

def show_movement(video_number):
    cap = cv2.VideoCapture(video_number)
    while(True):
        ret, frame = cap.read()

        # put the words "hello world" on the frame at the center
        #image = cv2.putText(image, 'OpenCV', org, font,
        #           fontScale, color, thickness, cv2.LINE_AA)
        cv2.putText(frame, "Hello World", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # print the dimensions of the frame in text on the frame
        cv2.putText(frame, "Width: {} Height: {}".format(frame.shape[1], frame.shape[0]), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

#Use Opencv to pull webcam image and display it
def show_video(video_number):
    cap = cv2.VideoCapture(video_number)
    while(True):
        ret, frame = cap.read()

        # put the words "hello world" on the frame at the center
        #image = cv2.putText(image, 'OpenCV', org, font,
        #           fontScale, color, thickness, cv2.LINE_AA)
        cv2.putText(frame, "Hello World", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # print the dimensions of the frame in text on the frame
        cv2.putText(frame, "Width: {} Height: {}".format(frame.shape[1], frame.shape[0]), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# determine which videocapture device qto use in opencv
def select_input():
    # build a list of available video capture devices
    devices = []
    for i in range(0, 5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            devices.append(i)
            cap.release()

    if len(devices)==1:
        print("Only one device found")
        return devices[0]

    # prompt the user to select a device
    print("Available video capture devices:")
    for i in range(len(devices)):
        print("{}: {}".format(i, devices[i]))
    device = int(input("Select a device: "))
    return device


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
