import numpy as np
import pyscreenshot as ImageGrab
import cv2
import time
import pyautogui

for i in range(1, 4):
    print(i)
    time.sleep(1)

last_time = time.time()

def process_image(original_image):
    processed_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    processed_image = cv2.Canny(processed_image, threshold1=200, threshold2=300)

    return processed_image


while(True):
    im = ImageGrab.grab(bbox=(67, 67, 579, 505))
    #im = im.resize((300, 300))

    print('loop took {}'.format(time.time() - last_time))
    last_time = time.time()

    im_cv = np.array(im)
    #cv2.resize(im_cv, (100, 100))

    im_processed = process_image(im_cv)
    cv2.imshow('window', im_processed)

    #print('right arrow pressed')
    #pyautogui.keyDown('right')

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

