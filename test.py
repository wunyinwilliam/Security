import numpy as np
import cv2
import sys
import tkinter as tk
from tkinter.simpledialog import askstring

DISMISS_COUNT = 30          # Dismiss the text after this number of frames


class AddWords:

    def __init__(self):
        self.dismiss_count = 0              # Dismiss the text if dismiss_count == 0
        self.undo_word_images = []
        cv2.namedWindow('Add Words', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Add Words', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setMouseCallback('Add Words', self.draw)
        self.cap = cv2.VideoCapture(0)

        # Delete later
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        _, img = self.cap.read()
        self.word_img = cv2.imread('word_overlay.png')             # Read previous word_overlay image
        if self.word_img is None:
            self.word_img = self.reset_image(img.shape)
        self.start()


    # Create a black image with same dimension
    def reset_image(self, shape):
        height, width, _ = shape
        image = np.zeros((height, width, 3), np.uint8)

        # Delete later
        image = np.zeros((height, width, 3), np.uint8)

        image[:] = (0, 0, 0)
        cv2.imwrite('word_overlay.png', image)
        return image


    def create_description_text(self, image):
        _, width, _ = image.shape
        # crop the sub-rect for text from the image
        x, y, w, h = width-220, 0, 220, 70
        sub_img = image[y:y+h, x:x+w]
        white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
        res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)
        cv2.putText(res, 'Clear All: Ctrl + a', (0, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
        cv2.putText(res, 'Undo: Ctrl + z', (0, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
        cv2.putText(res, 'Exit: Esc', (0, 60), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
        # Putting the sub-rect back to its position
        image[y:y+h, x:x+w] = res
        return image


    # mouse callback function
    def draw(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            root = tk.Tk()
            root.withdraw()
            texting = askstring('Input Text', 'What text you want to put down?')
            if texting is not None and texting != '':
                self.undo_word_images.append(self.word_img.copy())
                width = 20 * len(texting) + 10
                cv2.rectangle(self.word_img, (x-10, y-30), (x+width, y+10), (255, 255, 255), -1)
                cv2.putText(self.word_img, texting, (x, y), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
                cv2.imwrite('word_overlay.png', self.word_img)


    def start(self):
        while True:
            _, img = self.cap.read()
            image_without_description = cv2.bitwise_or(img, self.word_img)

            # Delete later
            crop_word_img = self.word_img[0:720, 0:1280]
            image_without_description = cv2.bitwise_or(img, crop_word_img)

            combine_image = self.create_description_text(image_without_description)

            if self.dismiss_count > 0:
                self.draw_img_with_text = combine_image.copy()
                cv2.rectangle(self.draw_img_with_text, (0, 0), (200, 50), (255, 255, 255), -1)
                cv2.putText(self.draw_img_with_text, self.text, (10, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (120, 120, 120), 1)
                cv2.imshow('Add Words', self.draw_img_with_text)
                self.dismiss_count -= 1
            else:
                self.text = None
                cv2.imshow('Add Words', combine_image)

            key = cv2.waitKey(1)
            if key != -1:
                print('You pressed %d (0x%x), LSB: %d (%s)' % (key, key, key % 256,
                repr(chr(key%256)) if key%256 < 128 else '?'))

            if key == 26:            # Ctrl + Z
                if len(self.undo_word_images) != 0:
                    self.word_img = self.undo_word_images.pop().copy()
                    self.text = 'Undo'
                    self.dismiss_count = DISMISS_COUNT

            elif key == 1:          # Ctrl + A
                self.word_img = self.reset_image(self.word_img.shape)
                self.undo_word_images = []
                self.text = 'Clear All'
                self.dismiss_count = DISMISS_COUNT

            elif key == 27:          # Esc
                cv2.destroyAllWindows()
                sys.exit(0)


if __name__ == "__main__":
    add_words = AddWords()