import pickle
import numpy as np
import cv2
import jetson.inference
import jetson.utils
import argparse
import sys

DISMISS_COUNT = 30          # Dismiss the text after this number of frames


class AddWords:

    def __init__(self, input_URI, argv):
        self.dismiss_count = 0              # Dismiss the text if dismiss_count == 0
        self.input = jetson.utils.videoSource(input_URI, argv)
        cv2.namedWindow('Add Words')
        # cv2.namedWindow('Add Words', cv2.WND_PROP_FULLSCREEN)
        # cv2.setWindowProperty('Add Words', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setMouseCallback('Add Words', self.draw)
        self.word_img_white = cv2.imread('word_overlay.png')             # Read previous word_overlay image
        self.word_img_black = cv2.imread('word_underlay.png')             # Read previous word_underlay image
        if self.word_img_white is None:
            img = self.read_cuda_image()                           # Read cuda image once to obtain the shape
            self.reset_image(img.shape, True)
        if self.word_img_black is None:
            img = self.read_cuda_image()                           # Read cuda image once to obtain the shape
            self.reset_image(img.shape, False)
        self.is_typing = False
        self.is_delete = False
        self.is_white = True
        self.text_size = 2
        self.text_input = ''
        self.text_location = None
        self.text_pointer_location_x = 0
        self.image_without_new_words = None
        self.word_img_without_pointer = None
        try:
            with open( "word_history.pkl", 'rb') as file:
                variables = pickle.load(file)
                self.word_history = variables["word_history"]
        except:
            self.word_history = []
            variables = { "word_history": self.word_history }
            with open( "word_history.pkl", 'wb') as file:
                pickle.dump(variables, file)
        self.start()


    def read_cuda_image(self):
        # load the image from CUDA memory
        rgb_img = self.input.Capture()
        # convert to BGR, since that's what OpenCV expects
        bgr_img = jetson.utils.cudaAllocMapped(width=rgb_img.width,
                    height=rgb_img.height,
                    format='bgr8')
        jetson.utils.cudaConvertColor(rgb_img, bgr_img)
        # Make sure the GPU is done work before we convert to cv2
        jetson.utils.cudaDeviceSynchronize()
        # convert to cv2 image (cv2 images are numpy arrays)
        img = jetson.utils.cudaToNumpy(bgr_img)
        return img


    # Create a black image with same dimension
    def reset_image(self, shape, white):
        height, width, _ = shape
        image = np.zeros((height, width, 3), np.uint8)        
        image[:] = (0, 0, 0)
        if white:
            self.word_img_white = image
        else:
            self.word_img_black = image
        self.word_history = []
        self.save_word_img()


    def create_description_text(self, image):
        _, width, _ = image.shape
        # crop the sub-rect for text from the image
        x, y, w, h = width-300, 0, 300, 150
        sub_img = image[y:y+h, x:x+w]
        white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
        res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)
        cv2.putText(res, 'Double Click: Start Typing', (0, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
        cv2.putText(res, 'Enter: Finish Typing & Save', (0, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
        cv2.putText(res, 'White/Black Text: F1', (0, 60), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
        cv2.putText(res, 'Text Size: + / -', (0, 80), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
        cv2.putText(res, 'Delete + Double Click: Remove', (0, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
        cv2.putText(res, 'Clear All: F12', (0, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
        cv2.putText(res, 'Exit: Esc', (0, 140), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
        # Putting the sub-rect back to its position
        image[y:y+h, x:x+w] = res
        return image


    # mouse callback function
    def draw(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            if not self.is_delete:
                if self.is_typing:
                    self.stop_typing(True)
                self.is_typing = True
                self.is_delete = False
                self.text_location = (x, y)
                if self.is_white:
                    self.image_without_new_words = self.word_img_white.copy()
                    self.word_img_without_pointer = self.word_img_white.copy()
                    cv2.rectangle(self.word_img_white, (self.text_location[0], self.text_location[1]-self.text_size*15), (self.text_location[0]+1, self.text_location[1]+10), (255, 255, 255), -1)
                else:
                    self.image_without_new_words = self.word_img_black.copy()
                    self.word_img_without_pointer = self.word_img_black.copy()
                    cv2.rectangle(self.word_img_black, (self.text_location[0], self.text_location[1]-self.text_size*15), (self.text_location[0]+1, self.text_location[1]+10), (255, 255, 255), -1)
                self.text = 'Enter Words'
                self.dismiss_count = DISMISS_COUNT
            else:       # Delete
                for word_history in self.word_history:
                    print(word_history)
                    if x > word_history[1][0] and x < word_history[1][0]+word_history[2] and y > word_history[1][1]-30 and y < word_history[1][1]+30:
                        self.word_history.remove(word_history)
                        break
                temp_word_history = self.word_history.copy()
                self.reset_image(self.word_img_white.shape, True)
                self.reset_image(self.word_img_black.shape, False)
                self.word_history = temp_word_history.copy()
                for word_history in self.word_history:
                    if word_history[3]:
                        cv2.putText(self.word_img_white, word_history[0], word_history[1], cv2.FONT_HERSHEY_PLAIN, word_history[4], (255, 255, 255), 2)
                    else:
                        cv2.putText(self.word_img_black, word_history[0], word_history[1], cv2.FONT_HERSHEY_PLAIN, word_history[4], (255, 255, 255), 2)
                self.save_word_img()


    def stop_typing(self, save):
        if save:
            if self.word_img_without_pointer is not None:
                if self.is_white:
                    self.word_img_white = self.word_img_without_pointer.copy()
                    if self.text_input != '' and self.text_location is not None and self.text_pointer_location_x != 0:
                        self.word_history.append((self.text_input, self.text_location, self.text_pointer_location_x, True, self.text_size))
                else:
                    self.word_img_black = self.word_img_without_pointer.copy()
                    if self.text_input != '' and self.text_location is not None and self.text_pointer_location_x != 0:
                        self.word_history.append((self.text_input, self.text_location, self.text_pointer_location_x, False, self.text_size))
            self.save_word_img()
        else:
            if self.is_white:
                self.word_img_white = self.image_without_new_words.copy()
            else:
                self.word_img_black = self.image_without_new_words.copy()
        self.is_typing = False
        self.text_input = ''
        self.text_location = None
        self.text_pointer_location_x = 0
        self.image_without_new_words = None
        self.word_img_without_pointer = None


    def save_word_img(self):
        cv2.imwrite('word_overlay.png', self.word_img_white)
        cv2.imwrite('word_underlay.png', self.word_img_black)
        variables = { "word_history": self.word_history }
        with open( "word_history.pkl", 'wb') as file:
            pickle.dump(variables, file)
        print(self.word_history)


    def start(self):
        while True:
            img = self.read_cuda_image()
            image_white_only = cv2.bitwise_or(img, self.word_img_white)
            word_img_inv = cv2.bitwise_not(self.word_img_black)
            image_without_description = cv2.bitwise_and(image_white_only, word_img_inv)
            combine_image = self.create_description_text(image_without_description)
            if self.dismiss_count > 0:
                word_img_with_text = combine_image.copy()
                cv2.rectangle(word_img_with_text, (0, 0), (230, 50), (255, 255, 255), -1)
                cv2.putText(word_img_with_text, self.text, (10, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)
                cv2.imshow('Add Words', word_img_with_text)
                self.dismiss_count -= 1
            else:
                self.text = None
                cv2.imshow('Add Words', combine_image)

            key = cv2.waitKey(1)
            if key != -1:
                print('You pressed %d (0x%x), LSB: %d (%s)' % (key, key, key % 256,
                repr(chr(key%256)) if key%256 < 128 else '?'))

            if (key > 31) and (key < 127) and (key != 43) and (key != 45):        # letters or integers
                if self.is_typing:
                    self.text_input += chr(key%256) if key%256 < 128 else '?' 
                    self.text_pointer_location_x = len(self.text_input) * self.text_size * 10
                    if self.is_white:
                        self.word_img_white = self.image_without_new_words.copy()
                        cv2.putText(self.word_img_white, self.text_input, self.text_location, cv2.FONT_HERSHEY_PLAIN, self.text_size, (255, 255, 255), 2)
                        self.word_img_without_pointer = self.word_img_white.copy()
                        cv2.rectangle(self.word_img_white, (self.text_location[0]+self.text_pointer_location_x, self.text_location[1]-self.text_size*15), (self.text_location[0]+self.text_pointer_location_x+1, self.text_location[1]+10), (255, 255, 255), -1)
                    else:
                        self.word_img_black = self.image_without_new_words.copy()
                        cv2.putText(self.word_img_black, self.text_input, self.text_location, cv2.FONT_HERSHEY_PLAIN, self.text_size, (255, 255, 255), 2)
                        self.word_img_without_pointer = self.word_img_black.copy()
                        cv2.rectangle(self.word_img_black, (self.text_location[0]+self.text_pointer_location_x, self.text_location[1]-self.text_size*15), (self.text_location[0]+self.text_pointer_location_x+1, self.text_location[1]+10), (255, 255, 255), -1)

            elif key == 8:                       # Backspace
                if self.is_typing and len(self.text_input) > 0:
                    self.text_input = self.text_input[:-1]
                    self.text_pointer_location_x = len(self.text_input) * self.text_size * 10
                    if self.is_white:
                        self.word_img_white = self.image_without_new_words.copy()
                        cv2.putText(self.word_img_white, self.text_input, self.text_location, cv2.FONT_HERSHEY_PLAIN, self.text_size, (255, 255, 255), 2)
                        self.word_img_without_pointer = self.word_img_white.copy()
                        cv2.rectangle(self.word_img_white, (self.text_location[0]+self.text_pointer_location_x, self.text_location[1]-self.text_size*15), (self.text_location[0]+self.text_pointer_location_x+1, self.text_location[1]+10), (255, 255, 255), -1)
                    else:
                        self.word_img_black = self.image_without_new_words.copy()
                        cv2.putText(self.word_img_black, self.text_input, self.text_location, cv2.FONT_HERSHEY_PLAIN, self.text_size, (255, 255, 255), 2)
                        self.word_img_without_pointer = self.word_img_black.copy()
                        cv2.rectangle(self.word_img_black, (self.text_location[0]+self.text_pointer_location_x, self.text_location[1]-self.text_size*15), (self.text_location[0]+self.text_pointer_location_x+1, self.text_location[1]+10), (255, 255, 255), -1)
                    self.text = 'Backspace'
                    self.dismiss_count = DISMISS_COUNT

            elif key == 13:          # Enter
                if self.is_typing:
                    self.stop_typing(True)
                    self.text = 'Enter'
                    self.dismiss_count = DISMISS_COUNT

            elif key == 190:          # F1: White/Black Text
                if self.is_typing:
                    self.stop_typing(True)
                self.is_white = not self.is_white
                if self.is_white:
                    self.text = 'White Text'
                else:
                    self.text = 'Black Text'
                self.dismiss_count = DISMISS_COUNT

            elif key == 43:           # +: Increase Text Size
                if self.text_size < 5:
                    self.text_size += 1
                self.text_pointer_location_x = len(self.text_input) * self.text_size * 10
                if self.is_typing:
                    if self.is_white:
                        self.word_img_white = self.image_without_new_words.copy()
                        cv2.putText(self.word_img_white, self.text_input, self.text_location, cv2.FONT_HERSHEY_PLAIN, self.text_size, (255, 255, 255), 2)
                        self.word_img_without_pointer = self.word_img_white.copy()
                        cv2.rectangle(self.word_img_white, (self.text_location[0]+self.text_pointer_location_x, self.text_location[1]-self.text_size*15), (self.text_location[0]+self.text_pointer_location_x+1, self.text_location[1]+10), (255, 255, 255), -1)
                    else:
                        self.word_img_black = self.image_without_new_words.copy()
                        cv2.putText(self.word_img_black, self.text_input, self.text_location, cv2.FONT_HERSHEY_PLAIN, self.text_size, (255, 255, 255), 2)
                        self.word_img_without_pointer = self.word_img_black.copy()
                        cv2.rectangle(self.word_img_black, (self.text_location[0]+self.text_pointer_location_x, self.text_location[1]-self.text_size*15), (self.text_location[0]+self.text_pointer_location_x+1, self.text_location[1]+10), (255, 255, 255), -1)
                self.text = f'Text Size: {self.text_size}'
                self.dismiss_count = DISMISS_COUNT

            elif key == 45:           # -: Decrease Text Size
                if self.text_size > 1:
                    self.text_size -= 1
                self.text_pointer_location_x = len(self.text_input) * self.text_size * 10
                if self.is_typing:
                    if self.is_white:
                        self.word_img_white = self.image_without_new_words.copy()
                        cv2.putText(self.word_img_white, self.text_input, self.text_location, cv2.FONT_HERSHEY_PLAIN, self.text_size, (255, 255, 255), 2)
                        self.word_img_without_pointer = self.word_img_white.copy()
                        cv2.rectangle(self.word_img_white, (self.text_location[0]+self.text_pointer_location_x, self.text_location[1]-self.text_size*15), (self.text_location[0]+self.text_pointer_location_x+1, self.text_location[1]+10), (255, 255, 255), -1)
                    else:
                        self.word_img_black = self.image_without_new_words.copy()
                        cv2.putText(self.word_img_black, self.text_input, self.text_location, cv2.FONT_HERSHEY_PLAIN, self.text_size, (255, 255, 255), 2)
                        self.word_img_without_pointer = self.word_img_black.copy()
                        cv2.rectangle(self.word_img_black, (self.text_location[0]+self.text_pointer_location_x, self.text_location[1]-self.text_size*15), (self.text_location[0]+self.text_pointer_location_x+1, self.text_location[1]+10), (255, 255, 255), -1)
                self.text = f'Text Size: {self.text_size}'
                self.dismiss_count = DISMISS_COUNT

            elif key == 255:          # Delete: Normal/Delete Mode
                if self.is_typing:
                    self.stop_typing(True)
                self.is_delete = not self.is_delete
                if self.is_delete:
                    self.text = 'Delete Mode'
                    self.dismiss_count = 10000
                else:
                    self.text = 'Normal Mode'
                    self.dismiss_count = DISMISS_COUNT

            elif key == 201:          # F12: Clear All
                self.reset_image(self.word_img_white.shape, True)
                self.reset_image(self.word_img_black.shape, False)
                if self.is_typing:
                    self.stop_typing(True)
                self.is_delete = False
                self.text = 'Clear All'
                self.dismiss_count = DISMISS_COUNT

            elif key == 27:          # Esc
                if self.is_typing:
                    self.stop_typing(False)
                else:
                    cv2.destroyAllWindows()
                    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pose estimation DNN on a video/image stream.",
                                formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.poseNet.Usage() +
                                jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())
    parser.add_argument("input_URI", type=str, default="", nargs='?', help="URI of the input stream")
    opt = parser.parse_known_args()[0]
    add_words = AddWords(input_URI=opt.input_URI, argv=sys.argv)