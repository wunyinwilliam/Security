import numpy as np
import cv2
import jetson.inference
import jetson.utils
import argparse
import sys

EDGE_PIXEL = 20
DRAW_SIZE = 10              # For Draw Mode 2, each mouse click is 10x10 pixels
DISMISS_COUNT = 30          # Dismiss the text after this number of frames

class DrawMask:

    def __init__(self, input_URI, argv):
        self.is_drawing = False             # True if mouse is pressed on the video
        self.draw_mode = 1                  # Draw mode 1 is for large area, Draw mode 2 is for small area
        self.is_eraser = False              # True if it is in eraser mode, false if normal draw mode
        self.undo_draw_images = []
        self.input = jetson.utils.videoSource(input_URI, argv)
        cv2.namedWindow('Draw Masks')
        cv2.setMouseCallback('Draw Masks', self.draw)
        img = self.read_cuda_image()                            # Read cuda image once to obtain the shape
        self.draw_img = cv2.imread('draw_mask.png')             # Read previous draw_mask image
        if self.draw_img is None:
            self.draw_img = self.reset_image(img.shape)
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

    def reset_image(self, shape):
        height, width, _ = shape
        edges = [[0, 0, EDGE_PIXEL, height],
                [0, 0, width, EDGE_PIXEL],
                [0, height-EDGE_PIXEL, width, height],
                [width-EDGE_PIXEL, 0, width, height]]
        # Create a white image with same dimension
        image = np.zeros((height, width, 3), np.uint8)        
        image[:] = (255, 255, 255)
        self.draw_edges(image, edges)
        cv2.imwrite('draw_mask.png', image)
        return image

    def draw_edges(self, frame, boxes):
        for box in boxes:
            mask = np.full((box[3]-box[1], box[2]-box[0], 3), 0).astype(np.uint8)
            frame[box[1]:box[3], box[0]:box[2]] = mask

    def create_description_text(self, image):
        _, width, _ = image.shape
        # crop the sub-rect for text from the image
        x, y, w, h = width-220, EDGE_PIXEL, 220-EDGE_PIXEL, 110
        sub_img = image[y:y+h, x:x+w]
        white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
        res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)
        cv2.putText(res, 'Clear All: Ctrl + a', (0, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
        cv2.putText(res, 'Undo: Ctrl + z', (0, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
        cv2.putText(res, 'Large/Small Area: +/-', (0, 60), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
        cv2.putText(res, 'Eraser ON/OFF: SPACE', (0, 80), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
        cv2.putText(res, 'Exit: Esc', (0, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
        # Putting the sub-rect back to its position
        image[y:y+h, x:x+w] = res
        return image

    # mouse callback function
    def draw(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Mouse click in the image
            self.undo_draw_images.append(self.draw_img.copy())
            self.is_drawing = True
            self.ix, self.iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            cv2.setWindowTitle("Draw Masks", f"({x},{y})")
            if self.is_drawing:
                if not self.is_eraser:
                    if self.draw_mode == 1:
                        cv2.rectangle(self.draw_img, (self.ix, self.iy), (x, y), (0, 0, 0), -1)
                    elif self.draw_mode == 2:
                        cv2.rectangle(self.draw_img, (int(x-DRAW_SIZE/2), int(y-DRAW_SIZE/2)), (int(x+DRAW_SIZE/2), int(y+DRAW_SIZE/2)), (0, 0, 0), -1)
                else:
                    if self.draw_mode == 1:
                        cv2.rectangle(self.draw_img, (self.ix, self.iy), (x, y), (255, 255, 255), -1)
                    elif self.draw_mode == 2:
                        cv2.rectangle(self.draw_img, (int(x-DRAW_SIZE/2), int(y-DRAW_SIZE/2)), (int(x+DRAW_SIZE/2), int(y+DRAW_SIZE/2)), (255, 255, 255), -1)
        elif event == cv2.EVENT_LBUTTONUP:
            if self.is_drawing:
                self.is_drawing = False
                cv2.imwrite('draw_mask.png', self.draw_img)

    def start(self):
        while True:
            img = self.read_cuda_image()
            combine_image = cv2.bitwise_and(img, self.draw_img)
            combine_image = self.create_description_text(combine_image)
            if self.dismiss_count > 0:
                self.draw_img_with_text = combine_image.copy()
                cv2.rectangle(self.draw_img_with_text, (0, 0), (200, 50), (255, 255, 255), -1)
                cv2.putText(self.draw_img_with_text, self.text, (10, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)
                cv2.imshow('Draw Masks', self.draw_img_with_text)
                self.dismiss_count -= 1
            else:
                self.text = None
                cv2.imshow('Draw Masks', combine_image)

            key = cv2.waitKey(1)
            if key != -1:
                print('You pressed %d (0x%x), LSB: %d (%s)' % (key, key, key % 256,
                repr(chr(key%256)) if key%256 < 128 else '?'))

            if key == 26:            # Ctrl + Z
                if len(self.undo_draw_images) != 0:
                    self.draw_img = self.undo_draw_images.pop().copy()
                    self.text = 'Undo'
                    self.dismiss_count = DISMISS_COUNT

            elif key == 1:          # Ctrl + A
                self.draw_img = self.reset_image(self.draw_img.shape)
                self.undo_draw_images = []
                self.text = 'Clear All'
                self.dismiss_count = DISMISS_COUNT

            elif key == 32:          # Space Bar
                self.is_eraser = not self.is_eraser
                if self.is_eraser:
                    self.text = 'Eraser'
                else:
                    self.text = 'Pen'
                self.dismiss_count = DISMISS_COUNT

            elif key == 43:          # +
                self.draw_mode = 1
                self.text = 'Large Area'
                self.dismiss_count = DISMISS_COUNT

            elif key == 45:          # -
                self.draw_mode = 2
                self.text = 'Small Area'
                self.dismiss_count = DISMISS_COUNT

            elif key == 27:          # Esc
                cv2.destroyAllWindows()
                sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pose estimation DNN on a video/image stream.",
                                formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.poseNet.Usage() +
                                jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())
    parser.add_argument("input_URI", type=str, default="", nargs='?', help="URI of the input stream")
    opt = parser.parse_known_args()[0]
    draw_mask = DrawMask(input_URI=opt.input_URI, argv=sys.argv)