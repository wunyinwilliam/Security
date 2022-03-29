import numpy as np
import cv2
import jetson.inference
import jetson.utils
import argparse
import sys

EDGE_PIXEL = 20
DRAW_SIZE = 10
BOTTOM_BAR_HEIGHT = 40


class DrawMask:

    def __init__(self, input_URI, argv):
        self.is_drawing = False             # true if mouse is pressed
        self.ix, self.iy = -1, -1
        self.input = jetson.utils.videoSource(input_URI, argv)
        cv2.namedWindow('Draw Masks')
        cv2.setMouseCallback('Draw Masks', self.draw_circle)
        img = cv2.cvtColor(jetson.utils.cudaToNumpy(self.input.Capture()), cv2.COLOR_BGR2RGB)
        self.draw_img = cv2.imread('draw_mask.png')
        if self.draw_img is None:
            self.draw_img = self.reset_image(img.shape)
        self.bottom_bar = self.create_bottom_bar(img.shape)
        self.start()

    def reset_image(self, shape):
        height, width, _ = shape
        edges = [[0, 0, EDGE_PIXEL, height],
                [0, 0, width, EDGE_PIXEL],
                [0, height-EDGE_PIXEL, width, height],
                [width-EDGE_PIXEL, 0, width, height]]
        image = np.zeros((height, width, 3), np.uint8)        # Create a white image with same dimension
        image[:] = (255, 255, 255)
        self.draw_boundary(image, edges)
        cv2.imwrite('draw_mask.png', image)
        return image

    def draw_boundary(self, frame, boxes):
        for box in boxes:
            mask = np.full((box[3]-box[1], box[2]-box[0], 3), 0).astype(np.uint8)
            frame[box[1]:box[3], box[0]:box[2]] = mask

    def create_bottom_bar(self, shape):
        _, width, _ = shape
        image = np.zeros((BOTTOM_BAR_HEIGHT, width, 3), np.uint8)
        image[:] = (255, 255, 255)
        cv2.line(image, (int(width/3), 0), (int(width/3), BOTTOM_BAR_HEIGHT), (0, 0, 0), 1)
        cv2.line(image, (int(width*2/3), 0), (int(width*2/3), BOTTOM_BAR_HEIGHT), (0, 0, 0), 1)
        cv2.putText(image, 'Clear All', (10, BOTTOM_BAR_HEIGHT-10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)
        cv2.putText(image, 'Undo', (int(width/3)+10, BOTTOM_BAR_HEIGHT-10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)
        cv2.putText(image, 'Eraser', (int(width*2/3)+10, BOTTOM_BAR_HEIGHT-10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)
        return image

    # mouse callback function
    def draw_circle(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.is_drawing = True
            # self.ix, self.iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            cv2.setWindowTitle("Draw Masks", f"({x},{y})")
            if self.is_drawing:
                # cv2.rectangle(self.draw_img, (self.ix, self.iy), (x, y), (0, 0, 0), -1)
                cv2.rectangle(self.draw_img, (int(x-DRAW_SIZE/2), int(y-DRAW_SIZE/2)), (int(x+DRAW_SIZE/2), int(y+DRAW_SIZE/2)), (0, 0, 0), -1)
        elif event == cv2.EVENT_LBUTTONUP:
            self.is_drawing = False
            # cv2.rectangle(self.draw_img, (self.ix, self.iy), (x, y), (0, 0, 0), -1)
            cv2.imwrite('draw_mask.png', self.draw_img)

    def start(self):
        while True:
            img = cv2.cvtColor(jetson.utils.cudaToNumpy(self.input.Capture()), cv2.COLOR_BGR2RGB)
            combine_image = cv2.bitwise_and(img, self.draw_img)
            combine_image_with_bottom_bar = np.concatenate((combine_image, self.bottom_bar), axis=0)
            cv2.imshow('Draw Masks', combine_image_with_bottom_bar)
            if cv2.waitKey(1) & 0xFF == 27:
                cv2.destroyAllWindows()
                sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pose estimation DNN on a video/image stream.",
                                formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.poseNet.Usage() +
                                jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())
    parser.add_argument("input_URI", type=str, default="", nargs='?', help="URI of the input stream")
    opt = parser.parse_known_args()[0]
    draw_mask = DrawMask(input_URI=opt.input_URI, argv=sys.argv)