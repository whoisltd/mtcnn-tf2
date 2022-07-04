import argparse
import cv2
import time

from mtcnn.mtcnn import MTCNN


def image_demo(img_input, img_output):
    """ mtcnn image demo """
    mtcnn = MTCNN('models/pnet.h5', 'models/rnet.h5', 'models/onet.h5')

    img = cv2.imread(img_input)
    # print('shape', img.shape)
    img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bboxes, landmarks, scores = mtcnn.detect(img_in)
    # print(bboxes.shape)
    img = draw_faces(img, bboxes, landmarks, scores)
    cv2.imwrite(img_output, img)

def draw_faces(img, bboxes, landmarks, scores):
    """ draw bounding boxes and facial landmarks on the image """
    for box, landmark, score in zip(bboxes, landmarks, scores):
        # print(box)
        img = cv2.rectangle(img, (int(box[0]), int(box[1])),
                            (int(box[2]), int(box[3])), (255, 0, 0), 2)
        #crop image 
        for i in range(5):
            x = int(landmark[i])
            y = int(landmark[i + 5])
            img = cv2.circle(img, (x, y), 1, (0, 255, 0))
        img = cv2.putText(img, '{:.2f}'.format(score), (int(box[0]), int(box[1])),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255))
    return img


parser = argparse.ArgumentParser(description='FaceID Demo')
parser.add_argument('--image_input', default='/home/whoisltd/Downloads/peww.jpg',help='Input image')
parser.add_argument('--image_output', default='output.jpg', help='Output image')


if __name__ == '__main__':
    args = parser.parse_args()
    if args.image_input:
        image_demo(args.image_input, args.image_output)
