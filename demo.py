import sys
sys.path.append('core')

import argparse
import cv2
import numpy as np
import torch
from PIL import Image
import threading
import queue
from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

DEVICE = 'cuda'

def setup_camera(index):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"Cannot open camera {index}")
        return None
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    return cap

def read_matrices(path):
    try:
        file_storage = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
        x_mat = file_storage.getNode("x").mat()
        y_mat = file_storage.getNode("y").mat()
    except Exception as e:
        print("Error reading file:", e)
        x_mat, y_mat = None, None
    return x_mat, y_mat

def image_rectification(frame, x_mat, y_mat):
    rectified = cv2.remap(frame, x_mat, y_mat, cv2.INTER_LINEAR)
    return rectified

def load_image_from_frame(frame):
    img = np.array(frame).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def viz(img, flo):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    cv2.imshow('Optical Flow', img_flo[:, :, [2,1,0]]/255.0)
    cv2.waitKey(1)

def frame_reader(cap, frame_queue, x_mat, y_mat, rotate=False):
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame")
            break
        if rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        frame_rectified = image_rectification(frame, x_mat, y_mat)
        frame_queue.put(frame_rectified)

def optical_flow_processing(model, frame_queue):
    with torch.no_grad():
        prev_img = None
        while True:
            if frame_queue.empty():
                continue
            frame = frame_queue.get()
            curr_img = load_image_from_frame(frame)

            if prev_img is not None:
                padder = InputPadder(prev_img.shape)
                prev_img_padded, curr_img_padded = padder.pad(prev_img, curr_img)

                flow_low, flow_up = model(prev_img_padded, curr_img_padded, iters=20, test_mode=True)
                viz(prev_img_padded, flow_up)

            prev_img = curr_img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficient correlation implementation')
    args = parser.parse_args()

    model = RAFT(args)
    model.load_state_dict(torch.load(args.model))
    model = model.to(DEVICE)
    model.eval()

    cap1 = setup_camera(0)
    cap2 = setup_camera(1)
    left_mat_path = "./calibration/left_rectification.xml"
    right_mat_path = "./calibration/right_rectification.xml"
    x_mat_left, y_mat_left = read_matrices(left_mat_path)
    x_mat_right, y_mat_right = read_matrices(right_mat_path)

    if cap1 is None or cap2 is None or x_mat_left is None or y_mat_right is None:
        print("Error setting up cameras or reading calibration matrices")
        return

    frame_queue1 = queue.Queue(maxsize=2)
    frame_queue2 = queue.Queue(maxsize=2)

    threading.Thread(target=frame_reader, args=(cap1, frame_queue1, x_mat_left, y_mat_left, True), daemon=True).start()
    threading.Thread(target=frame_reader, args=(cap2, frame_queue2, x_mat_right, y_mat_right, False), daemon=True).start()

    threading.Thread(target=optical_flow_processing, args=(model, frame_queue1), daemon=True).start()
    threading.Thread(target=optical_flow_processing, args=(model, frame_queue2), daemon=True).start()

    print("Press 'q' to quit.")
    while True:
        if cv2.waitKey(1) == ord('q'):
            break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
