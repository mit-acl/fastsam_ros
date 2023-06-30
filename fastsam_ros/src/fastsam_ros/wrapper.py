import os, sys
import random
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import torch

from ultralytics import YOLO

# add FastSAM submodule to path
FILE_ABS_DIR = pathlib.Path(__file__).absolute().parent
FASTSAM_ROOT = (FILE_ABS_DIR / 'FastSAM').as_posix()
if FASTSAM_ROOT not in sys.path:
    sys.path.append(FASTSAM_ROOT)

from utils.tools import fast_show_mask, fast_show_mask_gpu


class FastSAM:
    def __init__(self, weights, conf_thresh: float = 0.4, iou_thresh: float = 0.9,
                 img_size: int = 1024, retina: bool = True, device: str = "cuda"):
        self.__img_size = img_size
        self.__conf_thresh = conf_thresh
        self.__iou_thresh = iou_thresh
        self.__retina = retina
        self.__weights = weights
        self.__device = device

        self.model = YOLO(self.__weights)

    @torch.no_grad()
    def _inference(self, img: torch.Tensor):
        """
        :param img: tensor [c, h, w]
        :returns: tensor of shape [num_boxes, 6], where each item is represented as
            [x1, y1, x2, y2, confidence, class_id]
        """
        pred_results = self.model(img)[0]
        detections = non_max_suppression(pred_results, conf_thres=self.__conf_thresh, iou_thres=self.__iou_thresh)

        if detections:
            detections = detections[0]

        return detections


    def detect(self, img0, max_det=100):
        """
        Perform inference on an image to detect classes.
        
        Parameters
        ----------
        img0 : (h, w, c) np.array -- the input image

        Returns
        -------
        dets : (n, 6) np.array -- n detections
                Each detection is 2d bbox xyxy, confidence, class
        """

        results = self.model(
            img0,
            imgsz=self.__img_size,
            device=self.__device,
            retina_masks=self.__retina,
            iou=self.__iou_thresh,
            conf=self.__conf_thresh,
            max_det=max_det,
        )

        self.fast_process(img0, annotations=results[0].masks.data, mask_random_color=True)


    def fast_process(self, img0, annotations, save_path='output', mask_random_color=False, retina=True, bbox=None, points=None, edges=False, better_quality=False, with_contours=False):
        if isinstance(annotations[0], dict):
            annotations = [annotation["segmentation"] for annotation in annotations]
        image = cv.cvtColor(img0, cv.COLOR_BGR2RGB)
        original_h = image.shape[0]
        original_w = image.shape[1]
        plt.figure(figsize=(original_w/100, original_h/100))
        plt.imshow(image)
        if better_quality == True:
            if isinstance(annotations[0], torch.Tensor):
                annotations = np.array(annotations.cpu())
            for i, mask in enumerate(annotations):
                mask = cv.morphologyEx(
                    mask.astype(np.uint8), cv.MORPH_CLOSE, np.ones((3, 3), np.uint8)
                )
                annotations[i] = cv.morphologyEx(
                    mask.astype(np.uint8), cv.MORPH_OPEN, np.ones((8, 8), np.uint8)
                )
        # if args.device == "cpu":
            # annotations = np.array(annotations)
            # fast_show_mask(
            #     annotations,
            #     plt.gca(),
            #     random_color=mask_random_color,
            #     bbox=bbox,
            #     points=points,
            #     pointlabel=None,
            #     retinamask=retina,
            #     target_height=original_h,
            #     target_width=original_w,
            # )
        # else:
        if isinstance(annotations[0], np.ndarray):
            annotations = torch.from_numpy(annotations)
        fast_show_mask_gpu(
            annotations,
            plt.gca(),
            random_color=mask_random_color,
            bbox=bbox,
            points=points,
            pointlabel=None,
            retinamask=retina,
            target_height=original_h,
            target_width=original_w,
        )
        if isinstance(annotations, torch.Tensor):
            annotations = annotations.cpu().numpy()
        if with_contours == True:
            contour_all = []
            temp = np.zeros((original_h, original_w, 1))
            for i, mask in enumerate(annotations):
                if type(mask) == dict:
                    mask = mask["segmentation"]
                annotation = mask.astype(np.uint8)
                if retina == False:
                    annotation = cv.resize(
                        annotation,
                        (original_w, original_h),
                        interpolation=cv.INTER_NEAREST,
                    )
                contours, hierarchy = cv.findContours(
                    annotation, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
                )
                for contour in contours:
                    contour_all.append(contour)
            cv.drawContours(temp, contour_all, -1, (255, 255, 255), 2)
            color = np.array([0 / 255, 0 / 255, 255 / 255, 0.8])
            contour_mask = temp / 255 * color.reshape(1, 1, -1)
            plt.imshow(contour_mask)

        save_path = os.path.join(FILE_ABS_DIR, save_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.axis("off")
        fig = plt.gcf()
        # plt.draw()
        # plt.show()
        
        try:
            buf = fig.canvas.tostring_rgb()
        except AttributeError:
            fig.canvas.draw()
            buf = fig.canvas.tostring_rgb()
        
        cols, rows = fig.canvas.get_width_height()
        img_array = np.fromstring(buf, dtype=np.uint8).reshape(rows, cols, 3)
        cv.imwrite(os.path.join(save_path, 'out.jpg'), cv.cvtColor(img_array, cv.COLOR_RGB2BGR))



        # return dets.cpu().detach().numpy()
