import os
import time

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms
from sklearn.ensemble import IsolationForest
import numpy as np
import pprint

from .lib.config import update_config
from .lib.models import get_pose_net
from .lib.core.group import HeatmapParser
from .lib.utils.inference import get_outputs
from .lib.utils.transforms import get_final_preds
from .lib.utils.transforms import resize
from .lib.utils.DetectionRegresser import DetectionRegresser


def xyxy2rowh(x1, y1, x2, y2):
    cx = (x2+x1)//2
    cy = (y2+y1)//2
    w = abs(x2-x1)
    h = abs(y2-x1)
    return (cx, cy, w, h)


'''
def infer_efficient(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=(128, 128, 128), img_scale=1/256):
    height, width, _ = img.shape
    scale = net_input_height_size / height
    scaled_img = cv2.resize(img, (0, 0), fx=scale,
                            fy=scale, interpolation=cv2.INTER_CUBIC)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(
        scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)
    tensor_img = torch.from_numpy(padded_img).permute(
        2, 0, 1).unsqueeze(0).float()
    if not cpu:
        tensor_img = tensor_img.cuda()
    stages_output = net(tensor_img)
    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(
        stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio,
                          fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)
    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio,
                      fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)
    return heatmaps, pafs, scale, pad
'''


class EfficientHRNetKeypoints:
    def __init__(self, pose_cfg):  # , defaults_dict):
        self.colors = []
        self.pose_cfg = pose_cfg
        self.net = get_pose_net(pose_cfg, is_train=False)
        self.net.load_state_dict(torch.load(
            self.pose_cfg.TEST.MODEL_FILE, map_location='cpu'), strict=True)
        self.net.eval()
        self.eval_set = 2
        # self.net.cuda()
        self.d = dict()
        # FIXME: Add the DetectionRegresser configuration to YAML file
        # , defaults_dict['window_length'], defaults_dict['iou_threshold'],
        self.dr = DetectionRegresser(0)
        # defaults_dict['forgetfulness'], defaults_dict['regress_type'])
        self.parser = HeatmapParser(self.pose_cfg)
        cudnn.benchmark = self.pose_cfg.CUDNN.BENCHMARK
        torch.backends.cudnn.deterministic = self.pose_cfg.CUDNN.DETERMINISTIC
        torch.backends.cudnn.enabled = self.pose_cfg.CUDNN.ENABLED

        self.transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )  # ,
                # torchvision.transforms.Resize(cfg.DATASET.INPUT_SIZE)
            ]
        )

    def runPoseEstimation(self, image):
        # final_heatmaps = None
        # tags_list = []

        image_resized, center, scale = resize(
            image, self.pose_cfg.DATASET.INPUT_SIZE
        )
        image_resized = self.transforms(image_resized)
        image_resized = image_resized.unsqueeze(0)  # .cuda()

        t = time.time()
        outputs, heatmaps, tags = get_outputs(
            self.pose_cfg, self.net, image_resized, with_flip=False,
            project2image=self.pose_cfg.TEST.PROJECT2IMAGE
        )
        t = time.time() - t
        # print(f'{t:.5f}')
        grouped, scores = self.parser.parse(
            heatmaps, tags, self.pose_cfg.TEST.ADJUST, self.pose_cfg.TEST.REFINE
        )
        final_results = get_final_preds(
            grouped, center, scale,
            [heatmaps.size(3), heatmaps.size(2)]
        )
        # filters redundant poses

        final_pts = []
        for i in range(len(final_results)):
            final_pts.insert(i, list())
            for pts in final_results[i]:
                if len(final_pts[i]) > 0:
                    diff = np.mean(
                        np.abs(np.array(final_pts[i])[..., :2] - pts[..., :2]))
                    if np.any(diff < 3):
                        final_pts[i].append([-1, -1, pts[2], pts[3]])
                        continue
                final_pts[i].append(pts)
        final_results = final_pts
        for idx in range(len(final_results)):
            final_results[idx] = np.concatenate(final_results[idx], axis=0)
            final_results[idx] = np.reshape(final_results[idx], (-1, 4))
        # print("Final results after filter", len(final_results))

        keypoints = []
        h_scores = []
        x_coordinates = []
        y_coordinates = []
        for idx in range(len(final_results)):
            key_temp = []
            x_temp = []
            y_temp = []
            h_temp = []
            for i in range(len(final_results[idx])):
                keypoint = final_results[idx][i, :2]
                key_temp.append(keypoint)
                x_coor = final_results[idx][i, 0]
                x_temp.append(x_coor)
                y_coor = final_results[idx][i, 1]
                y_temp.append(y_coor)
                h_score = final_results[idx][i, 2]
                h_temp.append(h_score)

            keypoints.append(key_temp)
            x_coordinates.append(x_temp)
            y_coordinates.append(y_temp)
            h_scores.append(h_temp)
            keypoints[idx] = np.concatenate(keypoints[idx], axis=0)
            keypoints[idx] = np.reshape(keypoints[idx], (-1, 2))
        return keypoints, x_coordinates, y_coordinates, h_scores

    # this class should be here to pass data directly
    def getEfficientHRNetPoseData(self, image, outlier_thresh=-0.1, heatmap_th=0.075):
        # Calculate poses and regress bounding boxes from keypoints
        valid_keypoint_perc = []
        keypoints, x_coordinates, y_coordinates, h_scores = self.runPoseEstimation(
            image)
        bboxes = []
        final_keypoints = []

        # Iterate through each predicted person
        # TODO: if this code will be persisting for a while we may want to swtich
        #   to an enumerate(keypoints) for-each approach in the interest of legibility
        for idx in range(len(keypoints)):
            # Remove Outliers
            # Reshape this person's keypoint coordinates to n*2 array
            temp_xy = list(zip(x_coordinates[idx], y_coordinates[idx]))

            # Use scikit's IsolationForest to score how well each keypoint fits in this group
            clf = IsolationForest(random_state=0).fit(temp_xy)
            clf_scores = clf.decision_function(temp_xy)

            # Apply thresholding to remove outliers
            for clf_idx in range(len(clf_scores)):
                if clf_scores[clf_idx] < outlier_thresh:
                    keypoints[idx][clf_idx][0] = -1
                    keypoints[idx][clf_idx][1] = -1
            # End Remove Outliers

            num_found_keypoints = 0

            # Keep track of valid x,y to find bbox bounds
            x_ax = []
            y_ax = []

            # Iterate through this person's keypoints
            for i in range(len(keypoints[idx])):
                # Ignore thresholded outlier keypoints
                if (keypoints[idx][i][0] and keypoints[idx][i][1]) != -1:
                    # Ignore thresholded heatmap keypoints (?)
                    if h_scores[idx][i] >= heatmap_th:
                        num_found_keypoints += 1
                        x_coor = x_coordinates[idx][i]
                        x_ax.append(x_coor)
                        y_coor = y_coordinates[idx][i]
                        y_ax.append(y_coor)

            # Create a bbox only if valid keypoints have been found
            if len(x_ax) > 0 and len(y_ax) > 0:
                x_min = np.amin(x_ax)
                y_min = np.amin(y_ax)
                width = np.amax(x_ax) - x_min
                height = np.amax(y_ax) - y_min
                if (height >= width):
                    bboxes.append([x_min, y_min, width, height])
                    valid_keypoint_perc.append(
                        num_found_keypoints / len(keypoints[idx]))
                    final_keypoints.append(keypoints[idx])

        # Run linear regression
        if final_keypoints is not None and bboxes is not None:
            self.dr.update(bboxes, final_keypoints, valid_keypoint_perc)
            try:
                bboxes, final_keypoints, valid_keypoint_perc = self.dr.predict()
            except TypeError:
                # this is only an issue if the DR has dropped detections (aka there were some, now none)
                if len(bboxes) > 0:
                    print('DetectionRegresser.predict() returned None!')

        bboxes = np.array(bboxes)
        final_keypoints = np.array(final_keypoints)
        valid_keypoint_perc = np.array(valid_keypoint_perc)
        return bboxes, final_keypoints, valid_keypoint_perc

    def draw_bboxes_and_keypoints(self, keypoints, bboxes, image):
        image_crops_list = []
        self.d.clear()
        bboxes_xywh = []
        for i in range(len(bboxes)):
            color = (np.random.randint(0, 255), np.random.randint(
                0, 255), np.random.randint(0, 255))
            x, y, w, h = bboxes[i]
            x, y, w, h = int(x), int(y), int(w), int(h)  # rectangle
            x1 = x
            y1 = y
            x2, y2 = x1 + w, y1 + h
            x1, y1, x2, y2 = x1-10, y1-10, x2+10, y2+10
            check_keypoints = True
            for j in keypoints[i]:
                x, y = j
                x, y = int(round(x, 0)), int(round(y, 0))
                if x < 0 or y < 0:
                    continue
                if not(x1 <= x <= x2 and y1 <= y <= y2):
                    # pass
                    # check_keypoints = False  # if you disable this then check for bb won't be performed
                    print('failed for BB :{} ,({},{})<=({},{})<=({},{})'.format(
                        i, x1, y1, x, y, x2, y2))

            if check_keypoints:
                for j in keypoints[i]:
                    x, y = j
                    x, y = int(round(x, 0)), int(round(y, 0))
                    image = cv2.circle(image, (x, y), 2, color, 2)
                x1, y1, x2, y2 = x1+10, y1+10, x2-10, y2-10
                image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                image_crops_list.append(image[y:y + h, x:x + w])
                bboxes_xywh.append(xyxy2rowh(x1, y1, x2, y2))
            else:
                self.d[i] = 'DELETE'

        return image, image_crops_list, np.asarray(bboxes_xywh)

    def draw_bboxes(self, bboxes, image):
        image_crops_list = []
        i = 0
        for box in bboxes:
            x, y, w, h = box
            x, y, w, h = int(x), int(y), int(w), int(h)
            image = cv2.rectangle(
                image, (x, y), (x + w, y + h), self.colors[i], 2)
            image_crops_list.append(image[y:y + h, x:x + w])
            i += 1
        return image, image_crops_list

    def draw_keypoints(self, keypoints, image):
        for i in keypoints:
            color = (np.random.randint(0, 255), np.random.randint(
                0, 255), np.random.randint(0, 255))
            self.colors.append(color)
            for j in i:
                x, y = j
                x, y = int(x), int(y)
                image = cv2.circle(image, (x, y), 2, color, 2)
        return image

    # Draw colored pose skeleton on the image frame
    def drawFramePoses(self, frame, image, ObjectHistoryList):

        for objhist in ObjectHistoryList:
            if (self.eval_set % 2 == 0):
                image = self.drawPoseCOCO18(image, objhist)

        return image

    # Draw one skeleton on the image frame
    def drawPoseCOCO18(self, image, objhist):
        kp_pairs = [[14, 16],
                    [13, 15],
                    [12, 14],
                    [11, 13],
                    [6, 8],
                    [8, 10],
                    [5, 7],
                    [7, 9],
                    [1, 3],
                    [2, 4],
                    [3, 5],
                    [4, 6],
                    [0, 1],
                    [0, 2],
                    [5, 11],
                    [6, 12],
                    [5, 6],
                    [11, 12]]

        keypoints = objhist.keypoints
        personID = objhist.sendObject

        if (objhist.reIDFlag == 1):
            coloridx = int(personID.label % 9)
            colorlist = bbox_colors[coloridx]
        else:
            colorlist = grey_box

        bbox = xywh2xyxy(personID.bbox)
        start_point = (bx, bbox[1])
        color = (colorlist[0], colorlist[1], colorlist[2])
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.7
        thickness = 2
        # id_label = str(int(personID.label / 1000000)) + "-{:01d}".format(int(personID.label % 1000000))
        # id_label = '{:d}'.format(int(personID.label / 1000000))
        # image = cv2.putText(image, id_label, start_point, font, scale, color, thickness)

        for pair in kp_pairs:
            # print(keypoints.shape)
            kp0 = keypoints[pair[0], :]
            kp1 = keypoints[pair[1], :]
            if (kp0[0] != -1) and (kp1[0] != -1):
                image = cv2.line(image, (int(kp0[0]), int(kp0[1])), (int(
                    kp1[0]), int(kp1[1])), color, thickness)
            if (kp0[0] != -1):
                image = cv2.circle(image, (int(kp0[0]), int(kp0[1])), 3, color)
            if (kp1[0] != -1):
                image = cv2.circle(image, (int(kp1[0]), int(kp1[1])), 3, color)

        return image

    def process(self, image):  # , image_output_path, frame):

        # 1) Getting Keypoints From Efficient HR Net
        bboxes, final_keypoints, valid_keypoints = self.getEfficientHRNetPoseData(
            image)

        # print(len(bboxes),len(final_keypoints))
        # 2) Drawing bounding boxes and getting image crops list
        # image, image_crops_list = self.draw_bboxes(bboxes, image)
        # 3) Drawing Keypoints on image
        # image = self.draw_keypoints(final_keypoints, image)
        image, image_crops_list, bboxes_based_kp = self.draw_bboxes_and_keypoints(
            final_keypoints, bboxes, image)
        # image = self.drawFramePoses(final_keypoints,bboxes,image)
        # 4) Saving output to folder
        # cv2.imwrite(image_output_path['EfficentHRNet'] + '/' + "{0:4d}".format(int(frame)) + '.jpg', image)

        for i in sorted(self.d.keys(), reverse=True):
            # print('index :{}',i)
            final_keypoints = np.delete(final_keypoints, i, axis=0)
            valid_keypoints = np.delete(valid_keypoints, i)
            bboxes_based_kp = np.delete(bboxes_based_kp, i, axis=0)
            # print(bboxes)

        # keypoint 2 bboxes are giving me boxes with zero width and height. Don't know why though!! ;(
        new_bboxes = []
        new_final_keypoints = []
        new_valid_keypoints = []
        for box, kpt, valid_kpt in zip(bboxes_based_kp, final_keypoints, valid_keypoints):
            if (box[2] != 0 and box[3] != 0):
                new_bboxes.append(box)
                new_valid_keypoints.append(valid_kpt)
                new_final_keypoints.append(kpt)
        new_bboxes = np.asarray(new_bboxes)
        new_final_keypoints = np.asarray(new_final_keypoints)
        new_valid_keypoints = np.asarray(new_valid_keypoints)

        return new_bboxes, new_final_keypoints, new_valid_keypoints, image, image_crops_list

    def detect(self, image):  # , image_output_path):
        if image is not None:
            # print("Efficient HRNet :- Processing Frame : {0:4d}".format(int(frame)))
            bboxes, final_keypoints, valid_keypoints, image, image_crops_list = self.process(
                image)  # , image_output_path, frame)
            # print("Efficient HRNet :- Completed Processing Frame : {0:4d}".format(int(frame)))
            return bboxes, final_keypoints, valid_keypoints, image, image_crops_list
        else:
            # print("No Image Received " + image_path + '/' + "{0:4d}".format(int(frame)) + '.jpg')
            return None, None, None, None, None


def build_eff_hrnnet(cfg, args, logger):
    update_config(cfg, args)

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    return EfficientHRNetKeypoints(cfg)
