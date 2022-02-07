__copyright__ = """
    SLAMcore Limited
    All Rights Reserved.
    (C) Copyright 2022

    NOTICE:

    All information contained herein is, and remains the property of SLAMcore
    Limited and its suppliers, if any. The intellectual and technical concepts
    contained herein are proprietary to SLAMcore Limited and its suppliers and
    may be covered by patents in process, and are protected by trade secret or
    copyright law.
"""

__license__ = "CC BY-NC-SA 3.0"

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from scipy.spatial.transform import Rotation as R

from acdc.engine.factory import DatasetSplit
from acdc.loaders.datasets.dataset import Dataset
from acdc.utils.common import *


class TartanAirSequence:
    """Class to represent a sequence"""

    def __init__(
        self,
        timestamp: ArrayF32,
        cam0_path: ArrayStr,
        T_WB: ArrayF32,
        idx: ArrayInt,
        folder_name: ArrayStr,
    ):
        """Constructor

        :param timestamp: List of timestamps
        :param cam0_path: List of paths for Cam0 image
        :param T_WB: List of poses
        :param idx: Indexes of images to use (laser_on=1)
        :param folder_name: List of folders
        """
        self.timestamp: ArrayF32 = timestamp
        self.cam0_path: ArrayStr = cam0_path
        self.T_WB: ArrayF32 = T_WB
        self.idx: ArrayInt = idx
        self.n: int = len(timestamp)
        self.folder_name: ArrayStr = folder_name


class Tartanair(Dataset):
    def __init__(
        self,
        dataset_split: DatasetSplit,
        data_path: str,
        calibration_file: str,
        left_crop: int = 128,
        dist_threshold_min: float = 0.1,
        dist_threshold_max: float = 1,
        angle_threshold: float = 5,
        flexible_thresholds: bool = True,
        teacher_prediction_directory: Optional[str] = None,
    ):
        """Constructor

        :param dataset_split: Split of dataset to use
        :param data_path: Path to dataset
        :param calibration_file: Calibration file path
        :param dist_threshold_min: Minimum distance of previous and next frames (meters), defaults to 0.1
        :param dist_threshold_max: Maximum distance of previous and next frames (meters), defaults to 1
        :param angle_threshold: Minimum angle of previous and next frames (degrees), defaults to 5
        :param flexible_thresholds: Whether to use flexible thresholds, defaults to True
        :param teacher_prediction_directory: Path to teacher depths, defaults to None
        """
        super(Tartanair, self).__init__()

        self.left_crop = left_crop
        dataset_split_str = str(dataset_split)
        # test is equal to val
        if dataset_split_str == "test":
            dataset_split_str = "val"

        data_path = os.path.join(data_path, dataset_split_str)

        self.teacher_prediction_directory = teacher_prediction_directory

        self.dist_threshold_min = dist_threshold_min
        self.dist_threshold_max = dist_threshold_max
        self.angle_threshold = angle_threshold / 180.0 * np.pi  # radians
        self.flexible_thresholds = flexible_thresholds

        # load calibration files
        if not os.path.isfile(calibration_file):
            raise RuntimeError(f"Calibration file not found: {calibration_file}")

        with open(calibration_file) as file:
            calib = yaml.load(file, Loader=yaml.FullLoader)

        T_BS = np.array(calib["imu_params"]["T_BS"]).reshape(4, 4).astype(dtype=np.float32)
        self.default_calib_dict = {"imu_params": {"T_BS": T_BS}}

        cam_dict = {("Visible", 0): "cam0", ("Visible", 1): "cam1", ("Depth", 0): "dep"}
        for calib_camera in calib["cameras"]:
            sensor_id = tuple(calib_camera["SensorID"])
            if sensor_id in cam_dict:
                K, T, dis_coeff = self._get_intrinsic_extrinsic(calib_camera)
                cam_key = cam_dict[sensor_id]  # type: ignore
                self.default_calib_dict[cam_key] = {
                    "K": K,
                    "T": T,
                    "dis_coeff": dis_coeff,
                }

        assert set(list(cam_dict.values()) + ["imu_params"]) == set(
            self.default_calib_dict.keys()
        )

        # load data
        root_paths = []
        if dataset_split == DatasetSplit.TRAIN:
            for env in ["office", "office2", "carwelding"]:
                for lvl in ["Easy", "Hard"]:
                    root_paths += [
                        os.path.join(data_path, env, lvl, f)
                        for f in os.listdir(os.path.join(data_path, env, lvl))
                    ]
        elif dataset_split == DatasetSplit.VAL or dataset_split == DatasetSplit.TEST:
            for env in ["hospital", "japanesealley"]:
                for lvl in ["Easy"]:
                    root_paths += [
                        os.path.join(data_path, env, lvl, f)
                        for f in os.listdir(os.path.join(data_path, env, lvl))
                    ]
        else:
            NotImplementedError("Unknown dataset split")

        # load data
        self.data, self.index = self._load_data(
            root_paths, dataset_split != DatasetSplit.TRAIN
        )

    def __len__(self) -> int:
        """Number of elements

        :return: Number of elements
        """
        return len(self.index)

    def _get_intrinsic_extrinsic(
        self,
        data: Dict[str, ArrayF32],
    ) -> Tuple[ArrayF32, ArrayF32, ArrayF32,]:
        """Get instrinsic and extrinsic parameters

        :param data: Dictionary with camera parameters
        :return: Camera parameters
        """
        T = np.asarray(data["T_SC"]).reshape(4, 4).astype(dtype=np.float32)

        dis_coeff = data.get(
            "distortion_coefficients", np.zeros(4)
        )  # for position_v2 we do not have any distortion parameters.
        F = data["focal_length"]
        pp = data["principal_point"]

        pp[0] = pp[0] - self.left_crop

        K = np.asarray([[F[0], 0, pp[0]], [0, F[1], pp[1]], [0, 0, 1]]).astype(
            dtype=np.float32
        )

        dis_coeff = np.asarray(dis_coeff).reshape(4, 1)

        return K, T, dis_coeff

    def _get_frames(self, frame_idx: int, sequence: TartanAirSequence) -> Tuple[int, int]:
        """Get previous and next frames indexes

        :param frame_idx: Current frame index
        :param sequence: Sequence of the current frame
        :return: Previous and next frames
        """
        T = sequence.T_WB[frame_idx, ...]
        R0 = T[:3, :3]
        t0 = T[:3, 3]

        # Get transformations from the same sequences as index
        frameidx_from_seq = np.arange(sequence.T_WB.shape[0])
        T = sequence.T_WB
        R1 = T[:, :3, :3]
        t1 = T[:, :3, 3]

        # calculate translation distance
        dist = np.sum((t0 - t1) ** 2, axis=1) ** 0.5

        # calculate rotation distance
        ang = (R.from_matrix(R0) * R.from_matrix(R1).inv()).magnitude()

        dist_threshold_min = self.dist_threshold_min
        dist_threshold_max = self.dist_threshold_max
        angle_threshold = self.angle_threshold
        keep_trying = True
        num_trials = 0
        while keep_trying:
            c1 = dist > dist_threshold_min
            c2 = dist < dist_threshold_max
            c3 = ang < angle_threshold

            potential = frameidx_from_seq[c1 & c2 & c3]
            potential_prev = potential[potential < frame_idx]
            potential_next = potential[potential > frame_idx]

            if (
                (len(potential_prev) == 0 or len(potential_next) == 0)
                and num_trials < 10
                and self.flexible_thresholds
            ):
                dist_threshold_min = dist_threshold_min / 2
                dist_threshold_max = dist_threshold_max * 2
                angle_threshold = angle_threshold * 2
                num_trials += 1
            else:
                keep_trying = False

        # take the closest frame that is far enough away. If it doesn't exist, just take the prev/next frame
        prev_idx = max(potential_prev) if len(potential_prev) > 0 else None
        next_idx = min(potential_next) if len(potential_next) > 0 else None

        # check if lost
        if prev_idx is None:
            potential = frameidx_from_seq
            potential_prev = potential[potential < frame_idx]
            prev_idx = max(potential_prev) if len(potential_prev) > 0 else frame_idx - 1

        if next_idx is None:
            potential = frameidx_from_seq
            potential_next = potential[potential > frame_idx]
            next_idx = min(potential_next) if len(potential_next) > 0 else frame_idx + 1

        return prev_idx, next_idx

    def get_filename(self, index: int) -> str:
        """Get filename

        :param index: Index of the image
        :param projector_on: Whether to load image with projector on
        :return: Filename
        """
        seq_idx, frame_idx = self.index[index]
        sequence = self.data[seq_idx]

        return sequence.folder_name[frame_idx][0]

    def get_ir0(
        self, index: Union[int, Tuple[int, int]], projector_on: bool = False
    ) -> ArrayUI8:
        """Get IR0 image

        :param index: Index of the image
        :param projector_on: Whether to load image with projector on
        :return: Image
        """
        if isinstance(index, int):
            seq_idx, frame_idx = self.index[index]
        else:
            seq_idx, frame_idx = index
        sequence = self.data[seq_idx]

        return self._load_im(sequence.cam0_path[frame_idx], projector_on)

    def get_ir1(
        self, index: Union[int, Tuple[int, int]], projector_on: bool = False
    ) -> ArrayUI8:
        """Get IR1 image

        :param index: Index of the image
        :param projector_on: Whether to load image with projector on
        :return: Image
        """
        if isinstance(index, int):
            seq_idx, frame_idx = self.index[index]
        else:
            seq_idx, frame_idx = index
        sequence = self.data[seq_idx]

        return self._load_im(
            sequence.cam0_path[frame_idx].replace("cam0", "cam1"), projector_on
        )

    def get_depth(self, index: int) -> Tuple[ArrayF32, ArrayF32]:
        """Get depth image

        :param index: Index of the image
        :return: Depth image and valid mask
        """
        seq_idx, frame_idx = self.index[index]
        sequence = self.data[seq_idx]

        dep, valid_mask = self._load_depth(
            sequence.cam0_path[frame_idx].replace("cam0", "depth0_on")
        )
        return dep, valid_mask

    def get_depth_gt(self, index: int) -> Tuple[Optional[ArrayF32], Optional[ArrayF32]]:
        """Get ground truth depth image

        :param index: Index of the image
        :return: Ground truth depth and valid mask
        """
        seq_idx, frame_idx = self.index[index]
        sequence = self.data[seq_idx]

        filename = sequence.cam0_path[frame_idx].replace("cam0", "depth0")
        dep, valid_mask = self._load_depth(filename)

        return dep, valid_mask

    def get_current_sparse(self, index: int) -> Optional[ArrayF32]:
        """Get current sparse points

        :param index: Index of the image
        :return: List of points (x, y, z) where x and y are image coordinates
        """
        seq_idx, frame_idx = self.index[index]
        sequence = self.data[seq_idx]

        path = (
            sequence.cam0_path[frame_idx]
            .replace("cam0", "seen_features")
            .replace("png", "csv")
        )
        return self._load_sparse(path)

    def get_best_sparse(self, index: int) -> Optional[ArrayF32]:
        """Get best sparse points

        :param index: Index of the image
        :return: List of points (x, y, z) where x and y are image coordinates
        """
        seq_idx, frame_idx = self.index[index]
        sequence = self.data[seq_idx]

        path = (
            sequence.cam0_path[frame_idx]
            .replace("cam0", "best_features")
            .replace(".png", ".csv")
        )
        return self._load_sparse(path)

    def get_neighbor_frames(self, index: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Get neighbor frames

        :param index: Current frame index
        :raises ValueError: If there's an internal problem with the dataset
        :return: Indexes of neighbor frames
        """
        seq_idx, frame_idx = self.index[index]
        sequence = self.data[seq_idx]

        # get previous and next frames
        prev_idx, next_idx = self._get_frames(frame_idx, sequence)

        final_prev_idx = (seq_idx, prev_idx)
        final_next_idx = (seq_idx, next_idx)

        return final_prev_idx, final_next_idx

    def get_pose_ir0(self, index: Union[int, Tuple[int, int]]) -> ArrayF32:
        """Get pose of IR0

        :param index: Current frame index
        :return: Pose
        """
        if isinstance(index, int):
            seq_idx, frame_idx = self.index[index]
        else:
            seq_idx, frame_idx = index
        sequence = self.data[seq_idx]

        # body to sensor
        T_BS = self.default_calib_dict["imu_params"]["T_BS"]
        # world to body
        T_WB = sequence.T_WB[frame_idx]

        # camera to world (T_WC = T_WB @ T_BS @ T_SC)
        T_ir0 = self.default_calib_dict["cam0"]["T"]
        return np.matmul(np.matmul(T_WB, T_BS), T_ir0).astype(dtype=np.float32)

    def get_pose_ir1(self, index: Union[int, Tuple[int, int]]) -> ArrayF32:
        """Get pose of IR1

        :param index: Current frame index
        :return: Pose
        """
        if isinstance(index, int):
            seq_idx, frame_idx = self.index[index]
        else:
            seq_idx, frame_idx = index
        sequence = self.data[seq_idx]

        # body to sensor
        T_BS = self.default_calib_dict["imu_params"]["T_BS"]
        # world to body
        T_WB = sequence.T_WB[frame_idx]

        # camera to world (T_WC = T_WB @ T_BS @ T_SC)
        T_ir1 = self.default_calib_dict["cam1"]["T"]
        return np.matmul(np.matmul(T_WB, T_BS), T_ir1).astype(dtype=np.float32)

    def get_pose_depth(self, index: int) -> ArrayF32:
        """Get pose of depth

        :param index: Current frame index
        :return: Pose
        """
        seq_idx, frame_idx = self.index[index]
        sequence = self.data[seq_idx]

        T_BS = self.default_calib_dict["imu_params"]["T_BS"]
        T_WB = sequence.T_WB[frame_idx]

        # camera to world (T_WC = T_WB @ T_BS @ T_SC)
        T_dep = self.default_calib_dict["dep"]["T"]
        return np.matmul(np.matmul(T_WB, T_BS), T_dep).astype(dtype=np.float32)

    def get_K_ir0(self, index: int) -> ArrayF32:
        """Get IR0 K matrix

        :param index: Current frame index
        :return: K matrix
        """
        return self.default_calib_dict["cam0"]["K"]

    def get_K_ir1(self, index: int) -> ArrayF32:
        """Get IR1 K matrix

        :param index: Current frame index
        :return: K matrix
        """
        return self.default_calib_dict["cam1"]["K"]

    def get_K_depth(self, index: int) -> ArrayF32:
        """Get depth K matrix

        :param index: Current frame index
        :return: K matrix
        """
        return self.default_calib_dict["dep"]["K"]

    def _load_seq(self, datapath: str, is_val: bool) -> TartanAirSequence:
        """Loads a sequence

        :param datapath: Path of the sequence
        :param is_val: Whether this is a validation sequence
        :return: Sequence information
        """
        data = {}
        for sensor in ["cam0", "pose0"]:
            data_csv_file = os.path.join(datapath, sensor, "data.csv")
            if not os.path.isfile(data_csv_file):
                raise FileNotFoundError("File does not exist: {}".format(data_csv_file))
            data[sensor] = pd.read_csv(data_csv_file).values

        image_data = data["cam0"]
        image_data[:, 1] = [
            os.path.join(datapath, "cam0", "data", im) for im in image_data[:, 1]
        ]
        pose_data = data["pose0"]
        # both timestamp must match
        assert image_data.shape[0] == pose_data.shape[0]
        assert np.abs(image_data[:, 0] - pose_data[:, 0]).sum() == 0

        timestamp = image_data[:, 0]

        translation = pose_data[:, 1:4]
        quat = pose_data[:, 4:]  # (x, y, z, w)
        rot = R.from_quat(quat)

        # initialize T
        T = np.diag(np.ones(4))
        T = np.repeat(T[None, :, :], len(translation), axis=0)

        # insert into T (here we have some padding to get the same length of the images)
        # This makes indexing in getitem significantly easier
        T[:, :3, :3] = rot.as_matrix()
        T[:, :3, 3] = translation

        if is_val:
            val_data = pd.read_csv(os.path.join(datapath, "cam0", "val_data.csv")).values
            idx = val_data[:, 0]
        else:
            # Skip the first and last image
            idx = np.arange(1, image_data.shape[0] - 1)

        # append a attribute saying which sequence a frame is from

        path_obj = Path(datapath)
        folder_name = [
            str(path_obj.parent.parent.name) + "_" + str(path_obj.name) + "_" + str(ts)
            for ts in image_data[:, 0]
        ]
        folder_name = np.asarray(folder_name).reshape(-1, 1)

        sequence = TartanAirSequence(
            timestamp,
            image_data[:, 1],
            T,
            idx,
            folder_name,
        )

        return sequence

    def _load_data(
        self, root_paths: List[str], is_val: bool
    ) -> Tuple[List[TartanAirSequence], List[Tuple[int, int]]]:
        """Loads all sequences

        :param datapath: Path of the data
        :param is_val: Whether this is a validation set
        :return: List of sequences and valid indexes
        """
        data = []
        index = []

        for i, seq in enumerate(root_paths):
            seq_data = self._load_seq(seq, is_val)

            for j in seq_data.idx:
                # Save sequence id and image id
                index.append((i, j))

            data.append(seq_data)

        return data, index

    def _load_im(self, path: str, projector_on: bool) -> ArrayUI8:
        """Load an image

        :param path: Path
        :param projector_on: Whether to load image with projector on
        :return: Image
        """
        if projector_on:
            path = path.replace("cam1", "cam1_on").replace("cam0", "cam0_on")

        im = read_image(path)
        assert im.dtype == np.uint8

        im_pil = array_to_pilimage(im)

        # crop left side of image
        width, height = im_pil.size
        im_pil = im_pil.crop((self.left_crop, 0, width, height))

        return pilimage_to_array(im_pil)

    def _load_sparse(self, path: str) -> Optional[ArrayF32]:
        """Load sparse file

        :param path: Path to sparse file
        :return: List of points (x, y, z) where x and y are image coordinates
        """
        no_exist = not os.path.isfile(path)
        # empty file
        if no_exist or os.stat(path).st_size == 0:
            return None

        # x, y, z, id
        data = np.loadtxt(path).reshape(-1, 4)

        x = data[:, 0].astype(int)
        not_cropped = x >= self.left_crop

        points = data[not_cropped, :3].astype(np.float32)
        return points

    def _load_depth(self, path: str) -> Tuple[ArrayF32, ArrayF32]:
        """Load depth image

        :param path: Path to depth image
        :return: Depth image and valid mask Image
        """
        dep = read_image(path)
        assert dep.dtype == np.uint16

        dep_pil = array_to_pilimage(dep)

        # crop left side of image
        width, height = dep_pil.size
        dep_pil = dep_pil.crop((self.left_crop, 0, width, height))

        dep = pilimage_to_array(dep_pil)
        dep = dep.astype(np.float32) / (1000.0)
        # dep stored in milimeters so dividing with 1000 gives meters

        valid_mask = (dep > 0).astype(np.float32)
        valid_mask = cast(ArrayF32, valid_mask)

        return dep, valid_mask
