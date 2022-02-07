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
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from acdc.engine.factory import DatasetSplit
from acdc.loaders.datasets.dataset import Dataset
from acdc.utils.common import *


class RealsenseSequence:
    """Class to represent a sequence"""

    def __init__(
        self,
        name: str,
        timestamp: ArrayF32,
        depth_path: ArrayStr,
        ir0_path: ArrayStr,
        ir1_path: ArrayStr,
        laser_power: ArrayInt,
        T_WB: Optional[ArrayF32],
        lost: Optional[ArrayInt],
        idx: ArrayInt,
        folder_name: ArrayStr,
    ):
        """Constructor

        :param name: Name of the sequence
        :param timestamp: List of timestamps
        :param depth_path: List of paths for depth image
        :param ir0_path: List of paths for IR0 image
        :param ir1_path: List of paths for IR1 image
        :param laser_power: List of laser power
        :param T_WB: List of poses
        :param lost: Whether each frame was lost
        :param idx: Indexes of images to use (laser_on=1)
        :param folder_name: List of folders
        """
        self.name: str = name
        self.timestamp: ArrayF32 = timestamp
        self.depth_path: ArrayStr = depth_path
        self.ir0_path: ArrayStr = ir0_path
        self.ir1_path: ArrayStr = ir1_path
        self.laser_power: ArrayInt = laser_power
        self.T_WB: Optional[ArrayF32] = T_WB
        self.lost: Optional[ArrayInt] = lost
        self.idx: ArrayInt = idx
        self.n: int = len(timestamp)
        self.folder_name: ArrayStr = folder_name


class Realsense(Dataset):
    def __init__(
        self,
        dataset_split: DatasetSplit,
        data_path: str,
        calibration_file: str,
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
        super(Realsense, self).__init__()

        self.root_path = data_path
        data_path = os.path.join(data_path, str(dataset_split))
        self.is_test = dataset_split == DatasetSplit.TEST
        self.is_val = dataset_split == DatasetSplit.VAL

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

        cam_dict = {("Infrared", 0): "ir0", ("Infrared", 1): "ir1", ("Depth", 0): "dep"}
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
        self.data, self.index = self._load_data(data_path)

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

        K = np.asarray([[F[0], 0, pp[0]], [0, F[1], pp[1]], [0, 0, 1]]).astype(
            dtype=np.float32
        )

        dis_coeff = np.asarray(dis_coeff).reshape(4, 1)

        return K, T, dis_coeff

    def _get_frames(self, frame_idx: int, sequence: RealsenseSequence) -> Tuple[int, int]:
        """Get previous and next frames indexes

        :param frame_idx: Current frame index
        :param sequence: Sequence of the current frame
        :return: Previous and next frames
        """
        if sequence.T_WB is None or sequence.lost is None:
            assert self.is_test
            return frame_idx, frame_idx
        else:
            T = sequence.T_WB[frame_idx, ...]
            R0 = T[:3, :3]
            t0 = T[:3, 3]

            # Get transformations from the same sequences as index
            frameidx_from_seq = np.arange(sequence.T_WB.shape[0])
            T = sequence.T_WB
            laser_on = sequence.laser_power
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
                c4 = laser_on == 0

                potential = frameidx_from_seq[c1 & c2 & c3 & c4]
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
            if prev_idx is None or np.sum(sequence.lost[prev_idx:frame_idx] == 0) > 0:
                potential = frameidx_from_seq[laser_on == 0]
                potential_prev = potential[potential < frame_idx]
                prev_idx = max(potential_prev) if len(potential_prev) > 0 else frame_idx - 1

            if next_idx is None or np.sum(sequence.lost[frame_idx:next_idx] == 0) > 0:
                potential = frameidx_from_seq[laser_on == 0]
                potential_next = potential[potential > frame_idx]
                next_idx = min(potential_next) if len(potential_next) > 0 else frame_idx + 1

            return prev_idx, next_idx

    def get_filename(self, index: int) -> str:
        """Get filename

        :param index: Index of the image
        :return: Filename
        """
        seq_idx, frame_idx = self.index[index]
        sequence = self.data[seq_idx]

        return sequence.folder_name[frame_idx][0]

    def get_ir0(
        self, index: Union[int, Tuple[int, int]], projector_on: bool = False
    ) -> ArrayUI8:
        """Get infrared image

        :param index: Index of the image
        :param projector_on: Whether to load image with projector on
        :return: Image
        """
        if isinstance(index, int):
            seq_idx, frame_idx = self.index[index]
        else:
            seq_idx, frame_idx = index
        sequence = self.data[seq_idx]

        if not self.is_test:
            im_projector_on = sequence.laser_power[frame_idx] != 0
            assert im_projector_on == projector_on

        return self._load_im(sequence.ir0_path[frame_idx])

    def get_ir1(
        self, index: Union[int, Tuple[int, int]], projector_on: bool = False
    ) -> ArrayUI8:
        """Get infrared right image

        :param index: Index of the image
        :param projector_on: Whether to load image with projector on
        :return: Image
        """
        if isinstance(index, int):
            seq_idx, frame_idx = self.index[index]
        else:
            seq_idx, frame_idx = index
        sequence = self.data[seq_idx]

        if not self.is_test:
            im_projector_on = sequence.laser_power[frame_idx] != 0
            assert im_projector_on == projector_on

        return self._load_im(sequence.ir1_path[frame_idx])

    def get_depth(self, index: int) -> Tuple[ArrayF32, ArrayF32]:
        """Get depth image

        :param index: Index of the image
        :return: Depth image and valid mask
        """
        seq_idx, frame_idx = self.index[index]
        sequence = self.data[seq_idx]

        dep, valid_mask = self._load_depth(sequence.depth_path[frame_idx])
        return dep, valid_mask

    def get_depth_gt(self, index: int) -> Tuple[Optional[ArrayF32], Optional[ArrayF32]]:
        """Get ground truth depth image

        :param index: Index of the image
        :return: Ground truth depth and valid mask
        """
        seq_idx, frame_idx = self.index[index]
        sequence = self.data[seq_idx]

        filename = os.path.basename(sequence.depth_path[frame_idx])
        path = os.path.join(self.root_path, "colmap_gt_flt", sequence.name, filename)
        dep = None
        valid_mask = None
        if os.path.exists(path):
            dep, valid_mask = self._load_depth(path)
        elif self.is_val:
            # no GT for validation set
            depth, _ = self.get_depth(index)
            dep = np.zeros_like(depth)
            valid_mask = np.zeros_like(depth)

        return dep, valid_mask

    def get_current_sparse(self, index: int) -> Optional[ArrayF32]:
        """Get current sparse points

        :param index: Index of the image
        :return: List of points (x, y, z) where x and y are image coordinates
        """
        seq_idx, frame_idx = self.index[index]
        sequence = self.data[seq_idx]

        path = (
            sequence.depth_path[frame_idx]
            .replace("depth0", "seen_features")
            .replace(".png", "")
            + ".ir0.csv"
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
            sequence.depth_path[frame_idx]
            .replace("depth0", "best_features")
            .replace(".png", "")
            + ".ir0.csv"
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

        # make sure that projector is only on in the current frame
        proj_intensity_t0 = sequence.laser_power[frame_idx]
        proj_intensity_tm1 = sequence.laser_power[prev_idx]
        proj_intensity_tp1 = sequence.laser_power[next_idx]

        if (proj_intensity_t0 == 0 or proj_intensity_tm1 != 0 or proj_intensity_tp1 != 0) and (
            not self.is_test
        ):
            raise ValueError("Something is wrong with the dataset or loader")

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
        if sequence.T_WB is None:
            assert self.is_test
            T_WB = np.identity(4)
        else:
            T_WB = sequence.T_WB[frame_idx]

        # camera to world (T_WC = T_WB @ T_BS @ T_SC)
        T_ir0 = self.default_calib_dict["ir0"]["T"]
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

        T_BS = self.default_calib_dict["imu_params"]["T_BS"]
        if sequence.T_WB is None:
            assert self.is_test
            T_WB = np.identity(4)
        else:
            T_WB = sequence.T_WB[frame_idx]

        # camera to world (T_WC = T_WB @ T_BS @ T_SC)
        T_ir1 = self.default_calib_dict["ir1"]["T"]
        return np.matmul(np.matmul(T_WB, T_BS), T_ir1).astype(dtype=np.float32)

    def get_pose_depth(self, index: int) -> ArrayF32:
        """Get pose of depth

        :param index: Current frame index
        :return: Pose
        """
        seq_idx, frame_idx = self.index[index]
        sequence = self.data[seq_idx]

        T_BS = self.default_calib_dict["imu_params"]["T_BS"]
        if sequence.T_WB is None:
            assert self.is_test
            T_WB = np.identity(4)
        else:
            T_WB = sequence.T_WB[frame_idx]

        # camera to world (T_WC = T_WB @ T_BS @ T_SC)
        T_dep = self.default_calib_dict["dep"]["T"]
        return np.matmul(np.matmul(T_WB, T_BS), T_dep).astype(dtype=np.float32)

    def get_K_ir0(self, index: int) -> ArrayF32:
        """Get IR0 K matrix

        :param index: Current frame index
        :return: K matrix
        """
        return self.default_calib_dict["ir0"]["K"]

    def get_K_ir1(self, index: int) -> ArrayF32:
        """Get IR1 K matrix

        :param index: Current frame index
        :return: K matrix
        """
        return self.default_calib_dict["ir1"]["K"]

    def get_K_depth(self, index: int) -> ArrayF32:
        """Get depth K matrix

        :param index: Current frame index
        :return: K matrix
        """
        return self.default_calib_dict["dep"]["K"]

    def _load_seq(self, datapath: str, name: str) -> RealsenseSequence:
        """Loads a sequence

        :param datapath: Path of the sequence
        :param name: Name of the sequence
        :return: Sequence information
        """
        position_folder: str = "position_ssw"
        data = {}
        _lengths = []
        for sensor in ["depth0", "ir0", "ir1"]:
            data_csv_file = os.path.join(datapath, sensor, "data.csv")
            if not os.path.isfile(data_csv_file):
                raise FileNotFoundError("File does not exist: {}".format(data_csv_file))
            data[sensor] = pd.read_csv(data_csv_file).values
            data[sensor] = data[sensor][np.argsort(data[sensor][:, 0]), :]  # sort by timestamp
            _lengths.append(len(data[sensor]))

        # make sure they have equal length
        for sensor in ["depth0", "ir0", "ir1"]:
            data[sensor] = data[sensor][: min(_lengths)]
            data[sensor][:, 1] = [
                os.path.join(datapath, sensor, "data", fn) for fn in data[sensor][:, 1]
            ]

        # find corresponding ir0 and ir1 image
        depth_time = data["depth0"][:, 0]
        ir0_time = data["ir0"][:, 0]
        ir1_time = data["ir1"][:, 0]

        matching_index_depth = []
        matching_index_ir1 = []
        valid_ir = []
        for t in ir0_time:
            valid_depth_time = depth_time[depth_time <= t]
            if valid_depth_time.shape[0] == 0:
                valid_ir.append(False)
                continue

            closest_frame_idx_depth = np.argmin(t - valid_depth_time)
            depth_time_match = valid_depth_time[closest_frame_idx_depth]

            # get idx for id1
            closest_frame_idx_ir1 = np.argmin((ir1_time - t) ** 2)

            if depth_time_match == t and depth_time_match == ir1_time[closest_frame_idx_ir1]:
                valid_ir.append(True)
                matching_index_ir1.append(closest_frame_idx_ir1)
                matching_index_depth.append(closest_frame_idx_depth)
            else:
                valid_ir.append(False)

        timestamp = data["depth0"][matching_index_depth, 0].astype(np.float32)
        depth_path = data["depth0"][matching_index_depth, 1]
        ir0_path = data["ir0"][valid_ir, 1]
        ir1_path = data["ir1"][matching_index_ir1, 1]
        laser_power = data["ir1"][matching_index_ir1, -1].astype(np.int_)

        if self.is_test:
            has_gt = []
            assert os.path.isdir(os.path.join(self.root_path, "colmap_gt_flt"))
            for frame_idx in range(depth_path.shape[0]):
                filename = os.path.basename(depth_path[frame_idx])
                path = os.path.join(self.root_path, "colmap_gt_flt", name, filename)
                has_gt.append(os.path.isfile(path))

            # filter out images without ground truth
            timestamp = timestamp[has_gt]
            depth_path = depth_path[has_gt]
            ir0_path = ir0_path[has_gt]
            ir1_path = ir1_path[has_gt]
            laser_power = laser_power[has_gt]

        T_WB_interpolated: Optional[ArrayF32]
        lost_interpolated: Optional[ArrayInt]
        if self.is_test:
            # all images are valid
            valid_idx = np.array([True] * depth_path.shape[0])
            # only with laser images
            idx = np.where(laser_power[valid_idx] > 0)[0]
            T_WB_interpolated = None
            lost_interpolated = None
        else:
            # load postion information
            trajectory_file = os.path.join(
                datapath, position_folder, "optimised_trajectory_0.csv"
            )
            if not os.path.isfile(trajectory_file):
                raise FileNotFoundError("File does not exist: {}".format(trajectory_file))
            T_WB_disk = pd.read_csv(trajectory_file).values

            # quaternion (w, x, y, z) -> (x, y, z, w)
            quaternion_q = T_WB_disk[:, 4]
            T_WB = np.concatenate(
                (T_WB_disk[:, :4], T_WB_disk[:, 5:], quaternion_q[:, np.newaxis]), axis=1
            ).astype(np.float32)

            tracking_file = os.path.join(datapath, position_folder, "tracking.txt")
            if not os.path.isfile(tracking_file):
                raise FileExistsError("File does not exist: {}".format(tracking_file))
            lost = pd.read_csv(tracking_file).values.reshape(-1).astype(np.int_)

            T_WB_interpolated, lost_interpolated, valid_idx = self._interpolate(
                T_WB, lost, timestamp
            )

            # only with laser images
            idx = np.where(laser_power[valid_idx] > 0)[0]

            # Find indices that have both image in front and after
            idx = idx[(idx > 0) * (idx < len(laser_power[valid_idx]) - 1)].astype(np.int_)

        # append a attribute saying which sequence a frame is from
        seq = os.path.basename(datapath)
        folder_name = [
            os.path.basename(seq) + "_" + os.path.basename(ts).replace(".png", "")
            for ts in depth_path
        ]
        folder_name = np.asarray(folder_name).reshape(-1, 1)

        sequence = RealsenseSequence(
            name,
            timestamp[valid_idx],
            depth_path[valid_idx],
            ir0_path[valid_idx],
            ir1_path[valid_idx],
            laser_power[valid_idx],
            T_WB_interpolated,
            lost_interpolated,
            idx,
            folder_name,
        )

        return sequence

    def _load_data(
        self, datapath: str
    ) -> Tuple[List[RealsenseSequence], List[Tuple[int, int]]]:
        """Loads all sequences

        :param datapath: Path of the data
        :return: List of sequences and valid indexes
        """
        data = []
        index = []
        datasets = os.listdir(datapath)
        datasets_filtered = []

        for seq in datasets:
            datasets_filtered.append(seq)

        for i, seq in enumerate(datasets_filtered):
            seq_data = self._load_seq(os.path.join(datapath, seq), seq)

            for j in seq_data.idx:
                # Save sequence id and image id
                index.append((i, j))

            data.append(seq_data)

        return data, index

    def _interpolate(
        self,
        T_WB: ArrayF32,
        lost: ArrayInt,
        t_ir0: ArrayF32,
    ) -> Tuple[ArrayF32, ArrayInt, ArrayInt]:
        """Interpolate poses

        :param T_WB: List of poses
        :param lost: List of lost booleans
        :param t_ir0: List of timestamps
        :return: Interpolated values and valid indexes
        """
        offset: float = 0
        t = T_WB[:, 0] - offset

        # times (find idx where we have between slam postion estimates)
        idx = np.where((t_ir0 >= min(t)) * (t_ir0 <= max(t)))[0]
        t_ir0_with_pos = t_ir0[idx]

        # interpolate  translation
        x = T_WB[:, 1]
        y = T_WB[:, 2]
        z = T_WB[:, 3]

        f = interp1d(t, x, kind="linear")
        new_x = f(t_ir0_with_pos)

        f = interp1d(t, y, kind="linear")
        new_y = f(t_ir0_with_pos)

        f = interp1d(t, z, kind="linear")
        new_z = f(t_ir0_with_pos)

        # interpolate rotations
        q = T_WB[:, 4:]
        q = R.from_quat(q)

        f = Slerp(t, q)
        q_new = f(t_ir0_with_pos)

        # initialize T
        T = np.diag(np.ones(4))
        T = np.repeat(T[None, :, :], len(t_ir0_with_pos), axis=0)

        # interpolate
        lost = np.insert(lost, 0, 1, axis=0)  # you can be lost at step = 0
        lost = (lost == 1) * 1 + (lost == 3) * 1
        f = interp1d(t, lost)
        new_lost = f(t_ir0_with_pos)

        # insert into T (here we have some padding to get the same length of the images)
        # This makes indexing in getitem significantly easier
        T[:, :3, :3] = q_new.as_matrix()
        T[:, 0, 3] = new_x
        T[:, 1, 3] = new_y
        T[:, 2, 3] = new_z

        # reshape T to fit into dataframe
        return T, new_lost.reshape(-1, 1), idx

    def _load_im(self, path: str) -> ArrayUI8:
        """Load an image

        :param path: Path
        :return: Image
        """

        im = read_image(path)
        assert im.dtype == np.uint8

        return im

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

        points = data[:, :3].astype(np.float32)
        return points

    def _load_depth(self, path: str) -> Tuple[ArrayF32, ArrayF32]:
        """Load depth image

        :param path: Path to depth image
        :return: Depth image and valid mask Image
        """
        dep = read_image(path)
        assert dep.dtype == np.uint16

        dep = dep.astype(np.float32) / (1000.0)
        # dep stored in milimeters so dividing with 1000 gives meters

        valid_mask = (dep > 0).astype(np.float32)
        valid_mask = cast(ArrayF32, valid_mask)

        return dep, valid_mask
