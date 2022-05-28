import argparse
from textwrap import indent
import driveandact as daa
import cv2
import numpy as np
import os
from pprint import pprint

import open3d as o3d

class O3DVisualizer:
    '''
    Wrapper around Open3Ds basic visualizer providing convenience functions to draw 3d annotations of the dataset
    and managing the open3d window.
    '''
    def __init__(self):
        self.set_view = True
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
        origin.compute_triangle_normals()
        self.vis.add_geometry(origin, False)
        box = o3d.geometry.TriangleMesh.create_box(5, 5, 5)
        self.geometry = [box]

    def add_image(self, image):
        self.geometry.append(o3d.geometry.Image(image))

    def add_interior(self, interior):
        geometry = self._render_interior(interior)
        for name, geom in geometry:
            self.geometry.append(geom)

    def add_keypoints(self, positions, probabilities, names, color, size):
        geometry = self.create_pose_geometry(positions, probabilities, names,
                                             formatting={'keypoint_color': color,
                                                         'keypoint_radius': size})
        for name, geom in geometry:
            self.geometry.append(geom)

    def spin_once(self):
        for geom in self.geometry:
            self.vis.add_geometry(geom, reset_bounding_box=self.set_view)
            self.set_view = False
        res = self.vis.poll_events()
        self.vis.update_renderer()
        for geom in self.geometry:
            self.vis.remove_geometry(geom, reset_bounding_box=False)
        self.geometry = []
        return res

    def _render_interior(self, interior):
        out = []
        for element in interior.elements:
            if element.type == "Cube":
                box = o3d.geometry.TriangleMesh.create_box(element.dimensions[0],
                                                     element.dimensions[1],
                                                     element.dimensions[2])
                box = box.translate(-element.dimensions/2)
                box = box.transform(element.transformation_matrix)
                box.compute_vertex_normals()
                out.append((element.id + '_box', box))
            elif element.type == "Cylinder":
                cyl = o3d.geometry.TriangleMesh.create_cylinder(element.dimensions[0], element.dimensions[2])
                cyl = cyl.transform(element.transformation_matrix)
                cyl.compute_vertex_normals()
                # sphere.paint_uniform_color(center_color)
                out.append((element.id + '_cylinder', cyl))
            coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
            coord = coord.transform(element.transformation_matrix)
            out.append((element.id, coord))
        return out

    def create_pose_geometry(self, positions, probabilities, names, postfix='', formatting={}):
        keypoint_color = formatting.get('keypoint_color', (1, 0, 0))
        keypoint_radius = formatting.get('keypoint_radius', 0.02)
        if not isinstance(keypoint_radius, dict):
            keypoint_radius = {name: keypoint_radius for name in names}
        bone_color = formatting.get('bone_color', (0, 1, 0))
        bone_radius = formatting.get('bone_radius', 0.01)
        render_text = formatting.get('render_text', True)

        out = []
        for name, pos, prob in zip(names,
                                   positions,
                                   probabilities):
            if prob == 0:
                continue

            sphere = o3d.geometry.TriangleMesh.create_sphere(keypoint_radius[name], resolution=10)
            sphere.compute_vertex_normals()
            if not isinstance(keypoint_color, dict):
                sphere.paint_uniform_color(keypoint_color)
            else:
                sphere.paint_uniform_color(keypoint_color[name])
            sphere = sphere.translate(pos)
            out.append((name + postfix, sphere))

        bones = [
            # head
            ('neck', 'head'),
            ('neck', 'nose'),
            ('nose', 'rEye'),
            ('rEye', 'rEar'),
            ('nose', 'lEye'),
            ('lEye', 'lEar'),
            # torso
            ('neck', 'midHip'),
            ('neck', 'rShoulder'),
            ('neck', 'lShoulder'),
            ('midHip', 'rHip'),
            ('midHip', 'lHip'),
            # arms
            ('rShoulder', 'rElbow'),
            ('rElbow', 'rWrist'),
            ('rWrist', 'rHandTip'),
            ('lShoulder', 'lElbow'),
            ('lElbow', 'lWrist'),
            ('lWrist', 'lHandTip'),
            # legs
            ('rHip', 'rKnee'),
            ('rKnee', 'rAnkle'),
            ('lHip', 'lKnee'),
            ('lKnee', 'lAnkle'),
        ]
        valid_bone_ids = []
        for start, end in bones:
            try:
                start_id = names.index(start)
                end_id = names.index(end)
                if probabilities[start_id] > 0 and probabilities[
                    end_id] > 0:
                    valid_bone_ids.append([start_id, end_id])
            except:
                pass
        if len(valid_bone_ids) > 0:
            bones = o3d.geometry.LineSet()
            bones.points = o3d.utility.Vector3dVector(positions)
            bones.lines = o3d.utility.Vector2iVector(valid_bone_ids)
            out.append(('bones', bones))
        return out

class RenderCV2:
    '''
    Renders 2D data for specified frames using opencv drawing functions
    '''
    WIN_NAME = "DAA Viewer"
    TRACKBAR_NAME = 'progress'
    def __init__(self, num_frames, undistort_images):
        self._clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
        self.is_paused = False
        self.rate = 15
        self.help_text = [
            'Pause: SPACE',
            'Quit: q'
        ]
        self._undistort_images = undistort_images

    def render(self, data, frame_id):
        img = data.video[frame_id]
        if self._undistort_images:
            img = cv2.undistort(img, data.video_calibration.camera_matrix, data.video_calibration.distortion)
        gray_scale = np.allclose(img[:, :, 0], img[:, :, 1])
        if gray_scale:
            if len(img.shape) == 2:
                img = np.expand_dims(img,axis=2)
            img = (img.astype(np.float32) / np.quantile(img[:, :, 0], 0.99) * 255)
            img = np.clip(img, 0, 255).astype(np.uint8)
            img = self._clahe.apply(img[:, :, 0])
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


        if not data.keypoints_2d.empty:
            frame_index = data.keypoints_2d.index(frame_id)
            daa.draw_keypoints(img, data.keypoints_2d.positions[frame_index], data.keypoints_2d.probabilities[frame_index],
                           data.keypoints_2d.keypoint_names, color=(0,255,0))
        if not data.keypoints_3d_repro.empty:
            frame_index = data.keypoints_3d_repro.index(frame_id)
            daa.draw_keypoints(img, data.keypoints_3d_repro.positions[frame_index], data.keypoints_3d_repro.probabilities[frame_index],
                           data.keypoints_3d_repro.keypoint_names, color=(0,0,255), color_lines=(0,0,200))

        if not data.objects_3d_repro.empty:
            frame_index = data.objects_3d_repro.index(frame_id)
            daa.draw_keypoints(img, data.objects_3d_repro.positions[frame_index], data.objects_3d_repro.probabilities[frame_index],
                           data.objects_3d_repro.keypoint_names, color=(255,0,0), color_lines=(200,0,0))

        frame_bboxes = data.bounding_boxes.get_boxes_by_frame_id(frame_id)
        daa.draw_bounding_boxes(img, frame_bboxes)
        img = self._draw_interior(img, data.interior_2d_pos)

        action_midlevel = data.actions_midlevel.get_action_by_frame_id(data.video_id, frame_id)
        action_objectlevel = data.actions_objectlevel.get_action_by_frame_id(data.video_id, frame_id)
        action_tasklevel = data.actions_tasklevel.get_action_by_frame_id(data.video_id, frame_id)
        text = []
        text.extend(self.help_text)
        text.append(str(action_tasklevel))
        text.append(str(action_midlevel))
        text.append(str(action_objectlevel))
        self._draw_text(img, text, (5,5), (255, 255, 255))
        return img

    def _draw_interior(self, img, interior_2d_pos):
        for name, pt in interior_2d_pos:
            radius = 5
            scale = img.shape[1] / 1280
            cv2.circle(img, tuple(pt.astype(int)), radius, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.putText(img, name, tuple(pt.astype(int)) + np.array([radius, radius]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7 * scale, (0, 255, 0), thickness=int(2 * scale),
                        lineType=cv2.LINE_AA)
        return img

    def _draw_text(self, img, text, pos, color):
        spacing = 30
        for line_id, text in enumerate(text):
            cv2.putText(img, text, (pos[1] + 5, pos[0] + 30 + line_id * spacing), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color,
                        thickness=2, lineType=cv2.LINE_AA)
        return img

class RenderO3D:
    '''
    Renders 3D data for specific frames using the open3d wrapper from above
    '''
    def __init__(self):
        self.visualizer = O3DVisualizer()
        self.visualizer_2d = O3DVisualizer()

    def add_image(self, img):
        self.visualizer_2d.add_image(img)

    def render(self, data, frame_id):
        self.visualizer.add_interior(data.interior)
        if not data.keypoints_3d.empty:
            frame_index = data.keypoints_3d.index(frame_id)
            self.visualizer.add_keypoints(data.keypoints_3d.positions[frame_index],
                                          data.keypoints_3d.probabilities[frame_index],
                                          data.keypoints_3d.keypoint_names,
                                          (0,0,1), 0.02)
        if not data.objects_3d.empty:
            frame_index = data.objects_3d.index(frame_id)
            self.visualizer.add_keypoints(data.objects_3d.positions[frame_index],
                                          data.objects_3d.probabilities[frame_index],
                                          data.objects_3d.keypoint_names,
                                          (0,1,0), 0.1)

    def spin_once(self):
        res = self.visualizer.spin_once()
        res = res and self.visualizer_2d.spin_once()
        return res


class DAADataManager:
    '''
    Manages all annotations specified on the command line making sure to initialize all structures even if not specified.
    '''
    def __init__(self, args):
        self.video = daa.VideoReader.create(args.video)
        self.frame_ids = self.video.frame_ids
        self.video_id = daa.id_from_path(args.video)
        file_id, camera_id, vp = daa.split_id(self.video_id)
        if not args.no_undistort:
            calibrations = daa.load_all_calibrations(args.calibrations, cameras=[camera_id], participants=[vp])
            self.video_calibration = calibrations[self.video_id]

        # load 2d keypoints the function will return both camera streams of a participant.
        all_keypoints = daa.load_all_keypoints(args.keypoints2d, cameras=[camera_id], post_fix=args.keypoints2d_postfix, participants=[vp])
        # Select the right stream based on the video id. If it can not be found we initialize with an empty keypoint stream
        self.keypoints_2d = all_keypoints.get(self.video_id, daa.Keypoints())

        # load 3d keypoints
        all_keypoints = daa.load_all_keypoints(args.keypoints3d, cameras=[camera_id], post_fix=args.keypoints3d_postfix, participants=[vp])
        self.keypoints_3d = all_keypoints.get(self.video_id, daa.Keypoints())

        # project 3d keypoints onto the camera image plane
        self.keypoints_3d_repro = daa.Keypoints()
        if not self.keypoints_3d.empty:
            # only load calibrations when needed for projection
            calibrations = daa.load_all_calibrations(args.calibrations, cameras=[camera_id], participants=[vp])
            self.video_calibration = calibrations[self.video_id]
            self.keypoints_3d_repro = self.keypoints_3d.copy()
            self.keypoints_3d_repro.positions = self.video_calibration.transform_keypoints(self.keypoints_3d_repro.positions, daa.CameraCalibration.CAMERA, daa.CameraCalibration.RECTIFIED)

        # load bounding boxes
        all_bounding_boxes = daa.load_all_bounding_boxes(args.bbox, cameras=[camera_id], participants=[vp])
        self.bounding_boxes = all_bounding_boxes.get(self.video_id, daa.BBoxTrackStorage())

        #load interior
        all_interiors = daa.load_all_interiors(args.interior, cameras=[camera_id], participants=[vp])
        self.interior = all_interiors.get(self.video_id, daa.Interior())

        # project interior element positions to 2d as placeholders for 2d visualization
        self.interior_2d_pos =  np.array([])
        if len(self.interior.elements) > 0:
            self.interior_2d_pos = self._project_interior(self.interior, self.video_calibration)

        # load 3d object positions
        all_keypoints = daa.load_all_keypoints(args.objects, cameras=[camera_id], post_fix='objects.3d.csv', participants=[vp])
        self.objects_3d = all_keypoints.get(self.video_id, daa.Keypoints())
        # project 3d object positions onto the camera image plane
        self.objects_3d_repro = daa.Keypoints()
        if not self.objects_3d.empty:
            calibrations = daa.load_all_calibrations(args.calibrations, cameras=[camera_id], participants=[vp])
            self.video_calibration = calibrations[self.video_id]
            self.objects_3d_repro = self.objects_3d.copy()
            self.objects_3d_repro.positions = self.video_calibration.transform_keypoints(self.objects_3d_repro.positions, daa.CameraCalibration.CAMERA, daa.CameraCalibration.RECTIFIED)

        #load actions
        self.actions_tasklevel = self._load_activities(args.actions, level='tasklevel', video_id=self.video_id)
        self.actions_midlevel = self._load_activities(args.actions, level='midlevel', video_id=self.video_id)
        self.actions_objectlevel = self._load_activities(args.actions, level='objectlevel', video_id=self.video_id)

    def _project_interior(self, interior, calibration):
        '''
        3D primitives are hard to render on the image. As a workaround we project the center point of each interior primitive as placeholder
        '''
        if len(interior.elements) == 0:
            return np.array([])
        element_names = [element.id for element in interior.elements]
        center_points = [element.translation for element in interior.elements]
        center_points = np.stack(center_points, axis=0)
        center_points = calibration.transform_keypoints(np.expand_dims(center_points, axis=0),
                                                        daa.CameraCalibration.CAMERA,
                                                        daa.CameraCalibration.RECTIFIED)[0]
        named_center_points = [(name, center) for name, center in zip(element_names, center_points)]
        return named_center_points

    def _load_activities(self, activity_folder, level, video_id):
        '''
        While all other annotations are saved per video. Activities are stored in a single file but split into train, val, test.
        Activity_folder is therefore the folder of the specific camera. The function searches split0 train/val/test to find the
        annotations for the video specified by video_id
        '''
        if activity_folder is None:
            return daa.ActionAnnotations()
        parts = ['train', 'val', 'test']
        activity_path_template = os.path.join(activity_folder, '{}.chunks_90.split_0.{}.csv')
        for part in parts:
            activity_path = activity_path_template.format(level, part)
            activities = daa.ActionAnnotations(activity_path)
            if video_id in activities.video_ids:
                return activities
        return daa.ActionAnnotations()


def main(args):
    print("#" * 10)
    print(type(args))
    pprint(vars(args), indent= 4)
    print("#" * 10)

    data_manager = DAADataManager(args)
    if args.start >= len(data_manager.video):
        print('Can not start from frame {}. Video has only {} frames.'.format(args.start, len(data_manager.video)))
        return
    render_2d = RenderCV2(len(data_manager.video), not args.no_undistort)
    render_3d = RenderO3D()
    for frame_index in range(args.start, len(data_manager.video)):
        frame_id = data_manager.frame_ids[frame_index]
        final_img = render_2d.render(data_manager, frame_id)
        render_3d.add_image(final_img)
        render_3d.render(data_manager, frame_id)
        if not render_3d.spin_once():
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Basic visualizer for all of the drive and act annotations.'
                                                 'Annotations are on rectified images while video data is usually still unrectified.'
                                                 'To get matching visualizations video data should be undistorted first')

    parser.add_argument('-v', '--video', required=True, help='Path to the video file.')
    parser.add_argument('-u', '--no_undistort', action='store_true', help='Do not undistort images.')
    parser.add_argument('-k2', '--keypoints2d', default=None, help='Keypoint2d root dir.')
    parser.add_argument('--keypoints2d_postfix', default='openpose.2d.csv', help='Postfix of keypoint 2d annotation.')
    parser.add_argument('-k3', '--keypoints3d', default=None, help='Keypoint3d root dir.')
    parser.add_argument('--keypoints3d_postfix', default='openpose.3d.csv', help='Postfix of keypoint 3d annotation.')
    parser.add_argument('-c', '--calibrations', default=None, help='Base path of the calibration.')

    parser.add_argument('-b', '--bbox', default=None, help='Base path to bounding boxes.')
    parser.add_argument('-o', '--objects', default=None, help='Base path to 3d object annotations.')
    parser.add_argument('-i', '--interior', default=None, help='Base path to the interior definitions.')
    parser.add_argument('-a', '--actions', default=None, help='Camera folder of action annotations.')
    parser.add_argument('-s', '--start', type=int, default=0, help='Start frame for visualization.')

    args = parser.parse_args()

    if args.calibrations is None:
        if not args.no_undistort:
            print('either specify [--no_undistort] or provide [--calibrations]')
            print('if you work with original drive and act data you likely want to undistort to match the pose data.')
            exit(0)
        if args.objects or args.interior or args.keypoints3d:
            print('you specified 3d annotations please provide [--calibrations]')
            exit(0)
    main(args)
