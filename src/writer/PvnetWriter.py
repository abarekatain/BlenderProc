import json
import os
import numpy as np

import bpy

from src.writer.WriterInterface import WriterInterface

from src.utility.BlenderUtility import get_all_blender_mesh_objects

from mathutils import Matrix
from src.utility.WriterUtility import WriterUtility


class PvnetWriter(WriterInterface):

    def __init__(self, config):
        WriterInterface.__init__(self, config)
        
        # Parse configuration.
        #self._dataset = self.config.get_string("dataset", "")
    @staticmethod
    def save_json(path, content):
        """ Saves the content to a JSON file in a human-friendly format.
        From the BOP toolkit (https://github.com/thodan/bop_toolkit).

        :param path: Path to the output JSON file.
        :param content: Dictionary/list to save.
        """
        with open(path, 'w') as f:

            if isinstance(content, dict):
                f.write('{\n')
                content_sorted = sorted(content.items(), key=lambda x: x[0])
                for elem_id, (k, v) in enumerate(content_sorted):
                    f.write(
                        '  \"{}\": {}'.format(k, json.dumps(v, sort_keys=True)))
                    if elem_id != len(content) - 1:
                        f.write(',')
                    f.write('\n')
                f.write('}')

            elif isinstance(content, list):
                f.write('[\n')
                for elem_id, elem in enumerate(content):
                    f.write('  {}'.format(json.dumps(elem, sort_keys=True)))
                    if elem_id != len(content) - 1:
                        f.write(',')
                    f.write('\n')
                f.write(']')

            else:
                json.dump(content, f, sort_keys=True)

    @staticmethod
    def get_frame_gt(dataset_objects, destination_frame = ["X", "-Y", "-Z"]):
        """ Returns GT pose annotations between active camera and objects.
        
        :return: A list of GT annotations.
        """
        
        H_c2w_opencv = Matrix(WriterUtility.get_cam_attribute(bpy.context.scene.camera, 'cam2world_matrix', destination_frame))
        
        frame_gt = []
        for obj in dataset_objects:
            
            H_m2w = Matrix(WriterUtility.get_common_attribute(obj, 'matrix_world'))

            cam_H_m2c = H_c2w_opencv.inverted() @ H_m2w
            cam_R_m2c = cam_H_m2c.to_quaternion().to_matrix()
            cam_t_m2c = cam_H_m2c.to_translation()

            cam_t_m2c = list(cam_t_m2c)
            frame_gt.append({
                'cam_R_m2c': list(cam_R_m2c[0]) + list(cam_R_m2c[1]) + list(cam_R_m2c[2]),
                'cam_t_m2c': cam_t_m2c,
                'obj_id': obj["category_id"]
            })  
        return frame_gt

    def run(self):
        # Output paths.
        output_dir = self._determine_output_dir(False)
        dataset_dir = os.path.join(output_dir, "pvnet_test")
        rgb_dir = os.path.join(dataset_dir, "rgb")
        mask_dir = os.path.join(dataset_dir, 'mask')
        pose_dir = os.path.join(dataset_dir, 'pose')

        debug_path = os.path.join(dataset_dir, 'debug.json')

        # Create the output directory structure.
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
            os.makedirs(rgb_dir)
            os.makedirs(mask_dir)
            os.makedirs(pose_dir)

        # Return GT pose
        all_mesh_objects = get_all_blender_mesh_objects()

        temp = self.get_frame_gt(all_mesh_objects)

        print("-----------------------------------")
        print(temp)
        #self.save_json(debug_path,all_mesh_objects[0])
