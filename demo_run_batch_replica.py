import subprocess
import sys
    


command = f"CUDA_VISIBLE_DEVICES={cuda_device} python train.py -s {data}\
            -m {output_path}{folder_path}/only_ds --single_frame_id {frame_ids}\
            --save_iterations {save_iterations}  --iterations {iter} --checkpoint_iterations {iter} --init_w_gaussian --voxel_size {voxel_size} --densify_from_iter 100 --opacity_reset_interval {opacity_reset_interval} --DS 0.1 --CS 0"
subprocess.run(command+suffix, shell=True)

command = f"CUDA_VISIBLE_DEVICES={cuda_device} python train.py -s {data}\
            -m {output_path}{folder_path}/ds_fovmask --single_frame_id {frame_ids}\
            --save_iterations {save_iterations}  --iterations {iter} --checkpoint_iterations {iter} --init_w_gaussian --voxel_size {voxel_size} --densify_from_iter 100 --opacity_reset_interval {opacity_reset_interval} --DS 0.1 --fov_mask"
subprocess.run(command+suffix, shell=True)

command = f"CUDA_VISIBLE_DEVICES={cuda_device} python train.py -s {data}\
            -m {output_path}{folder_path}/dist_loss --single_frame_id {frame_ids}\
            --save_iterations {save_iterations}  --iterations {iter} --checkpoint_iterations {iter} --init_w_gaussian --voxel_size {voxel_size} --densify_from_iter 100 --opacity_reset_interval {opacity_reset_interval} --dist"
subprocess.run(command+suffix, shell=True)

command = f"CUDA_VISIBLE_DEVICES={cuda_device} python train.py -s {data}\
            -m {output_path}{folder_path}/reset_opa --single_frame_id {frame_ids}\
            --save_iterations {save_iterations}  --iterations {iter} --checkpoint_iterations {iter} --init_w_gaussian --voxel_size {voxel_size} --densify_from_iter 100 --opacity_reset_interval {opacity_reset_interval} --reset_opa_near"
subprocess.run(command+suffix, shell=True)

command = f"CUDA_VISIBLE_DEVICES={cuda_device} python train.py -s {data}\
            -m {output_path}{folder_path}/reset_opa_fovmask --single_frame_id {frame_ids}\
            --save_iterations {save_iterations}  --iterations {iter} --checkpoint_iterations {iter} --init_w_gaussian --voxel_size {voxel_size} --densify_from_iter 100 --opacity_reset_interval {opacity_reset_interval} --reset_opa_near --fov_mask"
subprocess.run(command+suffix, shell=True)

command = f"CUDA_VISIBLE_DEVICES={cuda_device} python train.py -s {data}\
            -m {output_path}{folder_path}/ds_reset_opa_fovmask --single_frame_id {frame_ids}\
            --save_iterations {save_iterations}  --iterations {iter} --checkpoint_iterations {iter} --init_w_gaussian --voxel_size {voxel_size} --densify_from_iter 100 --opacity_reset_interval {opacity_reset_interval} --DS 0.1 --reset_opa_near --fov_mask"
subprocess.run(command+suffix, shell=True)


# Ours
command = f"CUDA_VISIBLE_DEVICES={cuda_device} python train.py -s {data}\
            -m {output_path}{folder_path}/SAD_GS --single_frame_id {frame_ids}\
            --save_iterations {save_iterations} --iterations {iter} --checkpoint_iterations {iter} --init_w_gaussian --voxel_size {voxel_size} --densify_from_iter 100 --opacity_reset_interval {opacity_reset_interval} --CS 10 --cls_loss --full_reset_opa"
subprocess.run(command+suffix, shell=True)