cuda_device='0'
iter='2000'
voxel_size=0.1
# rendering_flags='--skip_train --skip_test'
rendering_flags=''

scene_tag="office0"
test_data="data/"$scene_tag"/train"
model_folder="outputs/"$scene_tag"/"$voxel_size
output_folder="evals/"$scene_tag"/"$voxel_size
mkdir -p $output_folder

python render.py \
-s $test_data \
-m $model_folder \
--iteration $iter \
$rendering_flags 