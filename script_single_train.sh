cuda_device='0'
iter='2000'
voxel_size=0.1
save_iterations='500 1000 2000 3000'
opacity_reset_interval='1000'

scene_tag="office0"
train_data="data/"$scene_tag"/train_full_byorder_85"
output_folder="outputs/"$scene_tag"/"$voxel_size
mkdir -p $output_folder

# echo \
# $scene_tag \
# $train_data \
# $output_folder \
# $save_iterations \
# $iter \
# $voxel_size \
# $opacity_reset_interval

python train.py \
-s $train_data \
-m $output_folder \
--save_iterations $save_iterations \
--iterations $iter \
--checkpoint_iterations $iter \
--init_w_gaussian \
--voxel_size $voxel_size \
--densify_from_iter 100 \
--opacity_reset_interval $opacity_reset_interval
