# insert your blender PATH
blender='/home/laurenz/blender/blender'

OOP=FALSE

#$blender --background --python gen.py -- --dataroot '/media/laurenz/Seagate Backup Plus Drive/ShapeNet/ShapeNetCore.v2/' --outroot '/home/laurenz/IITGN/motion_transfer/data_generation/data/'

$blender  --background --python Generate_Dataset.py -- --number_of_vids 50 --dataroot '/media/laurenz/Seagate Backup Plus Drive/ShapeNet/ShapeNetCore.v2/' --outroot '/home/laurenz/IITGN/motion_transfer/datasets/MT/train/'
