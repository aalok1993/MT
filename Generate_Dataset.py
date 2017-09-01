### Data Generation Code for Motion Transfer


# Make sure that GPU is used while rendering. (1) Enable in user prefs, (2) enable in render properties bpy.context.scene.cycles.device = 'GPU'
# Reference: https://blender.stackexchange.com/questions/7485/enabling-gpu-rendering-for-cycles/7486#7486

##############################
# Execute this script in the background in console
# Reference: https://blender.stackexchange.com/questions/6817/how-to-pass-command-line-arguments-to-a-blender-python-script
# $ blender --background test.blend --python Generate_Dataset.py -- example args 123
##################################
 
# All the essential libraries required are being imported below.
import bpy
from mathutils import Vector
from math import radians, tan
import random
from bpy.types import Operator
import numpy as np
#from bpy.props import *


# Define all the functions, parameters, options that are 
# going to be used in the code. 

# Variables
IMAGE_SIZE = 256
# MAX_COV = 0.4
# MIN_COV = 0.2
COVERAGE = 1/3
CAM_DISTANCE = 10
origin = Vector((0,0,0))
camera_location = Vector((0,0,CAM_DISTANCE))
camera_rot = Vector((0.0,0.0,0.0))
lamp_location = Vector((0,0,5))
lamp_rot = Vector((0.0,0.0,0.0))
cameraName1 = 'Camera1'
cameraName2 = 'Camera2'
lampName1 = 'Lamp1'
lampName2 = 'Lamp2'
sceneName1 = 'Scene1'
sceneName2 = 'Scene2'
OUT_PATH = '/media/innit/Zone_D/MotionTransfer/DataGeneration/Data/'
num_frames = 10
USE_GPU = True
N = 1

def addScene(sceneName, engine, filepath, image_size, resolution_percentage, use_gpu):
    scene = bpy.data.scenes.new(sceneName)
    scene.render.engine = engine
    scene.render.filepath = filepath
    scene.render.resolution_x = image_size
    scene.render.resolution_y = image_size
    scene.render.resolution_percentage = resolution_percentage
    if use_gpu: 
        scene.cycles.device = 'GPU'
    return scene

# A function to make a camera point towards any point in space
# Reference: https://blender.stackexchange.com/questions/5210/pointing-the-camera-in-a-particular-direction-programmatically
def look_at(obj, point):
    loc_obj = Vector(obj.location)
    direction = point - loc_obj
    rot_quat = direction.to_track_quat('-Z', 'Y')       # point the cameras '-Z' and use its 'Y' as up
    obj.rotation_euler = rot_quat.to_euler()     # assume we're using euler rotation
    
# A function to add camera and look at X
def addCameras(camera_name,loc_point,loc_rot,bIsPoint,pointTo = None):
    camera_data = bpy.data.cameras.new(camera_name)
    camera = bpy.data.objects.new(camera_name, camera_data)
    camera.location = loc_point
    #camera.rotation_euler = (loc_rot[0]*constRadToDeg,loc_rot[1]*constRadToDeg, loc_rot[2]*constRadToDeg)
    if bIsPoint:
        look_at(camera, pointTo)
    else:
        camera.rotation_euler = (loc_rot[0]*constRadToDeg,loc_rot[1]*constRadToDeg, loc_rot[2]*constRadToDeg)
    bpy.context.scene.objects.link(camera)
    bpy.context.scene.camera = camera
    return camera

# A function to add lamp and look at X
def addLamps(lamp_name, lamp_type, loc_point, loc_rot, bIsPoint, pointTo = None):
    lamp_data = bpy.data.lamps.new(name=lamp_name, type=lamp_type)
    lamp = bpy.data.objects.new(name=lamp_name, object_data=lamp_data)
    lamp.location = loc_point
    if bIsPoint:
        look_at(lamp, pointTo)
    else:
        lamp.rotation_euler = (loc_rot[0]*constRadToDeg,loc_rot[1]*constRadToDeg, loc_rot[2]*constRadToDeg)
    #lamp.energy = 10  # 10 is the max value for energy
    #lamp.type = 'POINT'  # in ['POINT', 'SUN', 'SPOT', 'HEMI', 'AREA']
    #lamp.distance = 100
    bpy.context.scene.objects.link(lamp)
    return lamp

#Load obj from path, join and select it
#Note: Follow shapenet defined axis in import to use correct object dimensions
def addObject(obj_path,axis_forward=None,axis_up=None):
    prior_objects = [object.name for object in bpy.context.scene.objects]
    if axis_forward and axis_up: 
        bpy.ops.import_scene.obj(filepath=obj_path, axis_forward=axis_forward, axis_up=axis_up, filter_glob="*.OBJ;*.obj")
    elif axis_forward and not axis_up: 
        bpy.ops.import_scene.obj(filepath=obj_path, axis_forward=axis_forward, filter_glob="*.OBJ;*.obj")
    elif not axis_forward and axis_up: 
        bpy.ops.import_scene.obj(filepath=obj_path, axis_up=axis_up, filter_glob="*.OBJ;*.obj")
    elif not axis_forward and not axis_up: 
        bpy.ops.import_scene.obj(filepath=obj_path, filter_glob="*.OBJ;*.obj")
    new_current_objects = [object.name for object in bpy.context.scene.objects]
    new_objects = list(set(new_current_objects)-set(prior_objects))
    bpy.context.scene.objects.active = bpy.data.objects[new_objects[0]]
    for obj in new_objects:
        bpy.data.objects[obj].select = True
    bpy.ops.object.join()
    return bpy.context.active_object

def get_random_rot_euler():
    theta1 = radians(random.uniform(0,360))#random.uniform(0,360)
    theta2 = radians(random.uniform(0,360))
    theta3 = radians(random.uniform(0,360))    
    return ((theta1, theta2, theta3))


# Generate 2 new scenes

# Scene 1
scene1 = addScene(sceneName1, 'CYCLES', OUT_PATH + 'A.png', IMAGE_SIZE, 100, USE_GPU)
scene2 = addScene(sceneName2, 'CYCLES', OUT_PATH + 'B.png', IMAGE_SIZE, 100, USE_GPU)

# Set scene 1 as screen for loading and processing object A
bpy.context.screen.scene = scene1

camera1 = addCameras(cameraName1, camera_location, camera_rot, True, origin)
lamp1 = addLamps(lampName1, 'POINT', lamp_location, lamp_rot, True, origin)
obj_path1 = '/media/innit/Zone_D/ShapeNet/ShapeNetCore.v2/02691156/1eb1d7d471f3c4c0634efb708169415/models/model_normalized.obj'
obj1 = addObject(obj_path1, axis_forward='-X', axis_up='Y')
obj1.rotation_euler = get_random_rot_euler()
scene1.update()


bpy.context.screen.scene = scene2

camera2 = addCameras(cameraName2, camera_location, camera_rot, True, origin)
lamp2 = addLamps(lampName2, 'POINT', lamp_location, lamp_rot, True, origin)
obj_path2 = '/media/innit/Zone_D/ShapeNet/ShapeNetCore.v2/02691156/1d4ff34cdf90d6f9aa2d78d1b8d0b45c/models/model_normalized.obj'
obj2 = addObject(obj_path2, axis_forward='-X', axis_up='Y')
obj2.rotation_euler = get_random_rot_euler()
scene2.update()


# Refer this for FOV length http://www.scantips.com/lights/subjectdistance.html
fov = (CAM_DISTANCE * tan(scene1.camera.data.angle/2))*2
# fov_in_meters = 9.14



def preprocess_object(obj,fov):
    Bx = np.array([(obj.matrix_world * v.co) for v in obj.data.vertices])
    bbox = [ [np.min(Bx[:,0]), np.min(Bx[:,1]), np.min(Bx[:,2])], [np.max(Bx[:,0]), np.max(Bx[:,1]), np.max(Bx[:,2])] ]
    size_obj = [ bbox[1][0]-bbox[0][0], bbox[1][1]-bbox[0][1], bbox[1][2]-bbox[0][2] ]
    #obj.location = obj.location - Vector(( (bbox[0][0]+bbox[1][0])/2, (bbox[0][1]+bbox[1][1])/2, (bbox[0][2]+bbox[1][2])/2 ))   
    size_max = max(size_obj[0:2])
    scale = (fov/size_max)*(COVERAGE)         # this makes the largest dimension of the object cover 1/3rd of the image
    obj.scale = Vector((scale,scale,scale))
    size_obj = size_obj*scale
    return size_obj[0], size_obj[1]

# Bx1 = np.array([(obj1.matrix_world * v.co) for v in obj1.data.vertices])
# bbox1 = [   [np.min(Bx1[:,0]), np.min(Bx1[:,1]), np.min(Bx1[:,2])], 
#             [np.max(Bx1[:,0]), np.max(Bx1[:,1]), np.max(Bx1[:,2])] 
#         ]
# size_obj1 = [ bbox1[1][0]-bbox1[0][0], bbox1[1][1]-bbox1[0][1], bbox1[1][2]-bbox1[0][2] ]
# obj1.location = obj1.location - Vector(( (bbox1[0][0]+bbox1[1][0])/2, (bbox1[0][1]+bbox1[1][1])/2, (bbox1[0][2]+bbox1[1][2])/2 ))   
# size_max = max(size_obj1[0:2])
# scale1 = fov/(3*size_max)         # this makes the largest dimension of the object cover 1/3rd of the image
# obj1.scale = Vector((scale1,scale1,scale1))

Sx = fov
Sy = fov
Lx_a, Ly_a = preprocess_object(obj1, fov)
Lx_b, Ly_b = preprocess_object(obj2, fov)

max_x = min(Sx - Lx_a, Sx - Lx_b)
max_y = min(Sy - Ly_a, Sy - Ly_b)

dist_x = random.uniform(0,max_x)
dist_y = random.uniform(0,max_y)

x1_a = -(Sx/2) + (Lx_a/2) + random.uniform(0, Sx -Lx_a - dist_x)
x2_a = x1_a + dist_x
y1_a = -(Sy/2) + (Ly_a/2) + random.uniform(0, Sy - Ly_a - dist_y)
y2_a = y1_a + dist_y

x1_b = -(Sx/2) + (Lx_b/2) + random.uniform(0, Sx -Lx_b - dist_x)
x2_b = x1_b + dist_x
y1_b = -(Sy/2) + (Ly_b/2) + random.uniform(0, Sy - Ly_b - dist_y)
y2_b = y1_b + dist_y

if random.randint(0,1)==1:
    x1_a, x2_a = x2_a, x1_a
    x1_b, x2_b = x2_b, x1_b

if random.randint(0,1)==1:
    y1_a, y2_a = y2_a, y1_a
    y1_b, y2_b = y2_b, y1_b

P = {'A':{},'B':{}}
P['A']['x'] = np.linspace(x1_a, x2_a, num_frames)
P['A']['y'] = np.linspace(y1_a, y2_a, num_frames)
P['B']['x'] = np.linspace(x1_b, x2_b, num_frames)
P['B']['y'] = np.linspace(y1_b, y2_b, num_frames)

Objs = {'A':obj1, 'B':obj2}
Scenes = {'A':scene1, 'B':scene2}


#bpy.context.scene.cycles.device = 'GPU'

for n in range(N):
    for name in ['A','B']:
        bpy.context.screen.scene = Scenes[name]
        bpy.context.scene.update()
        AutoNode()
        for i in range(num_frames):
            Objs[name].location = Vector((P[name]['x'][i], P[name]['y'][i], 0))
            bpy.context.scene.render.filepath = OUT_PATH + '{:s}_{:02d}.jpg'.format(name,i)
            bpy.ops.render.render(write_still=True)


