### Data Generation Code for Motion Transfer


# Make sure that GPU is used while rendering. (1) Enable in user prefs, (2) enable in render properties bpy.context.scene.cycles.device = 'GPU'
# Reference: https://blender.stackexchange.com/questions/7485/enabling-gpu-rendering-for-cycles/7486#7486

##############################
# Execute this script in the background in console
# Reference: https://blender.stackexchange.com/questions/6817/how-to-pass-command-line-arguments-to-a-blender-python-script
# $ blender --background test.blend --python Generate_Dataset.py -- example args 123
##################################

# TODO: uniform naming scheme: camel case or _
# TODO: Commentation methods. cleaning up

# All the essential libraries required are being imported below.
import bpy
from mathutils import Vector
from math import radians, tan
import random
#random.seed(123454)
from bpy.types import Operator
import numpy as np
import os
import timeit
import hashlib

start = timeit.default_timer()

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
lamp_location = Vector((6,6,6))
lamp_rot = Vector((0.0,0.0,0.0))
cameraName1 = 'Camera1'
cameraName2 = 'Camera2'
lampName1 = 'Lamp1'
lampName2 = 'Lamp2'
sceneName1 = 'Scene1'
sceneName2 = 'Scene2'
DATA_PATH = '/media/laurenz/Seagate Backup Plus Drive/ShapeNet/ShapeNetCore.v2/'
#OUT_PATH = '/media/innit/Zone_D/MotionTransfer/DataGeneration/Data/'
OUT_PATH = '/home/laurenz/IITGN/motion_transfer/data_generation/data/'
num_frames = 10
USE_GPU = True
N = 1
same_category = False # if True, A an B from same synset
align_direction_with_movement = True
animation = True
single_frames = False # export rendering as single .png images


synset_name_pairs = [('02691156', 'aeroplane'),
                             ('02834778', 'bicycle'),
                             ('02858304', 'boat'),
                             ('02876657', 'bottle'),
                             ('02924116', 'bus'),
                             ('02958343', 'car'),
                             ('03001627', 'chair'),
                             ('04379243', 'diningtable'),
                             ('03790512', 'motorbike'),
                             ('04256520', 'sofa'),
                             ('04468005', 'train'),
                             ('03211117', 'tvmonitor')]

def name_from_path(obj_path):
    hash_object = hashlib.md5(obj_path.encode())
    return hash_object.hexdigest()


def set_up_world(name, horizon_color= (0.460041, 0.703876, 1), zenith_color = (0.120707, 0.277449, 1)):
    new_world = bpy.data.worlds.new(name)
    new_world.use_sky_paper = True
    new_world.use_sky_blend = True
    new_world.use_sky_real = True
    new_world.horizon_color = horizon_color
    new_world.zenith_color = zenith_color

def addScene(sceneName, engine, filepath, image_size, resolution_percentage, world, use_gpu):
    scene = bpy.data.scenes.new(sceneName)
    scene.render.engine = engine
    scene.render.image_settings.color_mode ='RGBA'
    scene.render.filepath = filepath
    scene.render.resolution_x = image_size
    scene.render.resolution_y = image_size
    scene.render.resolution_percentage = resolution_percentage
    scene.world = bpy.data.worlds.get(world)
    #scene.render.alpha_mode = 'TRANSPARENT'
    scene.render.alpha_mode = 'SKY'
    print(scene.world)
    # add ligth from all directions
    scene.world.light_settings.use_environment_light = True
    scene.world.light_settings.environment_energy = np.random.uniform(0,1)
    scene.world.light_settings.environment_color = 'PLAIN'

    if use_gpu and engine == 'cycles':
        scene.cycles.device = 'GPU'
    return scene

def updateScene(scene, objects, lamps, camera):
    """Set lamps and cameras for given scene

    # Arguments
        scene: String, name of scene
        objects: list of object names
        lamps: list of lamp names to use in scene
        camera: camera name to use in scene
    """
    # select scene
    bpy.context.screen.scene = bpy.data.scenes[scene]

    # delete all objects currently in scene:
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.data.objects:
        obj.select == True
        bpy.ops.object.delete()
    for obj in objects:
        addObject()


# A function to make a camera point towards any point in space
# Reference: https://blender.stackexchange.com/questions/5210/pointing-the-camera-in-a-particular-direction-programmatically
def look_at(obj, point):
    loc_obj = Vector(obj.location)
    direction = point - loc_obj
    rot_quat = direction.to_track_quat('-Z', 'Y')       # point the cameras '-Z' and use its 'Y' as up
    obj.rotation_euler = rot_quat.to_euler()     # assume we're using euler rotation

def look_in(obj, direction):
    rot_quat = direction.to_track_quat('-Z', 'Y')       # point the object '-Z' and use its 'Y' as up
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
    if lamp_type == 'SUN':
        print('sun')
        bpy.data.lamps[lamp_name].sky.use_sky = True
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
    # name object for future Refer
    obj = bpy.context.active_object
    obj.name = name_from_path(obj_path)
    print('Added Object from Path: {}'.format(obj_path))
    return obj

def get_random_rot_euler():
    theta1 = radians(random.uniform(0,360))#random.uniform(0,360)
    theta2 = radians(random.uniform(0,360))
    theta3 = radians(random.uniform(0,360))
    return ((theta1, theta2, theta3))



######## Setting up .blend file ################################################

# sample two objects from ShapeNet dataset
synsets = os.listdir(DATA_PATH)
number_of_synsets = len(synsets)
synset_A = synsets[random.randint(0, number_of_synsets-1)]
if not same_category:
    synset_B = synsets[random.randint(0, number_of_synsets-1)]
else:
    synset_B = synset_A
print('synset_B: {}, synset_B: {}'.format(synset_A, synset_B))
object_A_ids = os.listdir(os.path.join(DATA_PATH, synset_A))
object_B_ids = os.listdir(os.path.join(DATA_PATH, synset_B))
number_of_objects_A = len(object_A_ids)
number_of_objects_B = len(object_B_ids)
obj_A_id = object_A_ids[random.randint(0, number_of_objects_A-1)]
obj_B_id = object_B_ids[random.randint(0, number_of_objects_B-1)]
obj_pathA = os.path.join(DATA_PATH, synset_A, obj_A_id, 'models/model_normalized.obj')
obj_pathB = os.path.join(DATA_PATH, synset_B, obj_B_id, 'models/model_normalized.obj')
paths = {}
paths['A']= obj_pathA
paths['B'] = obj_pathB
names = {}
names['A'] = name_from_path(obj_pathA)
names['B'] = name_from_path(obj_pathB)

# Generate 2 new scenes

set_up_world('world_A')
set_up_world('world_B')
scene1 = addScene(sceneName1, 'BLENDER_RENDER', OUT_PATH + 'A.png', IMAGE_SIZE, 100, 'world_A', USE_GPU)
scene2 = addScene(sceneName2, 'BLENDER_RENDER', OUT_PATH + 'B.png', IMAGE_SIZE, 100, 'world_B', USE_GPU)

# Set scene 1 as screen for loading and processing object A
bpy.context.screen.scene = scene1

camera1 = addCameras(cameraName1, camera_location, camera_rot, True, origin)
lamp1 = addLamps(lampName1, 'POINT', lamp_location, lamp_rot, True, origin)
#obj_pathA = '/home/laurenz/IITGN/motion_transfer/datasets/shapeNet/e480a15c22ee438753388b7ae6bc11aa/models/model_normalized.obj'
#obj_path1 = '/home/laurenz/IITGN/motion_transfer/data/shapeNet/test/test.obj'
obj1 = addObject(obj_pathA, axis_forward='-X', axis_up='Y')
obj1.rotation_euler = get_random_rot_euler()
scene1.update()


bpy.context.screen.scene = scene2

camera2 = addCameras(cameraName2, camera_location, camera_rot, True, origin)
lamp2 = addLamps(lampName2, 'POINT', lamp_location, lamp_rot, True, origin)
#obj_pathB = '/home/laurenz/IITGN/motion_transfer/datasets/shapeNet/1b90541c9d95d65d2b48e2e94b50fd01/models/model_normalized.obj'
obj2 = addObject(obj_pathB, axis_forward='-X', axis_up='Y')
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
    scale = (fov/size_max)*(COVERAGE)         # this makes the largest dimension of the object cover <coverage>*100 % of the image
    obj.scale = Vector((scale,scale,scale))
    size_obj = np.array(size_obj)*scale
    return size_obj[0], size_obj[1]


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

# change start and end with 50% chance
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
opposite = {'A': 'B', 'B': 'A'}


######## Create Animation ######################################################

for n in range(N):
    bpy.context.screen.scene = Scenes['A']
    bpy.context.scene.frame_end = 10*2
    for name in ['A','B']:
        add_frames = 0
        if name == 'B':
            add_frames = 10
        #bpy.context.screen.scene = Scenes[name]
        if name == "B":
            bpy.context.scene.objects.link(Objs[name])
            #bpy.ops.outliner.scene_drop(object = name_from_path(paths[name]), scene = Scenes['A'])
        bpy.context.scene.frame_set(add_frames)
        bpy.data.objects[names[name]].hide_render = False
        bpy.data.objects[names[opposite[name]]].hide_render = True
        print(name)
        for _, obj in Objs.items():
            obj.keyframe_insert(data_path='hide_render')
        bpy.context.scene.update()
        print(name)
        # set up animation
        #bpy.context.scene.frame_end = 10*2
        #bpy.ops.object.select_pattern(pattern=names[name])
        print('Active Object: {}'.format(bpy.context.scene.objects.active))
        #bpy.ops.object.mode_set(mode='OBJECT')
        if align_direction_with_movement:
            direction = Vector((P[name]['x'][1]-P[name]['x'][0], P[name]['y'][1]-P[name]['y'][0], 0))
            look_in(Objs[name], direction)
        if animation:
            for i in range(num_frames):
                bpy.context.scene.frame_set(i+add_frames)
                Objs[name].location = Vector((P[name]['x'][i], P[name]['y'][i], 0))
                Objs[name].keyframe_insert(data_path="location")
                if single_frames:
                    bpy.context.scene.render.filepath = OUT_PATH + '{:s}_{:02d}.jpg'.format(name,i)
                    bpy.ops.render.render(write_still=True)
    if not single_frames:
        bpy.context.scene.render.image_settings.file_format = 'AVI_JPEG'
        bpy.context.scene.render.fps = 6
        bpy.ops.render.render(animation=True)

        if not animation:
            imgs = []
            # code from: https://ammous88.wordpress.com/2015/01/16/blender-access-render-results-pixels-directly-from-python-2/
            # switch on nodes
            bpy.context.scene.use_nodes = True
            tree = bpy.context.scene.node_tree
            links = tree.links

            # clear default nodes
            for n in tree.nodes:
                tree.nodes.remove(n)

            # create input render layer node
            rl = tree.nodes.new('CompositorNodeRLayers')
            rl.location = 185,285

            # create output node
            v = tree.nodes.new('CompositorNodeViewer')
            v.location = 750,210
            v.use_alpha = False

            # Links
            links.new(rl.outputs[0], v.inputs[0])  # link Image output to Viewer input


            for i in range(num_frames):
                Objs[name].location = Vector((P[name]['x'][i], P[name]['y'][i], 0))
                bpy.context.scene.render.filepath = OUT_PATH + '{:s}_{:02d}.jpg'.format(name,i)
                bpy.ops.render.render(write_still=True)
                pixels = bpy.data.images['Viewer Node'].pixels
                pixels = np.array(pixels)
                pixels = np.reshape(pixels, (IMAGE_SIZE, IMAGE_SIZE, -1))
                imgs.append(pixels)

            from PIL import Image

            '''
            # test if imgs contrains correct frames/
            for i in range(num_frames):
                array = imgs[i]*255
                #tmp = array[:, :, 3]
                #array[:, :, 3] = array[:, :, 0]
                #array[:, :, 0] = tmp
                array = array[:, :, :3]
                array = array.astype('uint8')
                img = Image.fromarray(array, 'RGB')
                #background = Image.new("RGB", img.size, (255, 255, 255))
                #background.paste(img, mask=img.split()[3]) # 3 is the alpha channel

                img.save(os.path.join(OUT_PATH, "test_" + str(i) + '.png'), 'PNG')
                #print(array[:,:,:])
                #img.save(os.path.join(OUT_PATH, "test_" + str(i) + '.png'))
            '''
            import sys
            #sys.path.append('/usr/lib/python3.4/tkinter/')
            #sys.path.append('/home/laurenz/anaconda3/lib/python3.6/tkinter/')
            import matplotlib as mpl
            mpl.use('Agg')
            import matplotlib.pyplot as plt
            import subprocess


            dpi = 100
            # create a figure window that is the exact size of the image
            f = plt.figure(frameon=False, figsize=(float(IMAGE_SIZE)/dpi, float(IMAGE_SIZE)/dpi), dpi=dpi)
            canvas_width, canvas_height = f.canvas.get_width_height()
            ax = f.add_axes([0, 0, 1, 1])
            ax.axis('off')

            def update(frame):
                plt.imshow(imgs[frame], interpolation='nearest')

            # Open an ffmpeg process
            outf = os.path.join(OUT_PATH, name +'.mp4' )
            cmdstring = ('avconv',
                '-y', '-r', '10', # overwrite, 30fps
                '-s', '%dx%d' % (canvas_width, canvas_height), # size of image string
                '-pix_fmt', 'argb', # format
                '-f', 'rawvideo',  '-i', '-', # tell ffmpeg to expect raw video from the pipe
                '-vcodec', 'mpeg4',
                '-frames', '10', outf) # output encoding
            p = subprocess.Popen(cmdstring, stdin=subprocess.PIPE)

            # Draw frames and write to the pipe
            for frame in range(num_frames):
                # draw the frame
                update(frame)
                plt.draw()

                # extract the image as an ARGB string
                string = f.canvas.tostring_argb()

                # write to pipe
                p.stdin.write(string)

            # Finish up
            p.communicate()



runtime = timeit.default_timer() - start
print('Runtime: {} s'.format(runtime))
