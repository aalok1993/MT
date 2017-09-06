### Data Generation Code for Motion Transfer


# Make sure that GPU is used while rendering. (1) Enable in user prefs, (2) enable in render properties bpy.context.scene.cycles.device = 'GPU'
# Reference: https://blender.stackexchange.com/questions/7485/enabling-gpu-rendering-for-cycles/7486#7486

# Execute this script in the background in console
# Reference: https://blender.stackexchange.com/questions/6817/how-to-pass-command-line-arguments-to-a-blender-python-script
# $ blender --background test.blend --python Generate_Dataset.py -- example args 123

# TODO: uniform naming scheme: camel case or _
# TODO: Commentation methods. cleaning up

import timeit
start = timeit.default_timer()

import sys
import os
import numpy as np
from math import radians, tan
import random
#random.seed(123454)
import hashlib

import bpy
from mathutils import Vector
from bpy.types import Operator

# to import own modules
dir = os.path.dirname(bpy.data.filepath)
if not dir in sys.path:
    sys.path.append(dir)

from gen_options import GenOptions

# parse arguments
try:
    args = sys.argv[sys.argv.index('--') + 1:]
except ValueError:
    args = []
opt = GenOptions().parse(args)


# Define all the functions, parameters, options that are
# going to be used in the code.

# Variables
# MAX_COV = 0.4
# MIN_COV = 0.2
COVERAGE = 1.0/3
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

N = 1
same_category = False # if True, A an B from same synset
align_direction_with_movement = True
animation = True
single_frames = False # export rendering as single .png images
two_vids = False


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


def preprocess_object(obj,fov):
    Bx = np.array([(obj.matrix_world * v.co) for v in obj.data.vertices])
    bbox = [ [np.min(Bx[:,0]), np.min(Bx[:,1]), np.min(Bx[:,2])], [np.max(Bx[:,0]), np.max(Bx[:,1]), np.max(Bx[:,2])] ]
    size_obj = [ bbox[1][0]-bbox[0][0], bbox[1][1]-bbox[0][1], bbox[1][2]-bbox[0][2] ]
    #obj.location = obj.location - Vector(( (bbox[0][0]+bbox[1][0])/2, (bbox[0][1]+bbox[1][1])/2, (bbox[0][2]+bbox[1][2])/2 ))
    size_max = max(size_obj[0:2])
    scale = (fov/size_max)*(COVERAGE)         # this makes the largest dimension of the object cover <coverage>*100 % of the image
    obj.scale = Vector((scale,scale,scale))
    obj.rotation_euler = get_random_rot_euler()
    size_obj = np.array(size_obj)*scale
    return size_obj[0], size_obj[1]


def addScene(sceneName, world=None):
    scene = bpy.data.scenes.new(sceneName)
    scene.render.engine = opt.engine
    scene.render.image_settings.color_mode = opt.color_mode
    scene.render.filepath = os.path.join(opt.outroot + sceneName)
    scene.render.resolution_x = opt.image_size
    scene.render.resolution_y = opt.image_size
    scene.render.resolution_percentage = opt.resolution_percentage
    if world:
        scene.world = bpy.data.worlds.get(world)
        scene.render.alpha_mode = 'SKY'
        # add ligth from all directions
        scene.world.light_settings.use_environment_light = True
        scene.world.light_settings.environment_energy = np.random.uniform(0,1)
        scene.world.light_settings.environment_color = 'PLAIN'

    if use_gpu and engine == 'CYCLES':
        scene.cycles.device = 'GPU'
    return scene

def updateScene(scene_name, object_paths, lamps, camera = None):
    """Set lamps and cameras for given scene

    # Arguments
        scene: String, name of scene
        objects: list of object pathes
        lamps: list of dictionary with lamp properties
        camera: dictionary with camera properties
    """
    # select scene
    bpy.context.screen.scene = bpy.data.scenes[scene_name]

    # delete all objects currently in scene:
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.data.objects:
        obj.select == True
        bpy.ops.object.delete()
    objs = {}
    for n, obj_path in enumerate(object_paths):
        key = 'obj{}'.format(n)
        objs[key] = addObject(obj_path, axis_forward='-X', axis_up='Y')

    for lamp in lamps:
        addLamps(lamp['name'], lamp['type'], lamp['loc'], lamp['rot'],
            lamp['bIsPoint'], lamp['pointTo'])

    if camera:
        addCameras(camera['name'], camera['loc'], camera['rot'],
            camera['bIsPoint'], camera['pointTo'])

    bpy.context.screen.scene.update()

    if 'obj0' in objs:
        return objs['obj0']


######## Setting up Scenes and Objects #########################################

# sample two objects from ShapeNet dataset
synsets = os.listdir(opt.dataroot)
number_of_synsets = len(synsets)
synset_A = synsets[random.randint(0, number_of_synsets-1)]
if not same_category:
    synset_B = synsets[random.randint(0, number_of_synsets-1)]
else:
    synset_B = synset_A
print('synset_A: {}, synset_B: {}'.format(synset_A, synset_B))
object_A_ids = os.listdir(os.path.join(opt.dataroot, synset_A))
object_B_ids = os.listdir(os.path.join(opt.dataroot, synset_B))
number_of_objects_A = len(object_A_ids)
number_of_objects_B = len(object_B_ids)
obj_A_id = object_A_ids[random.randint(0, number_of_objects_A-1)]
obj_B_id = object_B_ids[random.randint(0, number_of_objects_B-1)]
obj_pathA = os.path.join(opt.dataroot, synset_A, obj_A_id, 'models/model_normalized.obj')
obj_pathB = os.path.join(opt.dataroot, synset_B, obj_B_id, 'models/model_normalized.obj')
paths = {}
paths['A']= obj_pathA
paths['B'] = obj_pathB
names = {}
names['A'] = name_from_path(obj_pathA)
names['B'] = name_from_path(obj_pathB)


if two_vids:
    # setup cameras and lamps
    cameraA = {'name': cameraName1,
            'loc': camera_location,
            'rot':  camera_rot,
            'bIsPoint': True,
            'pointTo': origin
    }
    cameraB = {'name': cameraName2,
            'loc': camera_location,
            'rot':  camera_rot,
            'bIsPoint': True,
            'pointTo': origin
    }

    lampsA = [
        {'name': lampName1,
            'type': 'POINT',
            'loc': lamp_location,
            'rot': lamp_rot,
            'bIsPoint': True,
            'pointTo': origin
        }
    ]
    lampsB = [
        {'name': lampName1,
            'type': 'POINT',
            'loc': lamp_location,
            'rot': lamp_rot,
            'bIsPoint': True,
            'pointTo': origin
        }
    ]

    set_up_world('world_A')
    set_up_world('world_B')
    scene1 = addScene(sceneName1, 'world_A')
    scene2 = addScene(sceneName2, 'world_B')
    camera_scene = scene1

    obj1 = updateScene(sceneName1, [obj_pathA], lampsA, cameraA)
    obj2 = updateScene(sceneName2, [obj_pathB], lampsB, cameraB)

if not two_vids:
    cameraT = {'name': 'target_camera',
            'loc': camera_location,
            'rot':  camera_rot,
            'bIsPoint': True,
            'pointTo': origin
    }

    lampsT = [
        {'name': 'target_lamp',
            'type': 'POINT',
            'loc': lamp_location,
            'rot': lamp_rot,
            'bIsPoint': True,
            'pointTo': origin
        }
    ]

    set_up_world('target')

    scene1 = addScene(sceneName1)
    scene2 = addScene(sceneName2)
    target_scene = addScene('AB', world = 'target')
    camera_scene = target_scene

    obj1 = updateScene(sceneName1, [obj_pathA], [])
    obj2 = updateScene(sceneName2, [obj_pathB], [])
    updateScene('target', [], lampsT, cameraT)

Scenes = {'A':scene1, 'B':scene2, 'target': target_scene}
Objs = {'A':obj1, 'B':obj2}
opposite = {'A': 'B', 'B': 'A'}

######## Calculate Trajectory ##################################################

# Refer this for FOV length http://www.scantips.com/lights/subjectdistance.html
fov = (CAM_DISTANCE * tan(camera_scene.camera.data.angle/2))*2
# fov_in_meters = 9.14

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
P['A']['x'] = np.linspace(x1_a, x2_a, opt.number_of_frames)
P['A']['y'] = np.linspace(y1_a, y2_a, opt.number_of_frames)
P['B']['x'] = np.linspace(x1_b, x2_b, opt.number_of_frames)
P['B']['y'] = np.linspace(y1_b, y2_b, opt.number_of_frames)

######## Create Animation ######################################################

for n in range(N):
    if not two_vids:
        bpy.context.screen.scene = Scenes['target']
        bpy.context.scene.frame_start = 0
        bpy.context.scene.frame_end = opt.number_of_frames*2 -1
    for name in ['A','B']:
        add_frames = 0
        if name == 'B':
            add_frames = 10

        if two_vids:
            bpy.context.screen.scene = Scenes[name]
            bpy.context.scene.frame_start = 0
            bpy.context.scene.frame_end = opt.number_of_frames -1
        if not two_vids:
            bpy.context.scene.objects.link(Objs[name])
            bpy.context.scene.frame_set(add_frames)
            bpy.data.objects[names[name]].hide_render = False
            bpy.data.objects[names[opposite[name]]].hide_render = True
            for _, obj in Objs.items():
                obj.keyframe_insert(data_path='hide_render')
            bpy.context.scene.update()

        # set up animation
        bpy.ops.object.select_pattern(pattern=names[name])
        print('Active Object: {}'.format(bpy.context.scene.objects.active))

        if align_direction_with_movement:
            direction = Vector((P[name]['x'][1]-P[name]['x'][0], P[name]['y'][1]-P[name]['y'][0], 0))
            look_in(Objs[name], direction)

        if animation:
            for i in range(opt.number_of_frames):
                bpy.context.scene.frame_set(i+add_frames)
                Objs[name].location = Vector((P[name]['x'][i], P[name]['y'][i], 0))
                Objs[name].keyframe_insert(data_path="location")
                if single_frames:
                    bpy.context.scene.render.filepath = opt.outroot + '{:s}_{:02d}.jpg'.format(name,i)
                    bpy.ops.render.render(write_still=True)

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


            for i in range(opt.number_of_frames):
                Objs[name].location = Vector((P[name]['x'][i], P[name]['y'][i], 0))
                bpy.context.scene.render.filepath = opt.outroot + '{:s}_{:02d}.jpg'.format(name,i)
                bpy.ops.render.render(write_still=True)
                pixels = bpy.data.images['Viewer Node'].pixels
                pixels = np.array(pixels)
                pixels = np.reshape(pixels, (opt.image_size, opt.image_size, -1))
                imgs.append(pixels)

            from PIL import Image

            '''
            # test if imgs contrains correct frames/
            for i in range(opt.number_of_frames):
                array = imgs[i]*255
                #tmp = array[:, :, 3]
                #array[:, :, 3] = array[:, :, 0]
                #array[:, :, 0] = tmp
                array = array[:, :, :3]
                array = array.astype('uint8')
                img = Image.fromarray(array, 'RGB')
                #background = Image.new("RGB", img.size, (255, 255, 255))
                #background.paste(img, mask=img.split()[3]) # 3 is the alpha channel

                img.save(os.path.join(opt.outroot, "test_" + str(i) + '.png'), 'PNG')
                #print(array[:,:,:])
                #img.save(os.path.join(opt.outroot, "test_" + str(i) + '.png'))
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
            f = plt.figure(frameon=False, figsize=(float(opt.image_size)/dpi, float(opt.image_size)/dpi), dpi=dpi)
            canvas_width, canvas_height = f.canvas.get_width_height()
            ax = f.add_axes([0, 0, 1, 1])
            ax.axis('off')

            def update(frame):
                plt.imshow(imgs[frame], interpolation='nearest')

            # Open an ffmpeg process
            outf = os.path.join(opt.outroot, name +'.mp4' )
            cmdstring = ('avconv',
                '-y', '-r', '10', # overwrite, 30fps
                '-s', '%dx%d' % (canvas_width, canvas_height), # size of image string
                '-pix_fmt', 'argb', # format
                '-f', 'rawvideo',  '-i', '-', # tell ffmpeg to expect raw video from the pipe
                '-vcodec', 'mpeg4',
                '-frames', '10', outf) # output encoding
            p = subprocess.Popen(cmdstring, stdin=subprocess.PIPE)

            # Draw frames and write to the pipe
            for frame in range(opt.number_of_frames):
                # draw the frame
                update(frame)
                plt.draw()

                # extract the image as an ARGB string
                string = f.canvas.tostring_argb()

                # write to pipe
                p.stdin.write(string)

            # Finish up
            p.communicate()

        if not single_frames and two_vids:
            if name == 'B':
                bpy.context.scene.frame_start = add_frames
                bpy.context.scene.frame_end = opt.number_of_frames*2 -1
            bpy.context.scene.render.image_settings.file_format = 'AVI_JPEG'
            bpy.context.scene.render.fps = 6
            bpy.ops.render.render(animation=True)
    if not single_frames and not two_vids:
        bpy.context.scene.render.image_settings.file_format = 'AVI_JPEG'
        bpy.context.scene.render.fps = 6
        bpy.ops.render.render(animation=True)

runtime = timeit.default_timer() - start
print('Runtime: {} s'.format(runtime))
