### Data Generation Code for Motion Transfer

# Make sure that GPU is used while rendering. (1) Enable in user prefs, (2) enable in render properties bpy.context.scene.cycles.device = 'GPU'
# Reference: https://blender.stackexchange.com/questions/7485/enabling-gpu-rendering-for-cycles/7486#7486

# Execute this script in the background in console
# Reference: https://blender.stackexchange.com/questions/6817/how-to-pass-command-line-arguments-to-a-blender-python-script
# $ blender --background test.blend --python Generate_Dataset.py -- example args 123

# TODO: uniform naming scheme: camel case or _

import timeit
start = timeit.default_timer()

import sys
import os
import numpy as np
from math import radians, tan
import random
random.seed(98)
import hashlib
import math

import bpy
from bpy.types import Operator
from mathutils import Vector
from mathutils import Color

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

coverage = 1.0/3.0
CAM_DISTANCE = 10
origin = Vector((0,0,0))
camera_location = Vector((0,0,CAM_DISTANCE))
camera_rot = Vector((0.0,0.0,0.0))
lamp_location = Vector((6,6,6))
lamp_rot = Vector((0.0,0.0,0.0))
lamp_energy = 2.0
cameraName1 = 'Camera1'
cameraName2 = 'Camera2'
lampName1 = 'Lamp1'
lampName2 = 'Lamp2'
sceneName1 = 'A'
sceneName2 = 'B'

alpha_sat = 2.5
beta_sat = 0.15
alpha_val = 1.5
beta_val = 0.1

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

def name_from_string(str_):
    """ generate hashed name from path """
    hash_object = hashlib.md5(str_.encode())
    return hash_object.hexdigest()

def look_in(obj, direction):
    """ rotates object to point in direction

    # Ref: https://blender.stackexchange.com/questions/5210/pointing-the-camera-in-a-particular-direction-programmatically
    """
    # point the objects '-Z' axis in direction and use its 'Y' as up
    rot_quat = direction.to_track_quat('-Z', 'Y')
    obj.rotation_euler = rot_quat.to_euler()

def look_at(obj, point):
    """ rotates object to point in direction of point"""
    loc_obj = Vector(obj.location)
    direction = point - loc_obj
    look_in(obj, direction)

def get_random_rot_euler():
    """ returns three random euler angles"""
    theta1 = radians(random.uniform(0,360))#random.uniform(0,360)
    theta2 = radians(random.uniform(0,360))
    theta3 = radians(random.uniform(0,360))
    return ((theta1, theta2, theta3))

def get_random_sky_colors(alpha_sat, beta_sat, alpha_val, beta_val):
    """Returns random colors for the sky

    # Arguments:
        alpha_sat, beta_sat: parameters of gamma distribution for saturation
        alpha_val, beta_val: parameters of gamma distribution for value


        E[x]=alpha*beta
        V[x]=alpha*beta^2
    """

    horizon_color = Color()
    h_hue = random.uniform(0, 1)
    h_sat = min(random.gammavariate(alpha_sat, beta_sat), 1)
    h_val = max(1 - random.gammavariate(alpha_val, beta_val), 0)
    horizon_color.hsv = h_hue, h_sat, h_val

    zenith_color = Color()
    z_hue = random.uniform(0, 1)
    z_sat = min(random.gammavariate(alpha_sat, beta_sat), 1)
    z_val = max(1 - random.gammavariate(alpha_val, beta_val), 0)
    zenith_color.hsv = z_hue, z_sat, z_val

    return horizon_color, zenith_color

def set_up_world(name, horizon_color= (0.460041, 0.703876, 1),
    zenith_color = (0.120707, 0.277449, 1)):
    """ create world object """
    new_world = bpy.data.worlds.new(name)
    new_world.use_sky_paper = True
    new_world.use_sky_blend = True
    new_world.use_sky_real = True
    new_world.horizon_color = horizon_color
    new_world.zenith_color = zenith_color
    # add ligth from all directions
    new_world.light_settings.use_environment_light = True
    new_world.light_settings.environment_energy = np.random.uniform(0,1)
    new_world.light_settings.environment_color = 'PLAIN'

def addCameras(camera_name, loc_point, point_to = None, loc_rot=None):
    """Add a camera to currently selected scene """
    camera_data = bpy.data.cameras.new(camera_name)
    camera = bpy.data.objects.new(camera_name, camera_data)
    camera.location = loc_point
    if point_to:
        look_at(camera, point_to)
    elif loc_rot:
        camera.rotation_euler = (loc_rot[0]*constRadToDeg,loc_rot[1]*constRadToDeg, loc_rot[2]*constRadToDeg)
    bpy.context.scene.objects.link(camera)
    bpy.context.scene.camera = camera
    return camera

def addLamps(lamp_name, lamp_type, lamp_energy, loc_point, point_to = None,
    loc_rot = None):
    """Add a lamp to currently selected scene """
    lamp_data = bpy.data.lamps.new(name=lamp_name, type=lamp_type)
    lamp_data.type = lamp_type
    if lamp_data.type == 'SUN':
        lamp_data.sky.use_sky = True
    lamp_data.energy = lamp_energy
    lamp = bpy.data.objects.new(name=lamp_name, object_data=lamp_data)
    lamp.location = loc_point
    if point_to:
        look_at(lamp, point_to)
    elif loc_rot:
        lamp.rotation_euler = (loc_rot[0]*constRadToDeg,loc_rot[1]*constRadToDeg, loc_rot[2]*constRadToDeg)
    bpy.context.scene.objects.link(lamp)
    return lamp

def addObject(obj_path,axis_forward=None,axis_up=None):
    """Load Object from path to current scene
    and align according to given axes]
    """
    prior_objects = [object.name for object in bpy.context.scene.objects]
    if axis_forward and axis_up:
        bpy.ops.import_scene.obj(filepath=obj_path, axis_forward=axis_forward,
            axis_up=axis_up, filter_glob="*.OBJ;*.obj")
    elif axis_forward and not axis_up:
        bpy.ops.import_scene.obj(filepath=obj_path, axis_forward=axis_forward,
            filter_glob="*.OBJ;*.obj")
    elif not axis_forward and axis_up:
        bpy.ops.import_scene.obj(filepath=obj_path, axis_up=axis_up,
            filter_glob="*.OBJ;*.obj")
    elif not axis_forward and not axis_up:
        bpy.ops.import_scene.obj(filepath=obj_path, filter_glob="*.OBJ;*.obj")

    # join parts of object
    new_current_objects = [object.name for object in bpy.context.scene.objects]
    new_objects = list(set(new_current_objects)-set(prior_objects))
    bpy.context.scene.objects.active = bpy.data.objects[new_objects[0]]
    for obj in new_objects:
        bpy.data.objects[obj].select = True
    bpy.ops.object.join()

    # set  origin to geometric center of bounding box
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')

    # name object for future Refer
    obj = bpy.context.active_object
    obj.name = name_from_string(obj_path)
    #print('Added Object from Path: {}'.format(obj_path))
    return obj

def preprocess_object(obj,cam_distance):
    """Ajust size of object to fill coverage of frame with longest side"""
    obj.rotation_euler = get_random_rot_euler()
    Bx = np.array([(obj.matrix_world * v.co) for v in obj.data.vertices])
    print(Bx.shape)
    bbox = [ [np.min(Bx[:,0]), np.min(Bx[:,1]), np.min(Bx[:,2])], [np.max(Bx[:,0]), np.max(Bx[:,1]), np.max(Bx[:,2])] ]
    size_obj = [ bbox[1][0]-bbox[0][0], bbox[1][1]-bbox[0][1], bbox[1][2]-bbox[0][2]]
    size_max = np.linalg.norm(size_obj[0:3])
    fov = cam_distance * tan(camera_scene.camera.data.angle/2)*2
    # fov at upper point of object
    fov_top = (cam_distance-size_obj[2]) * tan(camera_scene.camera.data.angle/2)*2
    # make the largest dim (x,y plane) of the object cover <coverage>*100 % of the image
    scale = (fov/size_max)*(coverage)
    print(size_obj)
    obj.scale = Vector((scale,scale,scale))
    size_obj = np.array(size_obj)*scale
    print(size_obj)
    return size_obj[0], size_obj[1], fov_top


def addScene(sceneName, world=None, file_name=None):
    if not file_name:
        file_name = sceneName
    scene = bpy.data.scenes.new(sceneName)
    scene.render.engine = opt.engine
    scene.render.image_settings.color_mode = opt.color_mode
    scene.render.filepath = os.path.join(opt.outroot + file_name)
    scene.render.resolution_x = opt.image_size
    scene.render.resolution_y = opt.image_size
    scene.render.resolution_percentage = opt.resolution_percentage
    if world:
        scene.world = bpy.data.worlds.get(world)
        scene.render.alpha_mode = 'SKY'

    if opt.gpu and opt.engine == 'CYCLES':
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
    objs = []
    for n, obj_path in enumerate(object_paths):
        objs.append(addObject(obj_path, axis_forward='-X', axis_up='Y'))

    for lamp in lamps:
        addLamps(lamp['name'], lamp['type'], lamp['energy'], lamp['loc'],
            point_to= lamp['point_to'], loc_rot=lamp['rot'])

    if camera:
        addCameras(camera['name'], camera['loc'], point_to = camera['point_to'],
            loc_rot = camera['rot'])

    bpy.context.screen.scene.update()
    return objs

def delete_scene_objects(scene=None):
    """Delete a scene and all its objects.
    # Ref: https://blender.stackexchange.com/questions/75228/how-to-delete-a-scene-with-its-content
    """
    if scene is None:
        # Not specified: it's the current scene.
        scene = bpy.context.screen.scene
    else:
        if isinstance(scene, str):
            # Specified by name: get the scene object.
            scene = bpy.data.scenes[scene]
        # Otherwise, assume it's a scene object already.

    # Remove objects.
    for object_ in scene.objects:
        bpy.data.objects.remove(object_, True)

    # Remove world
    if scene.world:
        bpy.data.worlds.remove(scene.world, True)

    # Remove scene.
    bpy.data.scenes.remove(scene, True)



######## Setting up Cameras and Lamps ##########################################

cameraA = {'name': cameraName1,
        'loc': camera_location,
        'point_to': origin,
        'rot': None
}
cameraB = {'name': cameraName2,
        'loc': camera_location,
        'point_to': origin,
        'rot': None
}

lampsA = [
    {'name': lampName1,
        'type': 'POINT',
        'energy': lamp_energy,
        'loc': lamp_location,
        'point_to': origin,
        'rot': None
    }
]
lampsB = [
    {'name': lampName1,
        'type': 'POINT',
        'energy': lamp_energy,
        'loc': lamp_location,
        'point_to': origin,
        'rot': None
    }
]

cameraT = {'name': 'target_camera',
        'loc': camera_location,
        'point_to': origin,
        'rot': None
}

lampsT = [
    {'name': 'target_lamp',
        'type': 'POINT',
        'energy': lamp_energy,
        'loc': lamp_location,
        'point_to': origin,
        'rot': None
    }
]

######## Sample Objects ans set up Scenes ######################################
synsets = os.listdir(opt.dataroot)
number_of_synsets = len(synsets)

for n in range(opt.number_of_vids):
    synset_A = synsets[random.randint(0, number_of_synsets-1)]
    if not opt.same_category:
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
    obj_pathA = os.path.join(opt.dataroot, synset_A, obj_A_id,
                            'models/model_normalized.obj')
    obj_pathB = os.path.join(opt.dataroot, synset_B, obj_B_id,
                            'models/model_normalized.obj')
    paths = {}
    paths['A']= obj_pathA
    paths['B'] = obj_pathB
    names = {}
    names['A'] = name_from_string(obj_pathA)
    names['B'] = name_from_string(obj_pathB)
    names['AB'] = name_from_string(obj_pathA+obj_pathB)

    colors_A = get_random_sky_colors(alpha_sat, beta_sat, alpha_val,
                                     beta_val)

    colors_B = get_random_sky_colors(alpha_sat, beta_sat, alpha_val,
                                     beta_val)
    print('######## COLORS: {} ######### {}'.format(colors_A, colors_B))
    set_up_world('A', horizon_color=colors_A[0],
                 zenith_color=colors_A[0])
    set_up_world('B', horizon_color=colors_B[0],
                 zenith_color=colors_B[0])


    if opt.two_vids:
        scene1 = addScene(sceneName1, world='A', file_name=names['A'])
        scene2 = addScene(sceneName2, world='world_B', file_name=names['B'])
        camera_scene = scene1

        obj1 = updateScene(sceneName1, [obj_pathA], lampsA, cameraA)[0]
        obj2 = updateScene(sceneName2, [obj_pathB], lampsB, cameraB)[0]
        Scenes = {'A':scene1, 'B':scene2}

    if not opt.two_vids:
        scene1 = addScene('A')
        scene2 = addScene('B')
        target_scene = addScene('AB', world = 'A', file_name=names['AB'])
        camera_scene = target_scene
        print('CAMERA SCENE: {}'.format(camera_scene))

        obj1 = updateScene(sceneName1, [obj_pathA], [])[0]
        obj2 = updateScene(sceneName2, [obj_pathB], [])[0]
        updateScene('AB', [], lampsT, cameraT)
        Scenes = {'A':scene1, 'B':scene2, 'AB': target_scene}

    Objs = {'A':obj1, 'B':obj2}
    opposite = {'A': 'B', 'B': 'A'}

    ######## Calculate Trajectory ##############################################

    Lx_a, Ly_a, fov_A = preprocess_object(obj1, CAM_DISTANCE)
    Lx_b, Ly_b, fov_B = preprocess_object(obj2, CAM_DISTANCE)

    Sx = min(fov_A, fov_B)
    Sy = min(fov_A, fov_B)

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

    ######## Create Animation ##################################################

    if not opt.two_vids:
        bpy.context.screen.scene = Scenes['AB']
        bpy.context.scene.frame_start = 0
        bpy.context.scene.frame_end = opt.number_of_frames*2 -1
    for name in ['A','B']:
        add_frames = 0
        if name == 'B':
            add_frames = 10

        if opt.two_vids:
            bpy.context.screen.scene = Scenes[name]
            bpy.context.scene.frame_start = 0
            bpy.context.scene.frame_end = opt.number_of_frames -1
        if not opt.two_vids:
            bpy.context.scene.objects.link(Objs[name])
            bpy.context.scene.frame_set(add_frames)
            bpy.data.objects[names[name]].hide_render = False
            bpy.data.objects[names[opposite[name]]].hide_render = True
            for _, obj in Objs.items():
                obj.keyframe_insert(data_path='hide_render')
            bpy.context.scene.update()

        # TODO: Does not work if not opt.two_vids, try this solution
        # https://blender.stackexchange.com/questions/7321/workaround-to-keyframe-the-world-setting
        bpy.context.screen.scene.world = bpy.data.worlds.get(name)

        # set up animation
        bpy.ops.object.select_pattern(pattern=names[name])
        #print('Active Object: {}'.format(bpy.context.scene.objects.active))

        if opt.align_obj_with_motion:
            direction = Vector((P[name]['x'][1]-P[name]['x'][0],
                                P[name]['y'][1]-P[name]['y'][0], 0))
            look_in(Objs[name], direction)

        for i in range(opt.number_of_frames):
            bpy.context.scene.frame_set(i+add_frames)
            Objs[name].location = Vector((P[name]['x'][i], P[name]['y'][i], 0))
            Objs[name].keyframe_insert(data_path="location")
            if opt.single_frames:
                bpy.context.scene.render.filepath = (opt.outroot +
                    '{:s}_{:02d}.jpg'.format(name,i))
                bpy.ops.render.render(write_still=True)

        if not opt.single_frames and opt.two_vids:
            if name == 'B':
                bpy.context.scene.frame_start = add_frames
                bpy.context.scene.frame_end = opt.number_of_frames*2 -1
            bpy.context.scene.render.image_settings.file_format = 'AVI_JPEG'
            bpy.context.scene.render.fps = 6
            bpy.ops.render.render(animation=True)

    if not opt.single_frames and not opt.two_vids:
        bpy.context.scene.render.image_settings.file_format = 'AVI_JPEG'
        bpy.context.scene.render.fps = 6
        bpy.ops.render.render(animation=True)

    # delete scenes for next run
    if opt.number_of_vids > 1:
        for key, scene in Scenes.items():
            delete_scene_objects(scene)

runtime = timeit.default_timer() - start
print('Runtime: {} s'.format(runtime))
