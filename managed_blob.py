import os
import hashlib

import bpy
from mathutils import Vector

class ManagedBlob():
    def __init__(self, name, obj, loc, opt, point_to=None):
        self.name = name
        self.object = obj
        self.object.location = loc
        if point_to:
            self.look_at(point_to)

    def look_in(self, direction):
        """ rotates object to point in direction
        # Ref: https://blender.stackexchange.com/questions/5210/pointing-the-camera-in-a-particular-direction-programmatically
        """
        # point the objects '-Z' axis in direction and use its 'Y' as up
        rot_quat = direction.to_track_quat('-Z', 'Y')
        self._object.rotation_euler = rot_quat.to_euler()

    def look_at(self, point):
        """ rotates object to point in direction of point"""
        loc_obj = Vector(self.object.location)
        direction = point - loc_obj
        self.look_in(direction)

class ManagedCamera(ManagedBlob):
    def __init__(self, name, loc, opt, point_to=None):
        self.data = bpy.data.cameras.new(name)
        super(ManagedCamera, self).__init__(name,
            bpy.data.objects.new(name, self.data), loc, opt, point_to=point_to)



class ManagedLamp(ManagedBlob):
    def __init__(self, name, type, opt):
        self.data = bpy.data.lamps.new(name=name, type=type_)
        if self.data.type == 'SUN':
            self.data.sky.use_sky = True
        self.data.energy = lamp_energy
        super(ManagedLamp, self).__init__(name,
            bpy.data.objects.new(name, self.data), loc,  opt, point_to)


class ManagedObject(ManagedBlob):
    def __init__(self, path, opt, axis_forward=None, axis_up=None):
        if axis_forward and axis_up:
            bpy.ops.import_scene.obj(filepath=path,axis_forward=axis_forward,
                axis_up=axis_up, filter_glob="*.OBJ;*.obj")
        elif axis_forward and not axis_up:
            bpy.ops.import_scene.obj(filepath=path,
                axis_forward=axis_forward, filter_glob="*.OBJ;*.obj")
        elif not axis_forward and axis_up:
            bpy.ops.import_scene.obj(filepath=path,
                axis_up=axis_up, filter_glob="*.OBJ;*.obj")
        elif not axis_forward and not axis_up:
            bpy.ops.import_scene.obj(filepath=path, filter_glob="*.OBJ;*.obj")
        print(bpy.context.selected_objects)
        bpy.context.scene.objects.active = bpy.context.selected_objects[0]
        bpy.ops.object.join()
        print(bpy.context.selected_objects)
        print('ok')
        name = name_from_path(path)
        loc = (0,0,0)
        super(ManagedObject, self).__init__(name,
            bpy.context.active_object, loc, opt)

def name_from_path(obj_path):
    """ generate hashed name from path """
    hash_object = hashlib.md5(obj_path.encode())
    return hash_object.hexdigest()



def main():
    camera = ManagedCamera('abc', (1, 2, 3), None)
    print('camera done')
    path = '/media/laurenz/Seagate Backup Plus Drive/ShapeNet/ShapeNetCore.v2/02880940/e072da5c1e38c11a7548281e465c9303/models/model_normalized.obj'
    object_ = ManagedObject(path, None)
    print(object_.object)
    scene = bpy.data.scenes.new('abcdefg')
    scene.objects.link(object_.object)
    bpy.context.screen.scene = bpy.data.scenes['abcdefg']
    bpy.context.screen.scene.update()


if __name__ == "__main__":
    main()
