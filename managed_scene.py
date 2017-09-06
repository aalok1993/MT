import bpy
import os

class ManagedScene():

    def __init__(self, scene_name, opt):
            self.scene = bpy.data.scenes.new(scene_name)
            self.scene.render.engine = opt.engine
            self.scene.render.image_settings.color_mode = opt.color_mode
            self.scene.render.filepath = os.path.join(opt.outroot + scene_name)
            self.scene.render.resolution_x = opt.image_size
            self.scene.render.resolution_y = opt.image_size
            self.scene.render.resolution_percentage = opt.resolution_percentage
            if opt.engine == 'CYCLES' and gpu:
                self.scene.cycles.devices = 'GPU'

    def set_world(name, horizon_color= (0.460041, 0.703876, 1),
        zenith_color = (0.120707, 0.277449, 1)):
        world = bpy.data.worlds.new(name)
        world.use_sky_paper = True
        world.use_sky_blend = True
        world.use_sky_real = True
        world.horizon_color = horizon_color
        world.zenith_color = zenith_color

    def add_blob(blob):
        pass
