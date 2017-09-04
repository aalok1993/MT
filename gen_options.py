import argparse

class GenOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # input properties
        self.parser.add_argument('--dataroot', required = True, help='path to ShapeNetCore.v2 directory')
        # output properties
        self.parser.add_argument('--img_size', type=int, default = 256, help='width and height of output frames')
        self.parser.add_argument('--number_of_frames', type=int, default = 10, help= 'number of frames per object')
        # render properties
        self.parser.add_argument('--render_engine', default='BLENDER_RENDER', choices=['BLENDER_RENDER', 'CYCLES'], help='engine used for rendering')

    def parse(self, args):
        self.opt, _ = self.parser.parse_known_args(args)

        args = vars(self.opt)
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('{}: {}'.format(str(k), str(v)))
        print('-------------- End ----------------')

        return self.opt
