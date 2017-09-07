import argparse

class GenOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # input properties
        self.parser.add_argument('--dataroot', required = True,
            help='path to ShapeNetCore.v2 directory')
        # output properties
        self.parser.add_argument('--number_of_vids', type=int, default=1,
            help='number of videos to be generated')
        self.parser.add_argument('--image_size', type=int, default = 256,
            help='width and height of output frames')
        self.parser.add_argument('--number_of_frames', type=int, default = 10,
            help= 'number of frames per object')
        self.parser.add_argument('--resolution_percentage', type=int,
            default=100,
            help='actual image size is resolution_percentage/100 * img_size')
        self.parser.add_argument('--outroot', required = True,
            help='output directory for rendered files')
        self.parser.add_argument('--color_mode', default = 'RGBA',
            choices=['RGBA', 'RGB'], help='color mode')
        # render properties
        self.parser.add_argument('--engine', default='BLENDER_RENDER',
            choices=['BLENDER_RENDER', 'CYCLES'],
            help='engine used for rendering')
        self.parser.add_argument('--gpu', action='store_true', default=False,
            help='if True, render on GPU')

    def parse(self, args):
        self.opt, _ = self.parser.parse_known_args(args)

        args = vars(self.opt)
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('{}: {}'.format(str(k), str(v)))
        print('-------------- End ----------------')

        return self.opt
