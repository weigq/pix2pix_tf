#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import random

__all__ = ['Options']


class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self._initial()

    def _initial(self):
        # =====================================================
        #               General Options
        # =====================================================
        self.parser.add_argument("--input_dir",     type=str, default='./dataset/cityscapes/train', help="path to data path")
        self.parser.add_argument("--output_dir",    type=str, default='./dataset/cityscapes/train_out', help="path to save result")
        self.parser.add_argument("--checkpoint",    type=str, default=None)
        self.parser.add_argument("--output_filetype", default="png", choices=["png", "jpeg"])
        self.parser.add_argument("--seed",          type=int, default=None)
        self.parser.add_argument("--lab_colorization", action="store_true", help="split input image into brightness (A) and color (B)")
        # =====================================================
        #               Model Options
        # =====================================================
        self.parser.add_argument("--mode",          type=str, default='train', choices=["train", "test", "export"])
        self.parser.add_argument("--ngf",           type=int, default=64, help="# of G filters in 1st conv layer")
        self.parser.add_argument("--ndf",           type=int, default=64, help="# of D filters in 1st conv layer")
        self.parser.add_argument("--which_direction", type=str, default="AtoB", choices=["AtoB", "BtoA"])
        # =====================================================
        #               Running Options
        # =====================================================
        self.parser.add_argument("--batch_size",    type=int, default=1)
        self.parser.add_argument("--scale_size",    type=int, default=286, help="image size before cropping to 256x256")
        self.parser.add_argument("--aspect_ratio",  type=float, default=1.0, help="ratio of output images(w/h)")
        self.parser.add_argument("--flip",          dest="flip", action="store_true", help="flip images horizontally")
        self.parser.add_argument("--no_flip",       dest="flip", action="store_false", help="nor flip images horizontally")
        self.parser.set_defaults(flip=True)
        self.parser.add_argument("--lr",            type=float, default=0.0002)
        self.parser.add_argument("--beta1",         type=float, default=0.5, help="momentum term of adam")
        self.parser.add_argument("--l1_weight",     type=float, default=100.0, help="weight on L1 loss of G")
        self.parser.add_argument("--gan_weight",    type=float, default=1.0, help="weight on GAN loss of G")
        self.parser.add_argument("--max_steps",     type=int, help="# of training steps")
        self.parser.add_argument("--max_epochs",    type=int, default=100, help="# of training epochs")
        self.parser.add_argument("--summary_freq",  type=int, default=100, help="frequency of updating summary")
        self.parser.add_argument("--progress_freq", type=int, default=50, help="frequency of display progress")
        self.parser.add_argument("--trace_freq",    type=int, default=0, help="frequency of trace execution")
        self.parser.add_argument("--display_freq",  type=int, default=0, help="frequency of saving current training images")
        self.parser.add_argument("--save_freq",     type=int, default=5000, help="frequency of saving model")

    def parse(self):
        self.opt = self.parser.parse_args()
        self.opt.seed = random.randint(0, 2**31 - 1)
        return self.opt
