

import argparse
import subprocess


p = subprocess.Popen(['python',
                      'imagenet.py',
                      '-a mobilenetv2',
                      '-d /media/xavier/data/train_data/ImageNet',
                      '--weight ./pretrained/mobilenetv2_0.25-b61d2159.pth',
                      '--width-mult 0.25',
                      '--input-size 224',
                      '-e',
                      '--status test'])