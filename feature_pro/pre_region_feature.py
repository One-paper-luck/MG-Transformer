from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import argparse
import numpy as np
import torch
import skimage.io
import skimage.transform
from torchvision import transforms as trn
from misc.resnet_utils import myResnet
import misc.resnet as resnet

preprocess = trn.Compose([
    # trn.ToTensor(),
    trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def main(params):
    net = getattr(resnet, params['model'])()
    net.load_state_dict(torch.load(os.path.join(params['model_root'], params['model'] + '.pth')))
    my_resnet = myResnet(net)
    my_resnet.cuda()
    my_resnet.eval()

    imgs = json.load(open(params['input_json'], 'r'))
    imgs = imgs['images']
    N = len(imgs)

    for i, img in enumerate(imgs):
        I = skimage.io.imread(os.path.join(params['images_root'], img['filename']))  # sydney 500 500 3
        I = skimage.transform.resize(I, (224, 224))

        if len(I.shape) == 2:
            I = I[:, :, np.newaxis]
            I = np.concatenate((I, I, I), axis=2)

        I = I.astype('float32') / 255.0
        I = torch.from_numpy(I.transpose([2, 0, 1])).cuda()
        I = preprocess(I)

        with torch.no_grad():
            conv5= my_resnet(I,params['att_size'])

        # np.save(os.path.join(params['output_dir'], str(img['filename'])), conv5.data.cpu().float().numpy())

        if i % 1000 == 0:
            print('processing %d/%d (%.2f%% done)' % (i, N, i * 100.0 / N))
    print('wrote ', params['output_dir'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_json', default='/media/dmd/ours/mlw/rs/Sydney_Captions/dataset.json')
    # parser.add_argument('--output_dir', default='/media/dmd/ours/mlw/rs/multi_scale_T/sydney_res152_con5_500')
    parser.add_argument('--output_dir', default='/media/dmd/ours/mlw/rs/multi_scale_T/sydney_name')
    parser.add_argument('--images_root', default='/media/dmd/ours/mlw/rs/Sydney_Captions/imgs',
                        help='root location in which images are stored, to be prepended to file_path in input json')

    parser.add_argument('--att_size', default=14, type=int, help='14x14 or 7x7')
    parser.add_argument('--model', default='resnet152', type=str, help='resnet50,resnet101, resnet152')
    parser.add_argument('--model_root', default='/media/dmd/ours/mlw/pre_model', type=str,
                        help='model root')

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent=2))
    main(params)
