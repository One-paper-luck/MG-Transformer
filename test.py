import random
from data import ImageDetectionsField, TextField, RawField
from data import DataLoader, Sydney, UCM, RSICD
import evaluation
from models.transformer import Transformer, MemoryAugmentedEncoder, MeshedDecoder, GlobalGroupingAttention_with_DC
import torch
from tqdm import tqdm
import argparse, os, pickle
import numpy as np
import warnings
warnings.filterwarnings("ignore")



def seed_torch(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_torch()


def predict_captions(model, dataloader, text_field):
    import itertools
    model.eval()
    gen = {}
    gts = {}
    with tqdm(desc='Evaluation', unit='it', total=len(dataloader)) as pbar:
        for it, (images, caps_gt) in enumerate(dataloader):
            detections = images[0].to(device)
            detections_gl = images[1].to(device)
            detections_mask = images[2].to(device)

            with torch.no_grad():
                out, _ = model.beam_search(detections, detections_gl, detections_mask, 20,
                                           text_field.vocab.stoi['<eos>'], 5, out_size=1)
            caps_gen = text_field.decode(out, join_words=False)
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen['%d_%d' % (it, i)] = [gen_i, ]
                gts['%d_%d' % (it, i)] = gts_i
            pbar.update()

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)
    return scores




if __name__ == '__main__':
    device = torch.device('cuda')

    parser = argparse.ArgumentParser(description='MG-Transformer')
    parser.add_argument('--exp_name', type=str, default='Sydney')  # Sydney，UCM，RSICD
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--workers', type=int, default=0)

    # sydney
    parser.add_argument('--annotation_folder', type=str,
                        default='/media/dmd/ours/mlw/rs/Sydney_Captions')
    parser.add_argument('--res_features_path', type=str,
                        default='/media/dmd/ours/mlw/rs/multi_scale_T/sydney_res152_con5_224')

    parser.add_argument('--clip_features_path', type=str,
                        default='/media/dmd/ours/mlw/rs/clip_feature/sydney_224')

    parser.add_argument('--mask_features_path', type=str,
                        default='/media/dmd/ours/mlw/rs/multi_scale_T/sydney_group_mask_8_6')


    args = parser.parse_args()

    print('Transformer Evaluation')

    # Pipeline for image regions
    image_field = ImageDetectionsField(clip_features_path=args.clip_features_path,
                                       res_features_path=args.res_features_path,
                                       mask_features_path=args.mask_features_path)
    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False)


    # Create the dataset  Sydney，UCM，RSICD
    if args.exp_name == 'Sydney':
        dataset = Sydney(image_field, text_field, 'Sydney/images/', args.annotation_folder, args.annotation_folder)
    elif args.exp_name == 'UCM':
        dataset = UCM(image_field, text_field, 'UCM/images/', args.annotation_folder, args.annotation_folder)
    elif args.exp_name == 'RSICD':
        dataset = RSICD(image_field, text_field, 'RSICD/images/', args.annotation_folder, args.annotation_folder)

    _, _, test_dataset = dataset.splits

    text_field.vocab = pickle.load(open('vocab_%s.pkl' % args.exp_name, 'rb'))



    # Model and dataloaders
    encoder = MemoryAugmentedEncoder(3, 0, attention_module=GlobalGroupingAttention_with_DC)
    decoder = MeshedDecoder(len(text_field.vocab), 127, 3, text_field.vocab.stoi['<pad>'])
    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)


    data = torch.load('./saved_models/Sydney_best.pth')
    # data = torch.load('./saved_models/Sydney_best_test.pth')

    model.load_state_dict(data['state_dict'])
    dict_dataset_test = test_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=args.batch_size, num_workers=args.workers)

    scores = predict_captions(model, dict_dataloader_test, text_field)


