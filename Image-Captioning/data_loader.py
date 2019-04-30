import nltk
import os
import torch
import torch.utils.data as data
from vocabulary import Vocabulary
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
from tqdm import tqdm
import random
import json

def get_loader(transform,mode='train',batch_size=1, vocab_threshold=None,vocab_file='./vocab.pkl',start_word="<start>",end_word="<end>",unk_word="<unk>",vocab_from_file=True,num_workers=0,cocoapi_loc='/opt'):
    """Returns the data loader.
    Args:
      transform: Image transform.
      mode: One of 'train' or 'test'.
      batch_size: Batch size (if in testing mode, must have batch_size=1).
      vocab_threshold: Minimum word count threshold.
      vocab_file: File containing the vocabulary. 
      start_word: Special word denoting sentence start.
      end_word: Special word denoting sentence end.
      unk_word: Special word denoting unknown words.
      vocab_from_file: If False, create vocab from scratch & override any existing vocab_file.
                       If True, load vocab from from existing vocab_file, if it exists.
      num_workers: Number of subprocesses to use for data loading 
      cocoapi_loc: The location of the folder containing the COCO API: https://github.com/cocodataset/cocoapi
    """
    #断言数据mode的取值范围
    assert mode in ['train', 'test'], "mode must be one of 'train' or 'test'."
    #断言如果不能从词汇文件提取词汇字典,mode必须为train,限制了词汇字典必须从训练数据得来
    if vocab_from_file==False: assert mode=='train', "To generate vocab from captions file, must be in training mode (mode='train')."

    # 根据不同的mode,获得相应的图片以及标注的文件
    if mode == 'train':
        #断言要求从文件中得到词汇字典,但是文件不存在的情况
        if vocab_from_file==True: assert os.path.exists(vocab_file), "vocab_file does not exist.  Change vocab_from_file to False to create vocab_file."
        #训练图片以及标注的文件位置
        img_folder = os.path.join(cocoapi_loc, 'cocoapi/images/train2014/')
        annotations_file = os.path.join(cocoapi_loc, 'cocoapi/annotations/captions_train2014.json')
    if mode == 'test':
        #断言test的batch_size必须为1
        assert batch_size==1, "Please change batch_size to 1 if testing your model."
        #断言在test mode下,词汇文件必须存在
        assert os.path.exists(vocab_file), "Must first generate vocab.pkl from training data."
        #断言必须从词汇文件读取词汇字典
        assert vocab_from_file==True, "Change vocab_from_file to True."
        #测试图片以及标注的文件位置
        img_folder = os.path.join(cocoapi_loc, 'cocoapi/images/test2014/')
        annotations_file = os.path.join(cocoapi_loc, 'cocoapi/annotations/image_info_test2014.json')

    # COCO caption dataset.
    #获取数据集
    dataset = CoCoDataset(transform=transform,mode=mode,batch_size=batch_size,vocab_threshold=vocab_threshold,vocab_file=vocab_file, start_word=start_word,end_word=end_word,unk_word=unk_word,annotations_file=annotations_file,vocab_from_file=vocab_from_file,img_folder=img_folder)
    if mode == 'train':
        # Randomly sample a caption length, and sample indices with that length.
        indices = dataset.get_train_indices()
        # Create and assign a batch sampler to retrieve a batch with the sampled indices.
        initial_sampler = data.sampler.SubsetRandomSampler(indices=indices)
        # data loader for COCO dataset.
        data_loader = data.DataLoader(dataset=dataset, num_workers=num_workers,batch_sampler=data.sampler.BatchSampler(sampler=initial_sampler,
                                                                       batch_size=dataset.batch_size,drop_last=False))
    else:
        data_loader = data.DataLoader(dataset=dataset,batch_size=dataset.batch_size,shuffle=True,num_workers=num_workers)

    return data_loader

class CoCoDataset(data.Dataset):#训练集数据和测试集数据获取方式不一
    
    def __init__(self, transform, mode, batch_size, vocab_threshold, vocab_file, start_word, 
        end_word, unk_word, annotations_file, vocab_from_file, img_folder):
        self.transform = transform
        self.mode = mode
        self.batch_size = batch_size
        self.vocab = Vocabulary(vocab_threshold, vocab_file, start_word,end_word, unk_word, annotations_file, vocab_from_file)#获取到词汇表的对象
        self.img_folder = img_folder
        if self.mode == 'train':
            self.coco = COCO(annotations_file)#获取到COCO对象
            self.ids = list(self.coco.anns.keys())#获取到图像和标注的统一keys
            print('Obtaining caption lengths...')
            all_tokens = [nltk.tokenize.word_tokenize(str(self.coco.anns[self.ids[index]]['caption']).lower()) for index in tqdm(np.arange(len(self.ids)))]#将ids遍历,然后通过str(self.coco.anns[self.ids[index]]['caption']获得标注信息,然后获得所有标注的分词列表
            self.caption_lengths = [len(token) for token in all_tokens]#获得每个标注的长度
        else:
            test_info = json.loads(open(annotations_file).read())
            self.paths = [item['file_name'] for item in test_info['images']]#获取图片文件名
        
    def __getitem__(self, index):
        # obtain image and caption if in training mode
        if self.mode == 'train':
            ann_id = self.ids[index]
            caption = self.coco.anns[ann_id]['caption']#获取标注
            img_id = self.coco.anns[ann_id]['image_id']#获取图像id
            path = self.coco.loadImgs(img_id)[0]['file_name']#获取图像文件名

            # 图像预处理
            image = Image.open(os.path.join(self.img_folder, path)).convert('RGB')
            image = self.transform(image)

            #获取标注
            tokens = nltk.tokenize.word_tokenize(str(caption).lower())
            caption = []
            caption.append(self.vocab(self.vocab.start_word))
            caption.extend([self.vocab(token) for token in tokens])
            caption.append(self.vocab(self.vocab.end_word))
            caption = torch.Tensor(caption).long()
            
            return image, caption

        else:
            #获取图片
            path = self.paths[index]
            PIL_image = Image.open(os.path.join(self.img_folder, path)).convert('RGB')
            orig_image = np.array(PIL_image)
            image = self.transform(PIL_image)
            return orig_image, image

    def get_train_indices(self):
        sel_length = np.random.choice(self.caption_lengths)#随机选取一个标注长度
        #选择标注的长度等于sel_length的索引列表
        all_indices = np.where([self.caption_lengths[i] == sel_length for i in np.arange(len(self.caption_lengths))])[0]
        indices = list(np.random.choice(all_indices, size=self.batch_size))#随机选择batch_size个索引
        return indices

    def __len__(self):
        if self.mode == 'train':
            return len(self.ids)
        else:
            return len(self.paths)