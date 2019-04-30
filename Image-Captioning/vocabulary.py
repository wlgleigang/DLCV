import nltk
import pickle
import os.path
from pycocotools.coco import COCO
from collections import Counter
#生成或者从文件中加载词汇表
class Vocabulary(object):

    def __init__(self,
        vocab_threshold,#统计进词汇表的阈值.达不到阈值的归为unk_word
        vocab_file='./vocab.pkl',#词汇表存储的文件目录
        start_word="<start>",#句子开始的表示
        end_word="<end>",#句子结束的表示
        unk_word="<unk>",#达不到阈值或者词汇表不存在的单词表示
        annotations_file='../cocoapi/annotations/captions_train2014.json',#标注文件的储存位置(可以看出词汇表只从训练集标注中创建)
        vocab_from_file=False):#表示词汇表的来源,是从文件中,还是创建
        """Initialize the vocabulary.
        Args:
          vocab_threshold: Minimum word count threshold.
          vocab_file: File containing the vocabulary.
          start_word: Special word denoting sentence start.
          end_word: Special word denoting sentence end.
          unk_word: Special word denoting unknown words.
          annotations_file: Path for train annotation file.
          vocab_from_file: If False, create vocab from scratch & override any existing vocab_file
                           If True, load vocab from from existing vocab_file, if it exists
        """
        self.vocab_threshold = vocab_threshold
        self.vocab_file = vocab_file
        self.start_word = start_word
        self.end_word = end_word
        self.unk_word = unk_word
        self.annotations_file = annotations_file
        self.vocab_from_file = vocab_from_file
        self.get_vocab()#获得词汇表

    def get_vocab(self):
        """Load the vocabulary from file OR build the vocabulary from scratch."""
        if os.path.exists(self.vocab_file) & self.vocab_from_file:#在词汇文件存在并且self.vocab_from_file为True的情况下,从文件中加载词汇表
            with open(self.vocab_file, 'rb') as f:#读取词汇表
                vocab = pickle.load(f)
                self.word2idx = vocab.word2idx
                self.idx2word = vocab.idx2word#所以可以通过对象.word2idx 和对象.idx2word
            print('Vocabulary successfully loaded from vocab.pkl file!')
        else:
            self.build_vocab()#独立创建词汇表
            with open(self.vocab_file, 'wb') as f:#存储词汇表
                pickle.dump(self, f)#存储了对象的字节码文件
        
    def build_vocab(self):
        """Populate the dictionaries for converting tokens to integers (and vice-versa)."""
        self.init_vocab()#创建保存词汇表的字典
        self.add_word(self.start_word)
        self.add_word(self.end_word)
        self.add_word(self.unk_word)
        self.add_captions()#添加词汇

    def init_vocab(self):#初始词汇字典
        """Initialize the dictionaries for converting tokens to integers (and vice-versa)."""
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        """Add a token to the vocabulary."""
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def add_captions(self):
        """Loop over training captions and add all tokens to the vocabulary that meet or exceed the threshold."""
        coco = COCO(self.annotations_file)#得到coco对象
        counter = Counter()#计数对象
        ids = coco.anns.keys()#获得标注的keys
        #取得所有训练集的标注的数据,然后单词计数
        for i, id in enumerate(ids):
            caption = str(coco.anns[id]['caption'])#得到相应图像的标注
            tokens = nltk.tokenize.word_tokenize(caption.lower())#分解句子为词汇组成
            counter.update(tokens)#counter对象计数

            if i % 100000 == 0:
                print("[%d/%d] Tokenizing captions..." % (i, len(ids)))

        words = [word for word, cnt in counter.items() if cnt >= self.vocab_threshold]#从计数器中按照要求取出创建词汇表的所有单词

        for i, word in enumerate(words):#遍历所有单词,添入词汇字典
            self.add_word(word)#添入字典的函数

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx[self.unk_word]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)