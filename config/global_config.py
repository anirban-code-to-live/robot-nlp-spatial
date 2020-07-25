class GlobalConfig(object):
    def __init__(self):
        super(GlobalConfig, self).__init__()

        self.MAX_SEQ_LEN = 40
        self.IMAGE_HEIGHT = 600
        self.IMAGE_WIDTH = 600
        self.EMOJI_HEIGHT = 15
        self.EMOJI_WIDTH = 15

        self.OFFSET_MIN = 5
        self.OFFSET_MAX = 10
        self.SCENE_MAX = 128

        self.UNIQUE_OBJECT_COUNT = 12
        self.MAX_SCENE_OBJECT_COUNT = 25

        self.PAD_TOKEN = 0
        self.UNK_TOKEN = 1

        self.VOCAB_SIZE = 43
        self.EMBED_SIZE = 32

        self.SCALE = 93
        self.MIN_OBJ_SEP = 0.05
