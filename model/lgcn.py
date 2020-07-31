import torch
import torch.nn.functional as F
from pyhocon import ConfigTree
from torch import nn
# from torchpie.config import config
# from torchpie.logging import logger
from utils.config import config
import logging

logger = logging.getLogger(__name__)

from .attention import BidafAttn
from .rnn import RNNEncoder
from .embedding import Embedding
from .gcn import GCN
from .PosEmbed import positionalencoding1d
import ipdb


class LGCN(nn.Module):
    multiple_choice_tasks = ['action', 'transition']

    def __init__(self, opt: ConfigTree):
        super().__init__()
        self.opt = opt

        self.vocab_size = opt.get_int('vocab_size')
        self.char_vocab_size = opt.get_int('char_vocab_size')
        self.hidden_size = opt.get_int('hidden_size')
        self.video_channels = opt.get_int('video_channels')
        self.c3d_channels = opt.get_int('c3d_channels')
        self.position_dim = 128
        self.num_classes = opt.get_int('num_classes')
        self.task = opt.get_string('task')
        self.num_frames = opt.get_int('num_frames')
        self.pooling = opt.get_string('pooling')
        self.use_char_embedding = opt.get_bool('character_embedding')
        self.use_gcn = opt.get_bool('use_gcn')
        self.use_c3d = opt.get_bool('use_c3d')
        self.use_bbox = opt.get_bool('use_bbox')
        self.use_bboxPos = opt.get_bool('use_bboxPos')
        self.use_framePos = opt.get_bool('use_framePos')
        self.use_image = opt.get_bool('use_image')
        self.use_boxFC = opt.get_bool('use_boxFC')
        self.use_boxLSTM = opt.get_bool('use_boxLSTM')
        self.num_box = opt.get_int('num_bbox')
        self.node_dim = opt.get_int('gcn.node_dim')

        self.is_multiple_choice = self.task in self.multiple_choice_tasks or opt.get_bool(
            'is_multiple_choice')

        logger.warning(f'self.is_multiple_choice: {self.is_multiple_choice}')

        logger.warning(f'Using {self.num_box} boxes!')

        if 'embedding_path' not in opt:
            self.embedding = nn.Embedding(self.vocab_size, 300)
            logger.info('Init embedding randomly.')
        else:
            embedding_path = opt.get_string('embedding_path')
            self.embedding = nn.Embedding.from_pretrained(
                torch.load(embedding_path), freeze=False
            )
            logger.info(f'Using pretrained embedding: {embedding_path}')

        if self.use_char_embedding:
            self.char_embedding = nn.Embedding(self.char_vocab_size, 64)
            self.mix_embedding = Embedding(300, 64, 300)
            logger.info('Using char embedding!')

        self.out_features = self.hidden_size * 2
        if self.use_bbox:

            node_dim = self.node_dim
            node_dim += self.position_dim if self.use_bboxPos else 0
            node_dim += self.position_dim if self.use_framePos else 0
            self.gcn_fc = nn.Sequential(
                nn.Linear(node_dim, self.out_features),
                nn.ELU(inplace=True),
            )

            if self.use_bboxPos:
                self.bbox_fc = nn.Sequential(
                    nn.Conv2d(4, 64, kernel_size=1),
                    nn.ReLU(),
                    nn.BatchNorm2d(64),
                    # nn.Dropout(0.5),
                    nn.Conv2d(64, 128, kernel_size=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    # nn.Dropout(0.5)
                )
                logger.info("Using bboxPos")

            if self.use_framePos:
                self.framePos = positionalencoding1d(128, self.num_frames)
                self.framePos = self.framePos.unsqueeze(
                    1).expand(-1, self.num_box, -1).cuda()
                logger.info("Using framePos")

            if self.use_gcn:
                logger.info('Init GCN')
                self.gcn = GCN(
                    self.out_features,
                    self.out_features,
                    self.out_features,
                    0.5,
                    opt.get_list('gcn.mode'),
                    True,
                    opt.get_int('gcn.num_layers'),
                    ST_n_next=opt.get_int('gcn.ST_n_next')
                )
            else:
                logger.warning('Use bbox only')

            if self.use_boxFC:
                self.boxFC = nn.Sequential(
                    nn.Linear(self.out_features, self.out_features),
                    nn.ELU(inplace=True),
                )
            if self.use_boxLSTM:
                self.boxLSTM = nn.LSTM(self.out_features, int(self.out_features/2),
                                    num_layers=1,
                                    batch_first=True,
                                    bidirectional=True,
                                    dropout=0)


        if self.use_c3d:
            logger.warning('Use c3d')
            self.c3d_fc = nn.Sequential(
                nn.Conv1d(self.c3d_channels,
                          self.out_features, 3, padding=1),
                nn.ELU(inplace=True)
            )

        self.num_streams = sum([self.use_image, self.use_bbox, self.use_c3d])
        self.merge = nn.Sequential(
            nn.Linear(self.out_features * self.num_streams, self.out_features),
            nn.ELU(inplace=True)
        )

        self.lstm_raw = RNNEncoder(
            300, self.hidden_size, bidirectional=True, num_layers=1, rnn=nn.LSTM)

        self.video_fc = nn.Sequential(
            nn.Conv1d(self.video_channels, self.out_features, 3, padding=1),
            nn.ELU(inplace=True)
        )

        self.attention = BidafAttn(None, method='dot')

        if self.is_multiple_choice:
            self.lstm_input_size = self.out_features * 5
        else:
            self.lstm_input_size = self.out_features * 3

        self.lstm_mature = RNNEncoder(
            self.lstm_input_size,
            self.out_features,
            bidirectional=True,
            num_layers=1,
            rnn=nn.LSTM
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.out_features * 2, self.num_classes)
        )

    def forward(
            self,
            question, question_length, question_chars,
            a1, a1_length, a1_chars,
            a2, a2_length, a2_chars,
            a3, a3_length, a3_chars,
            a4, a4_length, a4_chars,
            a5, a5_length, a5_chars,
            features, c3d_features, bbox_features, bbox
    ):

        # import ipdb; ipdb.set_trace()
        # B, T, N, _ = bbox_features.shape
        # assert N == self.num_box
        B = question.shape[0]
        if self.use_bbox:
            video_length = torch.tensor(
                [self.num_frames * self.num_box] * B, dtype=torch.long)
        else:
            video_length = torch.tensor(
                [self.num_frames] * B, dtype=torch.long)

        question_embedding = self.embedding(question)

        if self.use_char_embedding:
            question_chars = self.char_embedding(question_chars)

            question_embedding = self.mix_embedding(
                question_chars, question_embedding)

        if self.is_multiple_choice:
            a1_embedding = self.embedding(a1)
            a2_embedding = self.embedding(a2)
            a3_embedding = self.embedding(a3)
            a4_embedding = self.embedding(a4)
            a5_embedding = self.embedding(a5)

            if self.use_char_embedding:
                a1_chars = self.char_embedding(a1_chars)
                a2_chars = self.char_embedding(a2_chars)
                a3_chars = self.char_embedding(a3_chars)
                a4_chars = self.char_embedding(a4_chars)
                a5_chars = self.char_embedding(a5_chars)

                a1_embedding = self.mix_embedding(a1_chars, a1_embedding)
                a2_embedding = self.mix_embedding(a2_chars, a2_embedding)
                a3_embedding = self.mix_embedding(a3_chars, a3_embedding)
                a4_embedding = self.mix_embedding(a4_chars, a4_embedding)
                a5_embedding = self.mix_embedding(a5_chars, a5_embedding)

        raw_out_question, _ = self.lstm_raw(
            question_embedding, question_length)

        if self.is_multiple_choice:
            raw_out_a1, _ = self.lstm_raw(a1_embedding, a1_length)
            raw_out_a2, _ = self.lstm_raw(a2_embedding, a2_length)
            raw_out_a3, _ = self.lstm_raw(a3_embedding, a3_length)
            raw_out_a4, _ = self.lstm_raw(a4_embedding, a4_length)
            raw_out_a5, _ = self.lstm_raw(a5_embedding, a5_length)

        video_embedding = self.video_fc(
            features.transpose(1, 2)).transpose(1, 2)

        if self.use_bbox:
            video_embedding = video_embedding.unsqueeze(
                2).expand(-1, -1, self.num_box, -1).reshape(B, -1, self.out_features)

        streams = []
        if self.use_image:
            streams.append(video_embedding)

        if self.use_c3d:
            c3d_embedding = self.c3d_fc(
                c3d_features.transpose(1, 2)
            ).transpose(1, 2)
            if self.use_bbox:
                c3d_embedding = c3d_embedding.unsqueeze(
                    2).expand(-1, -1, self.num_box, -1).reshape(B, -1, self.out_features)

            streams.append(c3d_embedding)

        if self.use_bbox:

            """bboxPos and framePos"""
            if self.use_bboxPos:
                bbox_pos = self.bbox_fc(bbox.permute(
                    0, 3, 1, 2)).permute(0, 2, 3, 1)
                bbox_features = torch.cat(
                    [bbox_features, bbox_pos], dim=-1)
            if self.use_framePos:
                framePos = self.framePos.unsqueeze(0).expand(B, -1, -1, -1)
                bbox_features = torch.cat(
                    [bbox_features, framePos], dim=-1)

            bbox_features = self.gcn_fc(bbox_features)

            bbox_features = bbox_features.view(B, -1, self.out_features)

            if self.use_gcn:
                bbox_features = self.gcn(bbox_features, video_length, bbox)
            if self.use_boxFC:
                bbox_features = self.boxFC(bbox_features)
            if self.use_boxLSTM:
                bbox_features, _ = self.boxLSTM(bbox_features)

            streams.append(bbox_features)

        assert len(streams) != 0
        streams = torch.cat(streams, dim=-1)
        video_embedding = self.merge(streams)

        u_q, _ = self.attention(
            video_embedding, video_length, raw_out_question, question_length)

        if self.is_multiple_choice:
            u_a1, _ = self.attention(
                video_embedding, video_length, raw_out_a1, a1_length)
            u_a2, _ = self.attention(
                video_embedding, video_length, raw_out_a2, a2_length)
            u_a3, _ = self.attention(
                video_embedding, video_length, raw_out_a3, a3_length)
            u_a4, _ = self.attention(
                video_embedding, video_length, raw_out_a4, a4_length)
            u_a5, _ = self.attention(
                video_embedding, video_length, raw_out_a5, a5_length)

            concat_a1 = torch.cat(
                [video_embedding, u_a1, u_q, u_a1 * video_embedding, u_q * video_embedding], dim=-1
            )
            concat_a2 = torch.cat(
                [video_embedding, u_a2, u_q, u_a2 * video_embedding, u_q * video_embedding], dim=-1
            )
            concat_a3 = torch.cat(
                [video_embedding, u_a3, u_q, u_a3 * video_embedding, u_q * video_embedding], dim=-1
            )
            concat_a4 = torch.cat(
                [video_embedding, u_a4, u_q, u_a4 * video_embedding, u_q * video_embedding], dim=-1
            )
            concat_a5 = torch.cat(
                [video_embedding, u_a5, u_q, u_a5 * video_embedding, u_q * video_embedding], dim=-1
            )

            mature_out_a1, _ = self.lstm_mature(concat_a1, video_length)
            mature_out_a2, _ = self.lstm_mature(concat_a2, video_length)
            mature_out_a3, _ = self.lstm_mature(concat_a3, video_length)
            mature_out_a4, _ = self.lstm_mature(concat_a4, video_length)
            mature_out_a5, _ = self.lstm_mature(concat_a5, video_length)

            matrue_maxout_a1 = self._pooling(mature_out_a1, keepdim=True)
            matrue_maxout_a2 = self._pooling(mature_out_a2, keepdim=True)
            matrue_maxout_a3 = self._pooling(mature_out_a3, keepdim=True)
            matrue_maxout_a4 = self._pooling(mature_out_a4, keepdim=True)
            matrue_maxout_a5 = self._pooling(mature_out_a5, keepdim=True)

            mature_answers = torch.cat(
                [matrue_maxout_a1, matrue_maxout_a2, matrue_maxout_a3,
                 matrue_maxout_a4, matrue_maxout_a5],
                dim=1
            )

            out = self.classifier(mature_answers)
            out = out.squeeze()

            return out

        else:
            concat_q = torch.cat(
                [video_embedding, u_q, video_embedding * u_q], dim=-1
            )

            mature_out_q, _ = self.lstm_mature(concat_q, video_length)

            mature_maxout_q = self._pooling(mature_out_q, keepdim=False)

            out = self.classifier(mature_maxout_q)

            if self.task == 'count':
                # out = out.round().clamp(1, 10).squeeze()
                # out = torch.sigmoid(out)
                out = out.squeeze()

            return out

    def _pooling(self, x: torch.Tensor, keepdim=False) -> torch.Tensor:
        if self.pooling == 'max':
            out, _ = x.max(1, keepdim=keepdim)
        elif self.pooling == 'mean':
            out = x.mean(1, keepdim=keepdim)
        elif self.pooling == 'sum':
            out = x.sum(1, keepdim=keepdim)
        else:
            raise Exception(f'No such pooling: {self.pooling}')

        return out
