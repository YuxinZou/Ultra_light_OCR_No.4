# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import string
from abc import ABCMeta, abstractmethod

import numpy as np
import paddle
from queue import PriorityQueue


class BaseRecLabelDecode(metaclass=ABCMeta):
    """ Convert between text-label and text-index """

    def __init__(self,
                 character_dict_path=None,
                 character_type='ch',
                 use_space_char=False):
        support_character_type = [
            'ch', 'en', 'EN_symbol', 'french', 'german', 'japan', 'korean',
            'it', 'xi', 'pu', 'ru', 'ar', 'ta', 'ug', 'fa', 'ur', 'rs', 'oc',
            'rsc', 'bg', 'uk', 'be', 'te', 'ka', 'chinese_cht', 'hi', 'mr',
            'ne', 'EN', 'latin', 'arabic', 'cyrillic', 'devanagari'
        ]
        assert character_type in support_character_type, \
            "Only {} are supported now but get {}".format(
                support_character_type, character_type)

        self.beg_str = "sos"
        self.end_str = "eos"

        if character_type == "en":
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
        elif character_type == "EN_symbol":
            # same with ASTER setting (use 94 char).
            self.character_str = string.printable[:-6]
            dict_character = list(self.character_str)
        elif character_type in support_character_type:
            self.character_str = ""
            assert character_dict_path is not None, \
                "character_dict_path should not be None when " \
                "character_type is {}".format(character_type)
            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode('utf-8').strip("\n").strip("\r\n")
                    self.character_str += line
            if use_space_char:
                self.character_str += " "
            dict_character = list(self.character_str)

        else:
            raise NotImplementedError
        self.character_type = character_type
        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

    @abstractmethod
    def add_special_char(self, dict_character: str) -> str:
        pass

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """ convert text-index into text-label. """
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                if text_index[batch_idx][idx] in ignored_tokens:
                    continue
                if is_remove_duplicate:
                    # only for predict
                    if idx > 0 and text_index[batch_idx][
                        idx - 1] == text_index[batch_idx][idx]:
                        continue
                char_list.append(self.character[int(
                    text_index[batch_idx][idx])])
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)
            text = ''.join(char_list)
            result_list.append((text, np.mean(conf_list)))
        return result_list

    def get_ignored_tokens(self):
        return [0]  # for ctc blank


class DecodeNode:
    def __init__(self, indexes=[0], scores=[0.0]):
        self.indexes = indexes
        self.scores = scores

    def eval(self):
        """Calculate accumulated score."""
        accu_score = sum(self.scores)
        return accu_score


class CTCLabelBeamSearchDecode(BaseRecLabelDecode):
    """ Convert between text-label and text-index """

    def __init__(self,
                 character_dict_path=None,
                 character_type='ch',
                 use_space_char=False,
                 beam_width=2,
                 **kwargs):
        super(CTCLabelBeamSearchDecode, self).__init__(character_dict_path,
                                                       character_type,
                                                       use_space_char)
        self.beam_width = beam_width

    def __call__(self, preds, label=None, is_remove_duplicate=True, *args,
                 **kwargs):
        # preds shape: B * T * C
        B, T, C = preds.shape
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        for batch_idx in range(B):
            char_list = []
            conf_list = []
            q = PriorityQueue()
            init_node = DecodeNode([0], [0.0])
            q.put((-init_node.eval(), init_node))
            for idx in range(1, 1 + len(preds[batch_idx])):
                next_nodes = []
                beam_width = self.beam_width if idx > 1 else 1
                # print(f'beam_width:{beam_width}')
                for _ in range(beam_width):
                    _, node = q.get()
                    # print(node.indexes, node.scores)
                    output_char = preds[batch_idx][idx - 1, :]  # bsz * num_classes
                    topk_value, topk_idx = output_char.topk(self.beam_width)
                    # print(topk_value, topk_idx)
                    for k in range(self.beam_width):
                        kth_score = float(topk_value[k].numpy())
                        kth_idx = int(topk_idx[k].numpy())
                        next_node = DecodeNode(node.indexes + [kth_idx],
                                               node.scores + [kth_score])
                        # print(next_node.indexes, next_node.scores)
                        delta = k * 1e-6
                        # next_nodes.append(
                        #     (-node.eval() - kth_score - delta, next_node))
                        next_nodes.append(
                            (-node.eval() + np.log(kth_score) - delta, next_node))
                        # Use minus since priority queue sort
                        # with ascending order
                    # print("######")
                while not q.empty():
                    q.get()
                for next_node in next_nodes:
                    q.put(next_node)
            best_node = q.get()
            indexes = best_node[1].indexes[1:]
            scores = best_node[1].scores[1:]

            for idx in range(len(indexes)):
                if indexes[idx] in ignored_tokens:
                    continue
                if is_remove_duplicate:
                    # only for predict
                    if idx > 0 and indexes[idx - 1] == indexes[idx]:
                        continue
                # print(idx)
                char_list.append(self.character[int(
                    indexes[idx])])
                if conf_list is not None:
                    conf_list.append(scores[idx])
                else:
                    conf_list.append(1)
            text = ''.join(char_list)
            result_list.append((text, np.mean(conf_list)))
        if label is None:
            return result_list
        label = self.decode(label)
        return result_list, label

    def add_special_char(self, dict_character):
        dict_character = ['blank'] + dict_character
        return dict_character


class CTCLabelDecode(BaseRecLabelDecode):
    """ Convert between text-label and text-index """

    def __init__(self,
                 character_dict_path=None,
                 character_type='ch',
                 use_space_char=False,
                 **kwargs):
        super(CTCLabelDecode, self).__init__(character_dict_path,
                                             character_type, use_space_char)

    def __call__(self, preds, label=None, *args, **kwargs):
        if isinstance(preds, paddle.Tensor):
            preds = preds.numpy()
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=True)
        if label is None:
            return text
        label = self.decode(label)
        return text, label

    def add_special_char(self, dict_character):
        dict_character = ['blank'] + dict_character
        return dict_character


class AttnLabelDecode(BaseRecLabelDecode):
    """ Convert between text-label and text-index """

    def __init__(self,
                 character_dict_path=None,
                 character_type='ch',
                 use_space_char=False,
                 **kwargs):
        super(AttnLabelDecode, self).__init__(character_dict_path,
                                              character_type, use_space_char)

    def add_special_char(self, dict_character):
        self.beg_str = "sos"
        self.end_str = "eos"
        dict_character = dict_character
        dict_character = [self.beg_str] + dict_character + [self.end_str]
        return dict_character

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """ convert text-index into text-label. """
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        [beg_idx, end_idx] = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                if text_index[batch_idx][idx] in ignored_tokens:
                    continue
                if int(text_index[batch_idx][idx]) == int(end_idx):
                    break
                if is_remove_duplicate:
                    # only for predict
                    if idx > 0 and text_index[batch_idx][
                        idx - 1] == text_index[batch_idx][idx]:
                        continue
                char_list.append(self.character[int(
                    text_index[batch_idx][idx])])
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)
            text = ''.join(char_list)
            result_list.append((text, np.mean(conf_list)))
        return result_list

    def __call__(self, preds, label=None, *args, **kwargs):
        """
        text = self.decode(text)
        if label is None:
            return text
        else:
            label = self.decode(label, is_remove_duplicate=False)
            return text, label
        """
        if isinstance(preds, paddle.Tensor):
            preds = preds.numpy()

        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=False)
        if label is None:
            return text
        label = self.decode(label, is_remove_duplicate=False)
        return text, label

    def get_ignored_tokens(self):
        beg_idx = self.get_beg_end_flag_idx("beg")
        end_idx = self.get_beg_end_flag_idx("end")
        return [beg_idx, end_idx]

    def get_beg_end_flag_idx(self, beg_or_end):
        if beg_or_end == "beg":
            idx = np.array(self.dict[self.beg_str])
        elif beg_or_end == "end":
            idx = np.array(self.dict[self.end_str])
        else:
            assert False, "unsupport type %s in get_beg_end_flag_idx" \
                          % beg_or_end
        return idx


class SRNLabelDecode(BaseRecLabelDecode):
    """ Convert between text-label and text-index """

    def __init__(self,
                 character_dict_path=None,
                 character_type='en',
                 use_space_char=False,
                 **kwargs):
        super(SRNLabelDecode, self).__init__(character_dict_path,
                                             character_type, use_space_char)

    def __call__(self, preds, label=None, *args, **kwargs):
        pred = preds['predict']
        char_num = len(self.character_str) + 2
        if isinstance(pred, paddle.Tensor):
            pred = pred.numpy()
        pred = np.reshape(pred, [-1, char_num])

        preds_idx = np.argmax(pred, axis=1)
        preds_prob = np.max(pred, axis=1)

        preds_idx = np.reshape(preds_idx, [-1, 25])

        preds_prob = np.reshape(preds_prob, [-1, 25])

        text = self.decode(preds_idx, preds_prob)

        if label is None:
            text = self.decode(
                preds_idx, preds_prob, is_remove_duplicate=False)
            return text
        label = self.decode(label)
        return text, label

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """ convert text-index into text-label. """
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)

        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                if text_index[batch_idx][idx] in ignored_tokens:
                    continue
                if is_remove_duplicate:
                    # only for predict
                    if idx > 0 and text_index[batch_idx][
                        idx - 1] == text_index[batch_idx][idx]:
                        continue
                char_list.append(self.character[int(
                    text_index[batch_idx][idx])])
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)

            text = ''.join(char_list)
            result_list.append((text, np.mean(conf_list)))
        return result_list

    def add_special_char(self, dict_character):
        dict_character = dict_character + [self.beg_str, self.end_str]
        return dict_character

    def get_ignored_tokens(self):
        beg_idx = self.get_beg_end_flag_idx("beg")
        end_idx = self.get_beg_end_flag_idx("end")
        return [beg_idx, end_idx]

    def get_beg_end_flag_idx(self, beg_or_end):
        if beg_or_end == "beg":
            idx = np.array(self.dict[self.beg_str])
        elif beg_or_end == "end":
            idx = np.array(self.dict[self.end_str])
        else:
            assert False, "unsupport type %s in get_beg_end_flag_idx" \
                          % beg_or_end
        return idx
