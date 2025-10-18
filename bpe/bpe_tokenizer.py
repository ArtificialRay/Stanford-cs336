import json
import sys
sys.path.append(".")
from cs336_basics.pretokenization_example import find_chunk_boundaries
import regex as re
import heapq
from collections import defaultdict
from typing import Iterable,Iterator

# regex based pre-tokenizer
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
NUM_CHUNKS = 4


def merge(byte_text:str,new_idx:int,pair:tuple[bytes,bytes])->list[int]:
    doc_encode = []
    i = 0
    while i< len(byte_text):
        if i + 1 < len(byte_text) and byte_text[i] == pair[0] and byte_text[i + 1] == pair[1]:
            doc_encode.append(new_idx)
            i += 2
        else:
            doc_encode.append(byte_text[i])
            i += 1
    return doc_encode

def split_by_special_tokens(text:str,special_tokens:list[str])->list[str]:
    """
    Split on the special tokens
    example: 
        text = "Hello world! <|endoftext|> Great!" 
        special_tokens = "<|endoftext|>"
        result = ['Hello world! ', '<|endoftext|>', ' Great!']
    """
    if special_tokens == None:
        parts = [text]
    else:
        delimiter_pattern = "|".join(re.escape(token) for token in sorted(special_tokens))
        parts = re.split('('+delimiter_pattern+')',text)
    return parts

# def pretokenize(text:str,special_tokens:list[str])->list[bytes]:
#     """
#     separating text into pretokens, and treat special token as independent pre_tokens
#     """
#     PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
#     parts = split_by_special_tokens(text,special_tokens)
#     token_list = []
#     for part in parts:
#         if part == b'':
#             continue
#         if part in special_tokens:
#             token_list.extend([part.encode('utf-8',errors="ignore")])
#         else:
#             str_tokens = re.finditer(PAT,part)
#             part_tokens = [s.group().encode('utf-8',errors="ignore") for s in str_tokens]
#             token_list.extend(part_tokens)
#     return token_list

def pretokenize(text: str, special_tokens: list[str], drop_special_token: bool = True) -> list[bytes]:
    """
    Seperating text into pretokens
    Special tokens are independent pretokens
    """
    parts = split_by_special_tokens(text, special_tokens)

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    tokens_list = []
    for part in parts:
        if part in special_tokens:
            if not drop_special_token:  # Keep special tokens, otherwise ignore
                spec_tok_bytes = part.encode('utf-8')
                tokens_list.append([spec_tok_bytes])
        else:
            str_tokens = re.findall(PAT, part)
            part_tokens = [s.encode('utf-8') for s in str_tokens]
            tokens_list.append(part_tokens)
    tokens = [token for part_tokens in tokens_list for token in part_tokens]
    return tokens


class Tokenizer:
    """Abstract inferface of tokenizer """
    def encode(self, string: str) -> list[int]:
        raise NotImplementedError
    def decode(self, indices: list[int]) -> str:
        raise NotImplementedError

class BPETokenizer(Tokenizer):
    def __init__(self,vocab:dict[int,bytes],merges:list[tuple[bytes,bytes]],special_tokens:list[str]|None=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

    @classmethod
    def from_files(cls,vocab_filepath,merges_filepath,special_tokens=None):
        ## 从文件加载vocab和merges, 返回实例 ##
        vocab:dict[bytes,int] = {}
        with open(vocab_filepath,'r',encoding='utf-8') as f:
            content = json.load(f)
        for key,value in content.items():
            vocab[key.encode("utf-8",errors="ignore")] = value
        del content
        merges:list[tuple[bytes,bytes]] = []
        with open(merges_filepath,"rb") as f:
            for line in f:
                line = line.strip().split()
                merges.append((line[0],line[1]))
        
        return cls(vocab,merges,special_tokens)
    
    def encode(self,text:str)->list[int]:
        # pre-tokenize text
        inverted_vocab = dict(zip(self.vocab.values(),self.vocab.keys()))
        byte_pretokens = pretokenize(text, self.special_tokens, drop_special_token=False)   # list[bytes]
        byte_special_tokens = [token.encode('utf-8') for token in self.special_tokens]
        pretokens = []  # list[list[int]]

        # Convert pretokens from bytes to list[int] by vocab
        for i, pretoken in enumerate(byte_pretokens):

            new_pretoken = []

            if pretoken in byte_special_tokens:
                index = inverted_vocab[pretoken]
                new_pretoken.append(index)
            else:
                for b in pretoken:
                    index = inverted_vocab[bytes([b])]
                    new_pretoken.append(index)

            pretokens.append(new_pretoken)
        return self._merge_fast(pretokens,inverted_vocab)
        # pretokens_byte = pretokenize(text,self.special_tokens)
        # byte_special_tokens = [token.encode('utf-8') for token in self.special_tokens]
        # pretokens = [] #list[list[int]]

        # # convert pretokens from byte to list[int]
        # for pretoken in pretokens_byte:
        #     new_pretoken = []

        #     if pretoken in byte_special_tokens:
        #         index = self.vocab[pretoken]
        #         new_pretoken.append(index)
        #     else:
        #         for b in pretoken:
        #             index = self.vocab[bytes([b])]
        #             new_pretoken.append(index)

        #     pretokens.extend(new_pretoken)
        # return self._merge_fast(pretokens)

        # if self.special_tokens == None:
        #     doc_encodes = [self.vocab[char.encode("utf-8",errors="ignore")] for char in text]
        #     doc_encodes = self._merge_fast(doc_encodes)
        #     del doc
        #     return doc_encodes
        # else:
            
        #     delimiter_pattern = "|".join(re.escape(token) for token in self.special_tokens)
        #     docs = (re.split(delimiter_pattern,text))
        #     doc_encodes = []
        #     for i,doc in enumerate(docs):
        #         doc = [self.vocab[char.encode("utf-8",errors="ignore")] for char in doc if char != ' ']
        #         doc_encodes.extend(self._merge_fast(doc))
        #         del doc
        #         if i<len(docs)-1 and i<len(self.special_tokens):
        #             special_token_id = self.vocab[self.special_tokens[i].encode("utf-8",errors="ignore")]
        #             doc_encodes.append(special_token_id)

        # return doc_encodes

    def encode_iterable(self,iterable:Iterator[str])->Iterator[int]:
        for line in iterable:
            for idx in self.encode(line):
                yield idx
    
    def decode(self,ids:list[int]) -> str:
        # inverted_vocab = dict(zip(self.vocab.values(),self.vocab.keys()))
        # byte_sequences = []

        # for token_id in ids:
        #     byte_seq = inverted_vocab.get(token_id, b'')  # 处理未知ID
        #     byte_sequences.append(byte_seq)
        
        # all_bytes = b''.join(byte_sequences)
        
        # # 使用errors='replace'自动替换无效字节
        # text = all_bytes.decode('utf-8', errors='replace')
        # return text
        
        return_byte = b''
        vocab_size = len(self.vocab)
        replacement_char = "\uFFFD"
        for id in ids:
            if id > vocab_size:
                return_byte += bytes(replacement_char,encoding="utf-8")
            else:
                return_byte += self.vocab[id]

        return return_byte.decode("utf-8",errors="replace")
        
    def _merge_fast(self,tokens:list[int],inverted_vocab:dict[bytes,int])->list[int]:
        # build merge map: merge pair to id
        merge_map:dict[tuple[bytes,bytes],int] = {}
        for pair in self.merges:
            merge_map[pair] = inverted_vocab[pair[0] + pair[1]]
        # all possible merged positions
        positions = defaultdict(list)
        for i in range(len(tokens)-1):
            pair = (tokens[i],tokens[i+1])
            if pair in merge_map:
                heapq.heappush(positions[pair],i)
        
        # apply merge to positions, 从最先出现的pair开始合并
        while positions:
            best_pair = None
            best_pos = int('inf') # 整个文段中第一个出现merged pair的位置
            for pair,pos_list in positions.items():
                # 如果存在pos_list，即pair是可以被合并的，就找这个pair最先出现的位置
                if pos_list and pos_list[0] < best_pos:
                    best_pos = pos_list[0]
                    best_pair = pair
            if best_pair is None:
                break

            # apply merge to token
            tokens[best_pos] = merge_map[best_pair]
            tokens.pop(best_pos+1)# 去掉当前token的后一个token

            # 更新受影响的位置
            for pair in list(positions.keys()):
                new_positions = []
                for pos in positions[pair]: # 每一个pair对应的position列表
                    if pos==best_pos or pos==best_pos+1: # 删除当前pair
                        continue
                    elif pos > best_pos+1:
                        new_positions.append(pos-1) # 所有的pair位置往前移一位
                    else:
                        new_positions.append(pos)
                if new_positions:
                    positions[pair] = new_positions
                    heapq.heapify(positions[pair])
                else:
                    del positions[pair] # if pos==best_pos or pos==best_pos+1剩下的在这里删除
            # 检查新创建的合并对
            # 左侧邻居对
            if best_pos>0:
                left_pair = (tokens[best_pos-1],tokens[best_pos])
                if left_pair in merge_map:
                    heapq.heappush(positions[left_pair],best_pos-1)
            # 右侧邻居对
            if best_pos < len(tokens)-1:
                right_pair = (tokens[best_pos],tokens[best_pos+1])
                if right_pair in merge_map:
                    heapq.heappush(positions[right_pair],best_pos)
        return tokens



        
            


if __name__ == "__main__":
    ## test code ##
    tokenizer = BPETokenizer.from_files(
        vocab_filepath="tests/fixtures/gpt2_vocab.json",
        merges_filepath="tests/fixtures/gpt2_merges.txt",
        special_tokens=["<|endoftext|>","<|endoftext|><|endoftext|>"]
    )
    # print(tokenizer.vocab)
    # print(tokenizer.merges[:100]) 

    test_string = "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"
    ids = tokenizer.encode(test_string)
    print(ids)
    string = tokenizer.decode(ids)
    print(test_string == string)



        