import sys
sys.path.append(".")
import regex as re
import os
from cs336_basics import find_chunk_boundaries
import multiprocessing
from functools import partial

## bpe training implementation and scripts:
# regex based pre-tokenizer
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
#NUM_PROCESSES = 4
NUM_PROCESSES = multiprocessing.cpu_count() // 8


def merge(merged_pair:tuple,vocab:dict,new_token_id:int,pair_freq_table:dict,word_freq_table:dict):
    word_encodings = list(word_freq_table.keys()).copy() # word_freq_table may change, copy at first
    for word_encode in word_encodings:
        if word_encode not in word_freq_table:
            continue # 若当前word_encode已经被删除
        word_freq = word_freq_table[word_encode]
        merged_positions = []
        # find the position of merged pair in this word
        # merge pair可能会多次出现! 
        i=0
        while i < len(word_encode)-1:
            if (vocab[word_encode[i]],vocab[word_encode[i+1]]) == merged_pair:
                merged_positions.append(i)
                i+=2 # 跳过被合并的pair
            else:
                i+=1
        # for i in range(len(word_encode)-1):
        #     pair = (word_encode[i],word_encode[i+1])
        #     if(pair == merged_pair):
        #         merged_positions.append(i)
        # 当前word中找到了merged pair
        # if len(merged_positions) >1:
        #     print("hey!")
        curr_word_encode = word_encode
        for merged_pair_pos in sorted(merged_positions,reverse=True):
            # 取当前pair出现的概率
            # 生成新的word encode
            new_word_encode = curr_word_encode[:merged_pair_pos] + (new_token_id,) + curr_word_encode[merged_pair_pos+2:]
            # 在word frequency table中新增new word encode，old word encode出现的频率可能降低，但并不为0
            word_freq_table[new_word_encode] = word_freq_table.get(new_word_encode,0) + word_freq
            word_freq_table[curr_word_encode] -= word_freq
            if word_freq_table[curr_word_encode] <= 0:
                word_freq_table.pop(curr_word_encode)
            # if new_token_id==363:
            #     print([vocab[idx] for idx in word_encode])
            # 减少旧的pair出现次数
            pair_freq = word_freq
            if merged_pair_pos-1 >=0:
                left_pair_ori = (vocab[curr_word_encode[merged_pair_pos-1]],vocab[curr_word_encode[merged_pair_pos]])
                pair_freq_table[left_pair_ori] -= pair_freq # 旧的pair减去当前词的频率
                if pair_freq_table[left_pair_ori] <= 0:
                    pair_freq_table.pop(left_pair_ori) # 如果不再存在这个pair，将这个pair从frequency table中删除
            
            # 处理被合并的pair本身
            pair_freq_table[merged_pair] -= pair_freq
            if pair_freq_table[merged_pair] <= 0:
                pair_freq_table.pop(merged_pair)

            if merged_pair_pos+2 < len(curr_word_encode): # right_neighbor_pos+2指这个字节对后第一个字节的位置，检查是否存在这个字节
                right_pair_ori = (vocab[curr_word_encode[merged_pair_pos+1]],vocab[curr_word_encode[merged_pair_pos+2]])
                pair_freq_table[right_pair_ori] -= pair_freq
                if pair_freq_table[right_pair_ori] <= 0:
                    pair_freq_table.pop(right_pair_ori) # 如果不再存在这个pair，将这个pair从frequency table中删除
            
            # 增加因为merged pair而新出现的pair
            if merged_pair_pos > 0:
                left_pair_new = (vocab[curr_word_encode[merged_pair_pos-1]],vocab[new_token_id])
                pair_freq_table[left_pair_new] = pair_freq_table.get(left_pair_new,0) + pair_freq
            
            if merged_pair_pos + 2 < len(curr_word_encode):
                right_pair_new = (vocab[new_token_id],vocab[curr_word_encode[merged_pair_pos+2]])
                pair_freq_table[right_pair_new] = pair_freq_table.get(right_pair_new,0) + pair_freq
            
            curr_word_encode = new_word_encode # 更新word_encode以处理下一个合并位置


def train_bpe_chunks(chunks: str, vocab_size:int, special_tokens:list[str],
                     vocab:dict[int, bytes], merges:list[tuple[bytes, bytes]]):
    # first get rid of all special tokens,需要对special token前后的文段分别进行pre-tokenize
    delimiter_pattern = "|".join(re.escape(token) for token in special_tokens)
    docs = re.split(delimiter_pattern,chunks)
    pair_freq_table:dict[tuple[bytes],int] = dict() # pair frequency table for this chunk
    word_freq_table:dict[tuple[bytes],int] = dict()
    # special tokens 的初始位置排在vocab的第一位，索引vocab时要加入special token length
    offset = len(special_tokens)

    #pair_position = defaultdict(list) # position of pair: {pair: [(doc_idx, word_idx,position)], ...}
    
    for doc in docs:
        doc = re.finditer(PAT,doc) # 分词后的chunks迭代器
        for word in doc:
            word_encode = word.group().encode('utf-8')
            if word_encode == b'':
                continue
            word_encode = tuple([char+offset for char in word_encode])
            if word_encode in word_freq_table:
                word_freq_table[word_encode] += 1
            else:
                word_freq_table[word_encode] = 1
            for i in range(len(word_encode)-1):
                pair = (vocab[word_encode[i]],vocab[word_encode[i+1]])
                if pair in pair_freq_table:
                    pair_freq_table[pair] += 1
                else:
                    pair_freq_table[pair] = 1

    # merge each pair efficiently
    num_merges = vocab_size - (256+len(special_tokens)) # 可以新增的merge pair数量
    for i in range(num_merges):
        # find the most common pair
        pair = max(pair_freq_table.items(), key=lambda x:(x[1],x[0])) # choose lexicographically greater pair
        c1,c2 = pair[0]
        # merge that pair
        new_index = 256+offset+i
        # byte_pair = tuple(vocab[id] for id in pair[0])
        merges.append(pair[0])
        vocab[new_index] = c1+c2
        # 每次合并后，只有与合并对相邻的字节对计数会发生变化，其它字节对复用之前的计数
        merge(pair[0],vocab,new_index,pair_freq_table,word_freq_table)


def make_chunk_freq_table(chunk_bytes:bytes,vocab:dict,special_tokens:list[str]):
    # 只统计word/pair的frequency，不合并
    chunks = chunk_bytes.decode("utf-8",errors="ignore")
    # first get rid of all special tokens,需要对special token前后的文段分别进行pre-tokenize
    delimiter_pattern = "|".join(re.escape(token) for token in special_tokens)
    docs = re.split(delimiter_pattern,chunks)
    pair_freq_table:dict[tuple[bytes],int] = dict() # pair frequency table for this chunk
    word_freq_table:dict[tuple[bytes],int] = dict()
    # special tokens 的初始位置排在vocab的第一位，索引vocab时要加入special token length
    offset = len(special_tokens)

    for doc in docs:
        doc = re.finditer(PAT,doc) # 分词后的chunks迭代器
        for word in doc:
            word_encode = word.group().encode('utf-8')
            if word_encode == b'':
                continue
            word_encode = tuple([char+offset for char in word_encode])
            if word_encode in word_freq_table:
                word_freq_table[word_encode] += 1
            else:
                word_freq_table[word_encode] = 1
            for i in range(len(word_encode)-1):
                pair = (vocab[word_encode[i]],vocab[word_encode[i+1]])
                if pair in pair_freq_table:
                    pair_freq_table[pair] += 1
                else:
                    pair_freq_table[pair] = 1
    return pair_freq_table,word_freq_table


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
)-> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab:dict[int, bytes] = {x+len(special_tokens): bytes([x]) for x in range(256)} 
    merges:list[tuple[bytes, bytes]] = []
    for i in range(len(special_tokens)):
        vocab[i] = special_tokens[i].encode('utf-8')
    offset = len(special_tokens)

    # 全局统计所有pair和word的频率
    glb_pair_freq_table:dict[tuple[bytes],int] = dict() # pair frequency table for this chunk
    glb_word_freq_table:dict[tuple[int],int] = dict()

    with open(input_path,"rb") as f:
        boundaries = find_chunk_boundaries(f,NUM_PROCESSES,b"<|endoftext|>")
        chunks_data = []
        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk_bytes = f.read(end - start)
            chunks_data.append(chunk_bytes)
    # 多进程只用于统计全局的pair_freq和word_freq
    with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
        process_func = partial(make_chunk_freq_table,vocab=vocab,special_tokens=special_tokens) # 用partial固定部分参数
        freq_tables = pool.map(process_func,chunks_data)

        for chunk_pair_freq,chunk_word_freq in freq_tables:
            for pair,freq in chunk_pair_freq.items():
                glb_pair_freq_table[pair] = glb_pair_freq_table.get(pair,0) + freq
            for word,freq in chunk_word_freq.items():
                glb_word_freq_table[word] = glb_word_freq_table.get(word,0) + freq
    
    # merge each pair efficiently
    num_merges = vocab_size - (256+len(special_tokens)) # 可以新增的merge pair数量
    for i in range(num_merges):
        # find the most common pair
        best_pair = max(glb_pair_freq_table.items(), key=lambda x: (x[1], x[0]))[0]
        c1,c2 = best_pair
        # merge that pair
        new_index = 256+offset+i
        # byte_pair = tuple(vocab[id] for id in pair[0])
        merges.append(best_pair)
        vocab[new_index] = c1+c2
        # 每次合并后，只有与合并对相邻的字节对计数会发生变化，其它字节对复用之前的计数
        merge(best_pair,vocab,new_index,glb_pair_freq_table,glb_word_freq_table)
    
    # for the convenience of encoding at BPETokenizer, 将vocab的键改为byte, 值改为int
    return_vocab = dict(zip(vocab.values(), vocab.keys()))
    del vocab
    return return_vocab,merges

# def process_single_chunk(chunk_bytes:bytes,vocab_size:int,special_tokens:list[str]):
#     # 重新初始化vocab和merges，防止data race
#     vocab:dict[int, bytes] = {x+len(special_tokens): bytes([x]) for x in range(256)} 
#     merges:list[tuple[bytes, bytes]] = []
#     for i in range(len(special_tokens)):
#         vocab[i] = special_tokens[i].encode('utf-8')

#     chunks = chunk_bytes.decode("utf-8", errors="ignore")
#     train_bpe_chunks(chunks,vocab_size,special_tokens,vocab,merges)

#     return vocab,merges            


# def train_bpe(
#     input_path: str | os.PathLike,
#     vocab_size: int,
#     special_tokens: list[str],
#     **kwargs,
# )-> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
#     vocab:dict[int, bytes] = {x+len(special_tokens): bytes([x]) for x in range(256)} 
#     merges:list[tuple[bytes, bytes]] = []
#     for i in range(len(special_tokens)):
#         vocab[i] = special_tokens[i].encode('utf-8')
    
#     with open(input_path,"rb") as f:
#         boundaries = find_chunk_boundaries(f,NUM_PROCESSES,b"<|endoftext|>")
#         chunks_data = []
#         # The following is a serial implementation, but you can parallelize this
#         # by sending each start/end pair to a set of processes.
#         for start, end in zip(boundaries[:-1], boundaries[1:]):
#             f.seek(start)
#             chunk_bytes = f.read(end - start)
#             chunks_data.append(chunk_bytes)
#             # Run pre-tokenization on your chunk and store the counts for each pre-token
        
#             # chunks = chunk_bytes.decode("utf-8", errors="ignore")
#             # train_bpe_chunks(chunks,vocab_size,special_tokens,vocab,merges)
#         with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
#             process_func = partial(process_single_chunk,vocab_size=vocab_size,special_tokens=special_tokens) # 用partial固定部分参数
#             results = pool.map(process_func,[data for data in chunks_data])

#             # 合并结果
#             for chunk_vocab,chunk_merges in results:
#                 vocab.update(chunk_vocab)
#                 merges.extend(chunk_merges)      
        
            
#     return (vocab,merges)

## test usage ##

if __name__ == "__main__":
    ## test multi-processing ##
    import time
    input_path = "tests/fixtures/" +"tinystories_sample.txt"
    start_time = time.time()
    vocab,merges = train_bpe(input_path,vocab_size=500,special_tokens=["<|endoftext|>"])
    end_time = time.time()
    print(f"time consumed:{end_time - start_time}")
    print(vocab)
    print()
    print(merges)

    ## test single process function ##
    # string = "low low low low low lower lower widest widest widest <|endoftext|> newest newest newest newest newest newest"
    # vocab_size=263
    # special_tokens = ["<|endoftext|>"]
    # vocab:dict[int, bytes] = {x+len(special_tokens): bytes([x]) for x in range(256)} 
    # merges:list[tuple[bytes, bytes]] = []
    # for i in range(len(special_tokens)):
    #     vocab[i] = special_tokens[i].encode('utf-8')
    # train_bpe_chunks(string,vocab_size,special_tokens,vocab,merges)
    # print(vocab,merges,sep='\n')
