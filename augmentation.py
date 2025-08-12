from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import torch.nn as nn
import numpy as np
import spacy
import itertools
from vncorenlp import VnCoreNLP

MODEL_NAME = "bert-base-multilingual-cased"
NLP = spacy.load("en_core_web_sm")

class Dictionary_base_Augmentation(nn.Module):
    def __init__(self, model_name: str, batch_size: int, threshold: float, align_layer: int):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.rdrsegmenter = VnCoreNLP("./VnCoreNLP/VnCoreNLP-1.1.1.jar", annotators="wseg,pos", max_heap_size='-Xmx500m')
        self.batch_size = batch_size
        self.threshold = threshold
        self.align_layer = align_layer
    
    def get_batch_embeddings(self, sentences: list[str]) -> torch.Tensor:
        embeddings = []
        for i in range(len(sentences)):
            batch = sentences[i:i+self.batch_size]
            inputs = self.tokenizer(batch, return_tensors='pt').to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            batch_embeddings = outputs.pooler_output.numpy() # Access pooled output (from [CLS])
            embeddings.append(batch_embeddings)
        return torch.cat(embeddings, dim=0)
    
    def get_topN_src_sentences(self, src_ood_parallel_corpus_embeddings, src_ind_dic_embeddings, N: int) -> list[list[int]]:
        index = faiss.IndexFlatIP(src_ood_parallel_corpus_embeddings.shape[-1])
        index.add(src_ood_parallel_corpus_embeddings)
        faiss.normalize_L2(src_ood_parallel_corpus_embeddings) # Normalize L2
        faiss.normalize_L2(src_ind_dic_embeddings) # Normalize L2
        _, indices = index.search(src_ind_dic_embeddings, k=N)
        return indices.tolist()
            
    def extract_en_noun_phrases(self, sentence: str) -> list[str]:
        doc_sen = NLP(sentence)
        phrases = [chunk.text for chunk in doc_sen.noun_chunks]
        return phrases
    
    def extract_vi_noun_phraes(self, sentence: str) -> list[str]:
        pos_tagging = self.rdrsegmenter.pos_tag(sentence)[0]
        noun_tags = ['N', 'Np', 'Nc', 'Nu', 'Ny', 'Nb']
        prev_N = False
        noun_phrases = []
        for word, tag in pos_tagging:
            if tag in noun_tags:
                if prev_N:
                    noun_phrases[-1] += " " + word
                else:
                    noun_phrases.append(word)
                prev_N = True
            else:
                prev_N = False
        return [phrase.replace("_", " ") for phrase in noun_phrases]
    
    def align(self, src_phrases: list[str], tgt_phrases: list[str]):
        token_src, token_tgt = [self.tokenizer.tokenize(word) for word in src_phrases], [self.tokenizer.tokenize(word) for word in tgt_phrases]
        wid_src = [self.tokenizer.convert_tokens_to_ids(x) for x in token_src]
        wid_tgt = [self.tokenizer.convert_tokens_to_ids(x) for x in token_tgt]
        ids_src = self.tokenizer.prepare_for_model(
            list(itertools.chain(*wid_src)),
            return_tensors='pt',
            model_max_length=self.tokenizer.model_max_length,
            truncation=True
        )['input_ids']
        ids_tgt = self.tokenizer.prepare_for_model(
            list(itertools.chain(*wid_tgt)),
            return_tensors='pt',
            model_max_length=self.tokenizer.model_max_length,
            truncation=True
        )['input_ids']
        
        sub2word_map_src = []
        for i, word_list in enumerate(token_src):
            sub2word_map_src += [i for x in word_list]
            
        sub2word_map_tgt = []
        for i, word_list in enumerate(token_tgt):
            sub2word_map_tgt += [i for x in word_list]
            
        self.model.eval()
        with torch.no_grad():
            out_src = self.model(ids_src.unsqueeze(0), output_hidden_states=True)[2][self.align_layer][0, 1:-1]
            out_tgt = self.model(ids_tgt.unsqueeze(0), output_hidden_states=True)[2][self.align_layer][0, 1:-1]
            dot_prod = torch.matmul(out_src, out_tgt.transpose(-1, -2))
            softmax_srctgt = nn.Softmax(dim=-1)(dot_prod)
            softmax_tgtsrc = nn.Softmax(dim=-2)(dot_prod)
            softmax_inter = (softmax_srctgt > self.threshold) * (softmax_tgtsrc > self.threshold)
        
        align_subwords = torch.nonzero(softmax_inter, as_tuple=False)
        mapping_align_words = {}
        for i, j in align_subwords:
            if src_phrases[sub2word_map_src[i]] not in mapping_align_words:
                mapping_align_words[src_phrases[sub2word_map_src[i]]] = tgt_phrases[sub2word_map_tgt[j]]
            elif tgt_phrases[sub2word_map_tgt[j]] != mapping_align_words[src_phrases[sub2word_map_src[i]]]:
                mapping_align_words[src_phrases[sub2word_map_src[i]]] += " " + tgt_phrases[sub2word_map_tgt[j]]
        return mapping_align_words
    
    def substitute(self, ood_sentence: str, phrase_max_ood: str, phrase_ind: str) -> str:
        return ood_sentence.replace(phrase_max_ood, phrase_ind)
        
    def forward(self, ood_parallel_corpus, ind_dic, N):
        src_odd_parallel_corpus, tgt_odd_parallel_corpus = [], []
        for S, T in ood_parallel_corpus:
            src_odd_parallel_corpus.append(S)
            tgt_odd_parallel_corpus.append(T)
        src_ind_dic, tgt_ind_dic = [], []
        for src, tgt in ind_dic:
            src_ind_dic.append(src)
            tgt_ind_dic.append(tgt)
            
        src_ood_parallel_corpus_embeddings = self.get_batch_embeddings(src_odd_parallel_corpus)
        src_ind_dic_embeddings = self.get_batch_embeddings(src_ind_dic)
        src_ood_topN_sen_corpus_indices = self.get_topN_src_sentences(src_ood_parallel_corpus_embeddings, src_ind_dic_embeddings, N)
        assert len(src_ood_topN_sen_corpus_indices) == src_ind_dic_embeddings.shape[0], "Not Suitable Size"
        Gc = []
        for i in range(src_ind_dic_embeddings.shape[0]):
            src_embedding = src_ind_dic_embeddings[i, :]
            for id_sen in src_ood_topN_sen_corpus_indices[i]:
                phrases_Sk = self.extract_en_noun_phrases(src_odd_parallel_corpus[id_sen])
                phrases_Tk = self.extract_vi_noun_phraes(tgt_odd_parallel_corpus[id_sen])
                mapping_align_words = self.align(phrases_Sk, phrases_Tk)
                
                phrases_embeddings = self.get_batch_embeddings(phrases_Sk)
                index = faiss.IndexFlatIP(phrases_embeddings.shape[-1])
                index.add(phrases_embeddings)
                faiss.normalize_L2(phrases_embeddings)
                _, max_sim_index = index.search(src_embedding, k=1)
                phrase_max_Sk = phrases_Sk[max_sim_index]
                phrase_max_Tk = mapping_align_words[phrase_max_Sk]
                
                Gc.append(
                    (self.substitute(src_odd_parallel_corpus[id_sen], phrase_max_Sk, src_ind_dic[i]), 
                    self.substitute(tgt_odd_parallel_corpus[id_sen], phrase_max_Tk, tgt_ind_dic[i]))
                )
        
        return Gc      