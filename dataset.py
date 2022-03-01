
from lib import *

class DS(T.utils.data.Dataset):
    def __init__(self, NLP, path, typ, max_len):
        super().__init__()
        
        self.NLP = NLP
        self.dat = json.load(open('./_data/%s.json'%(path), 'r'))[typ]
        self.max_len = max_len
        
        self.POS = {}
        for p in self.NLP.pipe_labels['tagger']:
            self.POS[p] = len(self.POS)+1
        
    def __len__(self):
        return len(self.dat)
    
    def __getitem__(self, idx):
        item = self.dat[idx]
        sent, label = item['sentence'], item['label']
        
        s = ' '.join(sent)
        inp_sent, inp_pos = np.zeros((self.max_len, 300), dtype=np.float32), np.zeros((self.max_len, ), dtype=np.int64)
        dep_fw, dep_bw = np.zeros((self.max_len, self.max_len), dtype=np.float32), np.zeros((self.max_len, self.max_len), dtype=np.float32)
        ans_ne, ans_rel = np.ones((self.max_len, ), dtype=np.int64)*-1, np.ones((self.max_len, self.max_len), dtype=np.int64)*-1
        
        res = self.NLP(s)
        for i in range(len(res)):
            ans_ne[i] = 0
            for j in range(len(res)):
                ans_rel[i][j] = 0
        
        for i, w in enumerate(res):
            inp_sent[i], inp_pos[i] = w.vector, self.POS[w.tag_]
            
            dep_fw[i][i], dep_bw[i][i] = 1, 1
            for c in res[i].children:
                for j, t in enumerate(res):
                    if c==t:
                        dep_fw[i][j], dep_bw[j][i] = 1, 1
        L = len(res)
        dep_fw[:L], dep_bw[:L] = [dep_fw[:L]/dep_fw[:L].sum(axis=1, keepdims=True), 
                                  dep_bw[:L]/dep_bw[:L].sum(axis=1, keepdims=True)]
        
        for ne1, ne2, rel in label:
            def set_ne(ne):
                b, e = ne
                if b==e:
                    ans_ne[b] = 4 # 'S'
                else:
                    ans_ne[b], ans_ne[e] = 1, 3 # 'B', 'E'
                    ans_ne[b+1:e] = 2 # 'I'
            
            set_ne(ne1), set_ne(ne2)
            ans_rel[ne1[0]:ne1[1]+1, ne2[0]:ne2[1]+1] = rel
        
        return s, inp_sent, inp_pos, dep_fw, dep_bw, ans_ne, ans_rel

if __name__=='__main__':
    NLP = spacy.load('en_core_web_lg')
    ds_tr, ds_vl, ds_ts = [DS(NLP, 'nyt', typ, 120) for typ in ['train', 'val', 'test']]
    
    dl = T.utils.data.DataLoader(ds_tr, batch_size=64, shuffle=True, num_workers=32)
    for s, inp_sent, inp_pos, dep_fw, dep_bw, ans_ne, ans_rel in tqdm(dl, ascii=True):
        print(len(s), inp_sent.shape, inp_pos.shape, dep_fw.shape, dep_bw.shape, ans_ne.shape, ans_rel.shape)
    
