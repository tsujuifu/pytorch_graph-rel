
from lib import *
from dataset import *

class GCN(T.nn.Module):
    def __init__(self, size_hid):
        super().__init__()
        
        self.size_hid = size_hid
        
        self.W = T.nn.Parameter(T.FloatTensor(self.size_hid, self.size_hid//2))
        self.b = T.nn.Parameter(T.FloatTensor(self.size_hid//2, ))
        
        stdv = 1.0/math.sqrt(self.size_hid//2)
        self.W.data.uniform_(-stdv, stdv)
        self.b.data.uniform_(-stdv, stdv)
    
    def forward(self, inp, adj):
        out = T.matmul(inp, self.W) + self.b
        out = T.matmul(adj, out)
        out = T.nn.functional.relu(out)
        
        return out
    
    def __repr__(self):
        return self.__class__.__name__+'(size_hid=%d)'%(self.size_hid)

class GraphRel(T.nn.Module):
    def __init__(self, num_pos, num_ne, num_rel, 
                 size_hid, layer_rnn, layer_gcn, dp, 
                 arch='2p'):
        super().__init__()
        
        self.arch = arch
        
        self.emb_pos = T.nn.Embedding(num_pos, 15)
        
        self.rnn = T.nn.GRU(300+15, size_hid, num_layers=layer_rnn, dropout=dp, 
                            batch_first=True, bidirectional=True)
        self.gcn_fw, self.gcn_bw = [T.nn.ModuleList([GCN(size_hid*2) for _ in range(layer_gcn)]), 
                                    T.nn.ModuleList([GCN(size_hid*2) for _ in range(layer_gcn)])]
        
        self.rnn_ne = T.nn.GRU(size_hid*2, size_hid, batch_first=True)
        self.fc_ne = T.nn.Linear(size_hid, num_ne)
        
        self.fc_rf, self.fc_rb = [T.nn.Sequential(*[T.nn.Linear(size_hid*2, size_hid), T.nn.ReLU()]), 
                                  T.nn.Sequential(*[T.nn.Linear(size_hid*2, size_hid), T.nn.ReLU()])]
        self.fc_rel = T.nn.Linear(size_hid*2, num_rel)
        
        if self.arch=='2p':
            self.gcn2p_fw, self.gcn2p_bw = [T.nn.ModuleList([GCN(size_hid*2) for _ in range(num_rel)]), 
                                            T.nn.ModuleList([GCN(size_hid*2) for _ in range(num_rel)])]
        
        self.dp = T.nn.Dropout(dp)
        
    def head(self, feat):
        feat_ne, _ = self.rnn_ne(feat)
        out_ne = self.fc_ne(feat_ne)
        
        rf, rb = self.fc_rf(feat), self.fc_rb(feat)
        rf, rb = [rf.unsqueeze(2).expand([-1, -1, rf.shape[1], -1]), 
                  rb.unsqueeze(1).expand([-1, rb.shape[1], -1, -1])]
        out_rel = self.fc_rel(T.cat([rf, rb], dim=3))
        
        return out_ne, out_rel
        
    def forward(self, inp_sent, inp_pos, dep_fw, dep_bw):
        inp = T.cat([inp_sent, self.emb_pos(inp_pos)], dim=2)
        inp = self.dp(inp)
        
        feat, _ = self.rnn(inp)
        for gf, gb in zip(self.gcn_fw, self.gcn_bw):
            of, ob = gf(feat, dep_fw), gb(feat, dep_bw)
            feat = self.dp(T.cat([of, ob], dim=2))
            
        out_ne, out_rel = self.head(feat)
        
        if self.arch=='1p':
            return out_ne, out_rel
        
        # 2p
        feat1p, out_ne1p, out_rel1p = feat, out_ne, out_rel
        
        dep_fw = T.nn.functional.softmax(out_rel1p, dim=3)
        dep_bw = dep_fw.transpose(1, 2)
        
        feat2p = feat1p.clone()
        for i, (gf, gb) in enumerate(zip(self.gcn2p_fw, self.gcn2p_bw)):
            of, ob = gf(feat1p, dep_fw[:, :, :, i]), gb(feat1p, dep_bw[:, :, :, i])
            feat2p += self.dp(T.cat([of, ob], dim=2))
        
        out_ne2p, out_rel2p = self.head(feat2p)
        
        return out_ne1p, out_rel1p, out_ne2p, out_rel2p
    
if __name__=='__main__':    
    NLP = spacy.load('en_core_web_lg')
    ds_tr, ds_vl, ds_ts = [DS(NLP, 'nyt', typ, 120) for typ in ['train', 'val', 'test']]
    dl = T.utils.data.DataLoader(ds_tr, batch_size=64, 
                                 shuffle=True, num_workers=32, pin_memory=True)
    
    model = GraphRel(len(ds_tr.POS)+1, 5, 25, 
                     256, 2, 2, 0.5, 
                     '2p').cuda()
    
    for s, inp_sent, inp_pos, dep_fw, dep_bw, ans_ne, ans_rel in tqdm(dl, ascii=True):
        out = model(inp_sent.cuda(), inp_pos.cuda(), dep_fw.cuda(), dep_bw.cuda())
        print([o.shape for o in out])
        