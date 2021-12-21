
from lib import *
from dataset import *
from model import *

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--path", default='nyt', type=str)
    
    parser.add_argument("--max_len", default=120, type=int)
    parser.add_argument("--num_ne", default=5, type=int)
    parser.add_argument("--num_rel", default=25, type=int)
    
    parser.add_argument("--size_hid", default=256, type=int)
    parser.add_argument("--layer_rnn", default=2, type=int)
    parser.add_argument("--layer_gcn", default=2, type=int)
    parser.add_argument("--dropout", default=0.5, type=float)
    parser.add_argument("--arch", default='2p', type=str)
    
    parser.add_argument("--size_epoch", default=40, type=int)
    parser.add_argument("--size_batch", default=64, type=int)
    parser.add_argument("--lr", default=8e-4, type=float)
    parser.add_argument("--lr_decay", default=0.9, type=float)
    parser.add_argument("--weight_loss", default=2.0, type=float)
    parser.add_argument("--weight_alpha", default=3.0, type=float)
    
    args = parser.parse_args()
    args.path_output = '_snapshot/_%s_%s_%s'%(args.path, args.arch, datetime.now().strftime('%Y%m%d%H%M%S'))
    
    return args

def train_dl(args, model, dl, optzr):
    def get_loss(weight_loss, out, ans):
        out, ans = out.flatten(0, len(out.shape)-2), ans.flatten(0, len(ans.shape)-1).cuda()
        ls = T.nn.functional.cross_entropy(out, ans, ignore_index=-1, reduction='none')
        weight = 1.0-(ans==-1).float()
        weight.masked_fill_(ans>0, weight_loss)
        ls = (ls*weight).sum() / (weight>0).sum()
        return ls

    ret = {'ls_ne': [], 'ls_rel': []}
    for s, inp_sent, inp_pos, dep_fw, dep_bw, ans_ne, ans_rel in tqdm(dl, ascii=True):
        if args.arch=='1p':
            out_ne, out_rel = model(inp_sent.cuda(), inp_pos.cuda(), dep_fw.cuda(), dep_bw.cuda())
            ls_ne, ls_rel = get_loss(args.weight_loss, out_ne, ans_ne), get_loss(args.weight_loss, out_rel, ans_rel)
            ls = ls_ne + args.weight_alpha*ls_rel
        
        elif args.arch=='2p':
            out_ne1p, out_rel1p, out_ne2p, out_rel2p = model(inp_sent.cuda(), inp_pos.cuda(), dep_fw.cuda(), dep_bw.cuda())
            ls_ne1p, ls_rel1p = get_loss(args.weight_loss, out_ne1p, ans_ne), get_loss(args.weight_loss, out_rel1p, ans_rel)
            ls_ne2p, ls_rel2p = get_loss(args.weight_loss, out_ne2p, ans_ne), get_loss(args.weight_loss, out_rel2p, ans_rel)
            ls_ne, ls_rel = ls_ne2p, ls_rel2p
            ls = (ls_ne1p+ls_ne2p) + args.weight_alpha*(ls_rel1p+ls_rel2p)
        
        optzr.zero_grad()
        ls.backward()
        optzr.step()
        ret['ls_ne'].append(ls_ne.item()), ret['ls_rel'].append(ls_rel.item())
    ret = {k: float(np.average(l)) for k, l in ret.items()}
    
    return ret

def eval_dl(model, dl):
    ret = {'precision': [0, 0], 'recall': [0, 0], 'f1': 0}
    
    I = 0
    for s, inp_sent, inp_pos, dep_fw, dep_bw, ans_ne, ans_rel in tqdm(dl, ascii=True):
        if args.arch=='1p':
            out_ne, out_rel = model(inp_sent.cuda(), inp_pos.cuda(), dep_fw.cuda(), dep_bw.cuda())
        elif args.arch=='2p':
            _, _, out_ne, out_rel = model(inp_sent.cuda(), inp_pos.cuda(), dep_fw.cuda(), dep_bw.cuda())
        
        out_ne, out_rel = [T.argmax(out, dim=-1).data.cpu().numpy() for out in [out_ne, out_rel]]
        for o_ne, o_rel in zip(out_ne, out_rel):
            l = len(dl.dataset.dat[I]['sentence'])+1
            
            ne, pos = {}, -1
            for i in range(l):
                v = o_ne[i]
                if v==4:
                    ne[i] = [i, i]
                    pos = -1
                elif v==1:
                    pos = i
                elif v==2:
                    pass
                elif v==3:
                    if pos!=-1:
                        for p in range(pos, i+1):
                            ne[p] = [pos, i]
                elif v==0:
                    pos = -1
            
            pd = set()
            for i in range(l):
                for j in range(l):
                    if o_rel[i][j]!=0 and i in ne and j in ne:
                        pd.add((ne[i][1], ne[j][1], o_rel[i][j]))
            
            gt = set()
            for ne1, ne2, rel in dl.dataset.dat[I]['label']:
                gt.add((ne1[1], ne2[1], rel))
            
            ret['precision'][0] += len(pd.intersection(gt))
            ret['precision'][1] += len(pd)
            ret['recall'][0] += len(pd.intersection(gt))
            ret['recall'][1] += len(gt)
            
            I += 1
    
    ret['precision'] = ret['precision'][0]/ret['precision'][1] if ret['precision'][1]>0 else 0
    ret['recall'] = ret['recall'][0]/ret['recall'][1] if ret['recall'][1]>0 else 0
    ret['f1'] = 2*ret['precision']*ret['recall']/(ret['precision']+ret['recall']) if (ret['precision']+ret['recall'])>0 else 0
    
    return ret

if __name__=='__main__':
    args = get_args()
    os.makedirs(args.path_output, exist_ok=True)
    json.dump(vars(args), open('%s/args.json'%(args.path_output), 'w'), indent=2)
    print(args)
    
    NLP = spacy.load('en_core_web_lg')
    ds_tr, ds_vl, ds_ts = [DS(NLP, args.path, typ, args.max_len) for typ in ['train', 'val', 'test']]
    dl_tr, dl_vl, dl_ts = [T.utils.data.DataLoader(ds, batch_size=args.size_batch, 
                                                   shuffle=(ds is ds_tr), num_workers=32, pin_memory=True) \
                           for ds in [ds_tr, ds_vl, ds_ts]]
    
    log = {'ls_tr': [], 'f1_vl': [], 'f1_ts': []}
    json.dump(log, open('%s/log.json'%(args.path_output), 'w'), indent=2)
    
    model = GraphRel(len(ds_tr.POS)+1, args.num_ne, args.num_rel, 
                     args.size_hid, args.layer_rnn, args.layer_gcn, args.dropout, 
                     args.arch).cuda()
    T.save(model.state_dict(), '%s/model_0.pt'%(args.path_output))
    
    optzr = T.optim.AdamW(model.parameters(), lr=args.lr)
    for e in tqdm(range(args.size_epoch), ascii=True):
        model.train()
        ls_tr = train_dl(args, model, dl_tr, optzr)
        
        model.eval()
        f1_vl = eval_dl(model, dl_vl)
        f1_ts = eval_dl(model, dl_ts)
        
        log['ls_tr'].append(ls_tr), log['f1_vl'].append(f1_vl), log['f1_ts'].append(f1_ts)
        json.dump(log, open('%s/log.json'%(args.path_output), 'w'), indent=2)
        T.save(model.state_dict(), '%s/model_%d.pt'%(args.path_output, e+1))
        print('Ep %d:'%(e+1), ls_tr, f1_vl, f1_ts)
        
        for pg in optzr.param_groups:
            pg['lr'] *= args.lr_decay
        
