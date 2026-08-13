import torch
import torch.nn as nn


class SparseMultiheadSelfAttention(nn.Module):
    """Memory-efficient local + strided-global self-attention.

    Ordinary queries attend to a local radius, CLS, strided global keys, and variant keys.
    CLS, strided-global queries, and variant-overlapping queries additionally attend to all keys.
    Query chunking avoids materializing a T x T attention matrix for every query.
    """
    def __init__(self, dim, heads=8, dropout=0.2, local_radius=64, global_stride=32, query_chunk=16):
        super().__init__(); assert dim % heads == 0
        self.dim=dim; self.heads=heads; self.head_dim=dim//heads; self.scale=self.head_dim**-0.5
        self.qkv=nn.Linear(dim,dim*3); self.out=nn.Linear(dim,dim); self.dropout=nn.Dropout(dropout)
        self.local_radius=int(local_radius); self.global_stride=int(global_stride); self.query_chunk=int(query_chunk)
        self._cache={}

    def _base_indices(self,T,device):
        key=(T,str(device))
        if key in self._cache: return self._cache[key]
        globals_ = list(range(0,T,self.global_stride))
        rows=[]; maxk=0
        for q in range(T):
            s=set(globals_); s.add(0)
            lo=max(0,q-self.local_radius); hi=min(T,q+self.local_radius+1); s.update(range(lo,hi))
            row=sorted(s); rows.append(row); maxk=max(maxk,len(row))
        idx=torch.empty((T,maxk),dtype=torch.long,device=device)
        valid=torch.zeros((T,maxk),dtype=torch.bool,device=device)
        for q,row in enumerate(rows):
            n=len(row); idx[q,:n]=torch.tensor(row,device=device); idx[q,n:]=q; valid[q,:n]=True
        gq=torch.tensor(globals_,dtype=torch.long,device=device)
        self._cache[key]=(idx,valid,gq)
        return idx,valid,gq

    def forward(self,x,variant_mask=None,return_attention=False):
        B,T,D=x.shape
        qkv=self.qkv(x).reshape(B,T,3,self.heads,self.head_dim).permute(2,0,3,1,4)
        q,k,v=qkv[0],qkv[1],qkv[2]  # B,H,T,d
        base_idx,base_valid,strided_q=self._base_indices(T,x.device)
        if variant_mask is None:
            variant_mask=torch.zeros((B,T-1),dtype=torch.bool,device=x.device)
        maxv=max(1,int(variant_mask.sum(1).max().item()))
        var_idx=torch.zeros((B,maxv),dtype=torch.long,device=x.device); var_valid=torch.zeros((B,maxv),dtype=torch.bool,device=x.device)
        for b in range(B):
            vp=torch.where(variant_mask[b])[0]+1; n=min(maxv,vp.numel())
            if n: var_idx[b,:n]=vp[:n]; var_valid[b,:n]=True
        outputs=[]
        attn_summary=torch.zeros((B,self.heads,T,T),device=x.device,dtype=x.dtype) if return_attention and B<=2 and T<=4096 else None
        # sparse attention for ordinary queries, processed in chunks
        for qs in range(0,T,self.query_chunk):
            qe=min(T,qs+self.query_chunk); C=qe-qs
            bi=base_idx[qs:qe]  # C,Kb
            bv=base_valid[qs:qe]
            bi=bi[None,:,:].expand(B,-1,-1); bv=bv[None,:,:].expand(B,-1,-1)
            vi=var_idx[:,None,:].expand(-1,C,-1); vv=var_valid[:,None,:].expand(-1,C,-1)
            inds=torch.cat([bi,vi],dim=-1); valid=torch.cat([bv,vv],dim=-1)
            Kc=inds.size(-1)
            # Gather by batch/head/query; output B,H,C,Kc,d
            bidx=torch.arange(B,device=x.device)[:,None,None,None]
            hidx=torch.arange(self.heads,device=x.device)[None,:,None,None]
            keyidx=inds[:,None,:,:].expand(-1,self.heads,-1,-1)
            kg=k[bidx,hidx,keyidx,:]; vg=v[bidx,hidx,keyidx,:]
            qc=q[:,:,qs:qe,:].unsqueeze(-2)
            scores=(qc*kg).sum(-1)*self.scale
            scores=scores.masked_fill(~valid[:,None,:,:],torch.finfo(scores.dtype).min)
            a=torch.softmax(scores,dim=-1); a=self.dropout(a)
            ctx=(a.unsqueeze(-1)*vg).sum(-2)
            outputs.append(ctx)
            if attn_summary is not None:
                for b in range(B):
                    for ci in range(C):
                        attn_summary[b,:,qs+ci,:].scatter_add_(1,inds[b,ci][None,:].expand(self.heads,-1),a[b,:,ci,:])
        context=torch.cat(outputs,dim=2)  # B,H,T,d
        # Global queries attend to every key: CLS + strided + variant positions.
        for b in range(B):
            g=torch.unique(torch.cat([strided_q,torch.tensor([0],device=x.device),torch.where(variant_mask[b])[0]+1]))
            if g.numel()==0: continue
            qg=q[b,:,g,:]  # H,G,d
            scores=torch.matmul(qg,k[b].transpose(-2,-1))*self.scale  # H,G,T
            a=self.dropout(torch.softmax(scores,dim=-1)); cg=torch.matmul(a,v[b])
            context[b,:,g,:]=cg
            if attn_summary is not None: attn_summary[b,:,g,:]=a
        y=context.transpose(1,2).reshape(B,T,D)
        y=self.out(y)
        return y,attn_summary
