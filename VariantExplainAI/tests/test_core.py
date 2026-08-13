import torch
from variant_explain_ai.data.kmer import KmerTokenizer
from variant_explain_ai.models.model import VariantExplainAI
from variant_explain_ai.explainability.alignment import expanded_target_mask


def test_kmer_vocab():
    t=KmerTokenizer(5)
    assert t.vocab_size == 1026
    assert len(t.encode('ACGTACGT')) == 4


def test_model_forward():
    tok=KmerTokenizer(5)
    m=VariantExplainAI(tok.vocab_size,embedding_dim=16,hidden_dim=32,cnn_filters=(16,32),cnn_kernels=(5,3),transformer_layers=1,attention_heads=4,ff_dim=64,local_radius=4,global_stride=8,dropout=0.0)
    B,T=2,32
    ref=torch.randint(2,tok.vocab_size,(B,T)); alt=ref.clone(); alt[:,15]=3
    vm=torch.zeros(B,T,dtype=torch.bool); vm[:,14:17]=True
    meta=torch.tensor([[0,1,1,0],[0,1,1,0]],dtype=torch.float32)
    out=m(ref,alt,vm,meta)
    assert out['logits'].shape == (B,2)
    assert out['hidden'].shape == (B,T+1,32)


def test_target_mask():
    vm=torch.zeros(2,20,dtype=torch.bool); vm[:,10]=True
    t=expanded_target_mask(vm,2)
    assert torch.allclose(t.sum(1),torch.ones(2))
