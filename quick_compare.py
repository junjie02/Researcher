# quick_compare.py
import torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from verl.utils.dataset.sft_dataset import SFTDataset
from tqdm import tqdm 

@torch.no_grad()
def eval_loss(model_path, test_json='./o2searcher/data/coldstart/test.json'):
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2',
    ).cuda().eval()
    ds = SFTDataset(json_files=test_json, tokenizer=tok,
                    max_length=10240, truncation='right')

    total_loss, total_tokens = 0.0, 0
    for i in tqdm(range(len(ds))):
        item = ds[i]
        ids  = item['input_ids'].unsqueeze(0).cuda()
        attn = item['attention_mask'].unsqueeze(0).cuda()
        pos  = item['position_ids'].unsqueeze(0).cuda()
        mask = item['loss_mask'].cuda()
        labels = ids[:, 1:].contiguous()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out = model(input_ids=ids, attention_mask=attn, position_ids=pos, use_cache=False)
        logits = out.logits[..., :-1, :].contiguous()
        loss = nn.CrossEntropyLoss(reduction='none')(
            logits.view(-1, model.config.vocab_size), labels.view(-1))
        loss = loss * mask[:-1]
        total_loss   += loss.sum().item()
        total_tokens += mask[:-1].sum().item()

    print(f'{model_path}:  avg_loss = {total_loss/total_tokens:.4f}  (valid_tokens={total_tokens})')
    del model
    torch.cuda.empty_cache()
    return total_loss / total_tokens

eval_loss('checkpoints/o2searcher/coldstart/global_step_306')
eval_loss('checkpoints/o2searcher/coldstart/global_step_612')
