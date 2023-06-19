
import sys
import torch

model_type = sys.argv[1]
model_name_or_path = sys.argv[2]
input_str = sys.argv[3]

class DefaultSetting:
    model_type = model_type
    model_name_or_path = model_name_or_path
    temperature = 1.0
    top_k = 10
    top_p = 1.0
    repetition_penalty = 1.0
    min_length = 0
    max_length = 20
    do_sample = True
    
sett = DefaultSetting()

if sett.model_type == 'pegasus':
    from tokenizers_pegasus import PegasusTokenizer
    tokenizer = PegasusTokenizer.from_pretrained("IDEA-CCNL/Randeng-Pegasus-238M-Chinese")
else:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(sett.model_name_or_path)

if sett.model_type == 'pegasus':
    from transformers import PegasusForConditionalGeneration
    model = PegasusForConditionalGeneration.from_pretrained(sett.model_name_or_path)
elif sett.model_type == 'bart':
    from transformers import BartForConditionalGeneration
    model = BartForConditionalGeneration.from_pretrained(sett.model_name_or_path)
else:
    from transformers import T5ForConditionalGeneration
    model = T5ForConditionalGeneration.from_pretrained(sett.model_name_or_path)

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')  # 使用cpu或者gpu
model.to(device)

input_dict = tokenizer(input_str, max_length=10, return_tensors='pt')

input_dict = {k: v.to(device) for k,v in input_dict.items()}

summary_ids = model.generate(**input_dict, max_length=sett.max_length, do_sample=sett.do_sample, temperature=sett.temperature, top_k=sett.top_k, top_p=sett.top_p, repetition_penalty=sett.repetition_penalty)

print(tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])