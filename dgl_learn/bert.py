from transformers import AutoModel,AutoTokenizer

config = 'bert-base-uncased'

tokenizer = AutoTokenizer.from_pretrained(config)
model = AutoModel.from_pretrained(config)

texts = ['Henry Ford Was a sexy man','हेनरी फोर्ड एक चतुर व्यक्ति थे']
max_length = 128
inputs = tokenizer(texts, max_length=max_length, truncation=True, padding='max_length', return_tensors='pt')
outputs = model(inputs.input_ids, inputs.attention_mask)
print(outputs.last_hidden_state[0][:, 0].shape)