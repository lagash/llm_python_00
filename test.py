import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import convert_slow_tokenizer

# モデル名
model_name = "HachiML/myBit-Llama2-jp-127M-7"

# トークナイザーとモデル
tokenizer = AutoTokenizer.from_pretrained(model_name,add_prefix_space=False)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)

# プロンプト

prompt = '''
以下に、あるタスクを説明する指示があり、それに付随する入力が更なる文脈を提供しています。リクエストを適切に完了するための回答を記述してください。

### 指示:

以下のトピックに関する詳細な情報を提供してください。

### 入力:国内のオススメのワーケーション先を10個くらい教えて

### 応答:
'''


# エンコード
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

# アテンションマスク (シンプルにするためにすべて 1)
attention_mask = torch.ones_like(input_ids)

# input_ids を GPU に移動 (利用可能な場合)
if torch.cuda.is_available():
  input_ids = input_ids.to('cuda')
  attention_mask = attention_mask.to('cuda')

# テキスト生成
outputs = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_length=200,
    # max_new_tokens=200,  # Generate 100 new tokens after the prompt
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.7,
)

# デコード
predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# レスポンスの表示
# print(f"質問: {prompt}")
print(f"{predicted_text}")
