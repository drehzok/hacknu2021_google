import pandas as pd
import argparse
from tqdm import tqdm
parser = argparse.ArgumentParser(description='for summary',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--dataset', type=str, default="1")
args = parser.parse_args()

df = pd.read_csv('for_summary/for_sum{}.csv'.format(args.dataset))
batch_size = args.batch_size

from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch

model_name = 'google/pegasus-multi_news'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)

try:
    for i in tqdm(range(0, len(df), batch_size)):
        src_text = df.iloc[i:i+batch_size,10].tolist()
        batch = tokenizer(src_text, truncation=True, padding='longest', return_tensors="pt").to(device)
        translated = model.generate(**batch)
        tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
        df.iloc[i:i+batch_size, 12] = tgt_text
except:
    print('Something went wrong!')
    df.to_csv('summary{}.csv'.format(args.dataset))
print('Gooood!')
df.to_csv('summary{}.csv'.format(args.dataset))
