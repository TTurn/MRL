from sentence_transformers import SentenceTransformer, InputExample, losses
#data function
def read_chatbot_csv(chatbot_csv):
    samples = []
    with open(chatbot_csv, 'r') as f:
        for line in f:
            line = line.strip('\n')
            line_s = line.split("\t")
            if len(line_s) != 3:
                continue
            query, key, value = line_s
            if value not in ['0', '1']:
                continue
            value = float(value)
            samples.append(InputExample(texts=[query, key], label=value))
    return samples

# load data
path = "./data/"
train_csv = path + 'train.tsv'
dev_csv = path + 'dev.tsv'
test_csv = path + 'test.tsv'
train_samples = []
dev_samples = []
test_samples = []
train_samples = read_chatbot_csv(train_csv)
dev_samples = read_chatbot_csv(dev_csv)
test_samples = read_chatbot_csv(test_csv)
print(len(train_samples))
print(len(dev_samples))
print(len(test_samples))

#build model
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import CoSENTLoss, MatryoshkaLoss
from torch.utils.data import DataLoader
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from datetime import datetime
import math

model = SentenceTransformer("tuhailong/PairSupCon-roberta-wwm-ext", device="cuda:1")
#train_loss = losses.MultipleNegativesRankingLoss(model=model)
train_loss = CoSENTLoss(model)
train_loss = losses.Matryoshka2dLoss(model, train_loss, [768, 512, 256, 128, 64])

evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='dev')
model_save_path = 'm2d_model/m2d_'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
num_epochs = 5
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=16)
warmup_steps = warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=model_save_path)

