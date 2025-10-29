from datasets import BlochNPZ
train_data = BlochNPZ("data/npz/train.npz")
print(len(train_data))
sample = train_data[0]
for k, v in sample.items():
    print(k, v.shape)
