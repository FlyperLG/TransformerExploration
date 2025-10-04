import torch
from transformer import Transformer
from data_generator import generate_random_data, batchify_data
from model_trainer import fit, predict

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

model = Transformer(num_tokens=4, dim_model=16, num_heads=2, num_encoder_layers=6, num_decoder_layers=6, dropout_p=0.1).to(device)
opt = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = torch.nn.CrossEntropyLoss()

train_data = generate_random_data(3000)
val_data = generate_random_data(300)

train_dataloader = batchify_data(train_data)
val_dataloader = batchify_data(val_data)

fit(model, opt, loss_fn, train_dataloader, val_dataloader, epochs=10, device=device)

examples = [
    torch.tensor([[2, 0, 0, 0, 0, 0, 0, 0, 0, 3]], dtype=torch.long, device=device),
    torch.tensor([[2, 1, 1, 1, 1, 1, 1, 1, 1, 3]], dtype=torch.long, device=device),
    torch.tensor([[2, 1, 0, 1, 0, 1, 0, 1, 0, 3]], dtype=torch.long, device=device),
    torch.tensor([[2, 0, 1, 0, 1, 0, 1, 0, 1, 3]], dtype=torch.long, device=device),
    torch.tensor([[2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 3]], dtype=torch.long, device=device),
    torch.tensor([[2, 0, 1, 3]], dtype=torch.long, device=device)
]

for idx, example in enumerate(examples):
    result = predict(model, example, device)
    print(f"Example {idx}")
    print(f"Input: {example.view(-1).tolist()[1:-1]}")
    print(f"Continuation: {result[1:-1]}")
    print()