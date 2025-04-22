import base64
import io
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import torch

from vlm.vlm import VisionLanguageModel


batch_size = 16  # how many independent sequences will we process in parallel?
block_size = 32  # what is the maximum context length for predictions?
max_iters = 100
eval_interval = 10
learning_rate = 1e-3
epochs = 1
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 40
num_blks = 3
head_size = 16
n_embd = 128
n_head = 8
n_layer = 8
dropout = 0.1
img_size = 96
patch_size = 16
image_embed_dim = 512
emb_dropout = blk_dropout = 0.1

# Ensure every computation happens on the GPU when available
device = "cuda" if torch.cuda.is_available() else "cpu"

# To build the encoding and decoding functions we use the tinyshakespear dataset. However for the sake of brevity we do not pretrain the decoder model on it
# the training function should be able to do it without an issue as well as it could take both images and text
text_path = "./input.txt"
with open(text_path, "r", encoding="utf-8") as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
stoi["<pad>"] = 65
itos = {i: ch for i, ch in enumerate(chars)}
itos[65] = "<pad>"
encode = lambda s: [
    stoi[c] for c in s
]  # encoder: take a string, output a list of integers
decode = lambda l: "".join(
    [itos[i] for i in l]
)  # decoder: take a list of integers, output a string
vocab_size = len(stoi.keys())

def base64_to_tensor(base64_str, img_size=96):
    image = Image.open(io.BytesIO(base64.b64decode(base64_str)))
    if image.mode != "RGB":
        image = image.convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(image).unsqueeze(0)  # Add batch dimension


# Adjusting the data loader from makemore for multimodal data
def get_batch(df, batch_size, split="train", img_size=96, val_batch_size=8):
    # Split data into training and validation sets
    n = int(0.9 * len(df))  # first 90% will be train, rest val
    df_train = df.iloc[:n]
    df_val = df.iloc[n:]
    data = df_train if split == "train" else df_val
    batch_size = batch_size if split == "train" else val_batch_size
    replace = False if split == "train" else True
    batch = data.sample(n=batch_size, replace=replace)

    images = torch.cat(
        [base64_to_tensor(img, img_size) for img in batch["b64string_images"]], dim=0
    ).to(device)
    text_indices = [
        torch.tensor(encode(desc), dtype=torch.long) for desc in batch["caption"]
    ]
    max_length = max(len(t) for t in text_indices)

    padded_text = torch.full(
        (batch_size, max_length), fill_value=stoi["<pad>"], dtype=torch.long
    ).to(device)
    for i, text in enumerate(text_indices):
        padded_text[i, : len(text)] = text

    targets = torch.cat(
        [
            padded_text[:, 1:],
            torch.full(
                (batch_size, 1),
                fill_value=stoi["<pad>"],
                dtype=torch.long,
                device=device,
            ),
        ],
        dim=1,
    )

    # Truncate or pad targets to match the length of padded_text
    if targets.size(1) > padded_text.size(1):
        targets = targets[:, : padded_text.size(1)]
    elif targets.size(1) < padded_text.size(1):
        targets = torch.cat(
            [
                targets,
                torch.full(
                    (batch_size, padded_text.size(1) - targets.size(1)),
                    fill_value=stoi["<pad>"],
                    dtype=torch.long,
                    device=device,
                ),
            ],
            dim=1,
        )

    return images, padded_text, targets


# Adjusting the training loop from makemore for multimodal data
def train_model(model, df, epochs, vocab_size, img_size=96):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)
    for epoch in range(epochs):
        model.train()
        for _ in range(max_iters):
            images, idx, targets = get_batch(df, batch_size, "train", img_size)
            optimizer.zero_grad()
            logits, loss = model(images, idx, targets)
            loss.backward()
            optimizer.step()
            if _ % eval_interval == 0:
                print(f"Loss at iteration {_}: {loss.item()}")
        val_loss = estimate_loss(model, df, "val", img_size, val_batch_size=8)
        print(f"Validation Loss after epoch {epoch}: {val_loss}")


def estimate_loss(model, df, split, img_size=96, val_batch_size=8):
    losses = []
    model.eval()
    for _ in range(eval_iters):
        images, idx, targets = get_batch(
            df, batch_size, split, img_size, val_batch_size=val_batch_size
        )
        _, loss = model(images, idx, targets)
        losses.append(loss.item())
    return sum(losses) / len(losses)


def main():
    # Load the dataset
    df = pd.read_csv("./inputs.csv")
    # Expanding dataframe so that there's enough data to test. This is just duplicating data. A real dataset would have more rows
    df = pd.concat([df] * 30)[["b64string_images", "caption"]]

    # Initialize the model
    model = VisionLanguageModel(
        n_embd,
        image_embed_dim,
        vocab_size,
        n_layer,
        img_size,
        patch_size,
        n_head,
        num_blks,
        emb_dropout,
        blk_dropout,
    )
    model.to(device)

    # Dummy data to initialize lazy modules
    dummy_img = torch.randn(1, 3, img_size, img_size).to(device)
    dummy_idx = torch.randint(0, vocab_size, (1, block_size)).to(device)
    model(dummy_img, dummy_idx)  # Forward pass to initialize all parameters

    # Train the model
    train_model(model, df, epochs, vocab_size, img_size)


if __name__ == "__main__":
    main()
