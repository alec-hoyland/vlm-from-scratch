import torch
import torch.nn as nn
from block import Block
from multimodal_projector import MultiModalProjector
from torch.nn import functional as F
from vit import ViT


class DecoderLanguageModel(nn.Module):
    def __init__(
        self, n_embd, image_embed_dim, vocab_size, num_heads, n_layer, use_images=False
    ):
        super().__init__()

        self.use_images = use_images

        # Token embedding table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        # Position embedding table
        self.position_embedding_table = nn.Embedding(1000, n_embd)

        if use_images:
            # Image projection layer to align image embeddings with text embeddings
            self.image_projection = MultiModalProjector(n_embd, image_embed_dim)

        # Stack of transformer decoder blocks
        self.blocks = nn.Sequential(
            *[Block(n_embd, num_heads, is_decoder=True) for _ in range(n_layer)]
        )

        # Final layer normalization
        self.ln_f = nn.LayerNorm(n_embd)

        # Language modeling head
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, image_embeds=None, targets=None):
        # Get token embeddings from the input indices
        tok_emb = self.token_embedding_table(idx)

        if self.use_images and image_embeds is not None:
            # Project and concatenate image embeddings with token embeddings
            img_emb = self.image_projection(image_embeds).unsqueeze(1)
            tok_emb = torch.cat([img_emb, tok_emb], dim=1)

        # Get position embeddings
        pos_emb = self.position_embedding_table(
            torch.arange(tok_emb.size(1), device=self.device)
        ).unsqueeze(0)

        # Add position embeddings to token embeddings
        x = tok_emb + pos_emb

        # Pass through the transformer decoder blocks
        x = self.blocks(x)

        # Apply final layer normalization
        x = self.ln_f(x)

        # Get the logits from the language modeling head
        logits = self.lm_head(x)

        if targets is not None:
            if self.use_images and image_embeds is not None:
                # Prepare targets by concatenating a dummy target for the image embedding
                batch_size = idx.size(0)
                targets = torch.cat(
                    [
                        torch.full(
                            (batch_size, 1), -100, dtype=torch.long, device=self.device
                        ),
                        targets,
                    ],
                    dim=1,
                )

            # Compute the cross-entropy loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100
            )
            return logits, loss

        return logits

    def generate(self, idx, image_embeds, max_new_tokens):
        generated = idx

        if self.use_images and image_embeds is not None:
            img_emb = self.image_projection(image_embeds).unsqueeze(1)
            current_output = torch.cat(
                [img_emb, self.token_embedding_table(idx)], dim=1
            )
        else:
            current_output = self.token_embedding_table(idx)

        for i in range(max_new_tokens):
            T_current = current_output.size(1)
            current_pos_emb = self.position_embedding_table(
                torch.arange(T_current, device=self.device)
            ).unsqueeze(0)
            current_output += current_pos_emb

            for block in self.blocks:
                current_output = block(current_output)

            logits = self.lm_head(current_output[:, -1, :])
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            generated = torch.cat((generated, idx_next), dim=1)
            idx_next_emb = self.token_embedding_table(idx_next)
            current_output = torch.cat((current_output, idx_next_emb), dim=1)

        return generated


class VisionLanguageModel(nn.Module):
    def __init__(
        self,
        n_embd,
        image_embed_dim,
        vocab_size,
        n_layer,
        img_size,
        patch_size,
        num_heads,
        num_blks,
        emb_dropout,
        blk_dropout,
    ):
        super().__init__()

        # Set num_hiddens equal to image_embed_dim
        num_hiddens = image_embed_dim

        # Assert that num_hiddens is divisible by num_heads
        assert num_hiddens % num_heads == 0, (
            "num_hiddens must be divisible by num_heads"
        )

        # Initialize the vision encoder (ViT)
        self.vision_encoder = ViT(
            img_size,
            patch_size,
            num_hiddens,
            num_heads,
            num_blks,
            emb_dropout,
            blk_dropout,
        )

        # Initialize the language model decoder (DecoderLanguageModel)
        self.decoder = DecoderLanguageModel(
            n_embd, image_embed_dim, vocab_size, num_heads, n_layer, use_images=True
        )

    def forward(self, img_array, idx, targets=None):
        # Get the image embeddings from the vision encoder
        image_embeds = self.vision_encoder(img_array)

        # Check if the image embeddings are valid
        if image_embeds.nelement() == 0 or image_embeds.shape[1] == 0:
            raise ValueError()

        if targets is not None:
            # If targets are provided, compute the logits and loss
            logits, loss = self.decoder(idx, image_embeds, targets)
            return logits, loss
        else:
            # If targets are not provided, compute only the logits
            logits = self.decoder(idx, image_embeds)
            return logits

    def generate(self, img_array, idx, max_new_tokens):
        # Get the image embeddings from the vision encoder
        image_embeds = self.vision_encoder(img_array)

        # Check if the image embeddings are valid
        if image_embeds.nelement() == 0 or image_embeds.shape[1] == 0:
            raise ValueError()

        # Generate new tokens using the language model decoder
        generated_tokens = self.decoder.generate(idx, image_embeds, max_new_tokens)
        return generated_tokens
