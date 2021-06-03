import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torchsort import soft_rank
from .vits import vit_tiny, vit_small, vit_base, Head


class ResNetNoPool(torchvision.models.ResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.avgpool
        del self.fc
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


# From https://github.com/facebookresearch/EGG/
class RelaxedEmbedding(nn.Embedding):
    """
    A drop-in replacement for `nn.Embedding` such that it can be used _both_ with Reinforce-based training
    and with Gumbel-Softmax one.
    Important: nn.Linear and nn.Embedding have different initialization strategies, hence replacing nn.Linear with
    `RelaxedEmbedding` might change results.
    >>> emb = RelaxedEmbedding(15, 10)  # vocab size 15, embedding dim 10
    >>> long_query = torch.tensor([[1], [2], [3]]).long()
    >>> long_query.size()
    torch.Size([3, 1])
    >>> emb(long_query).size()
    torch.Size([3, 1, 10])
    >>> float_query = torch.zeros((3, 15)).scatter_(-1, long_query, 1.0).float().unsqueeze(1)
    >>> float_query.size()
    torch.Size([3, 1, 15])
    >>> emb(float_query).size()
    torch.Size([3, 1, 10])
    # make sure it's the same query, one-hot and symbol-id encoded
    >>> (float_query.argmax(dim=-1) == long_query).all().item()
    1
    >>> (emb(float_query) == emb(long_query)).all().item()
    1
    """

    def forward(self, x):
        if isinstance(x, torch.LongTensor) or (
            torch.cuda.is_available() and isinstance(x, torch.cuda.LongTensor)
        ):
            return F.embedding(
                x,
                self.weight,
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            )
        else:
            return torch.matmul(x, self.weight)


class Speaker(nn.Module):
    """
    """
    def __init__(
        self, patch_size=16, image_size=224, 
        hidden_size=768, dropout_rate=0.1, 
        arch='resnet', use_context=True, norm='sort', 
        topk=None
    ):
        super(Speaker, self).__init__()
        self.arch = arch
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.use_context = use_context
        self.norm = norm
        self.topk = topk

        # For patch importance, always use context for ranking
        if arch == 'mobilenet':
            # With mobilenet, we allow only 32x32 patches
            rank_hidden_size = 576
            self.patch_size = 32
            self.patch_rank = torchvision.models.mobilenet_v3_small().features

        elif arch == 'resnet':
            # Use a tiny ResNet-9
            rank_hidden_size = 512
            replace_stride = patch_size==16
            self.patch_rank = ResNetNoPool(
                torchvision.models.resnet.BasicBlock, 
                [1, 1, 1, 1],
                replace_stride_with_dilation=[False, False, replace_stride]
            )

        else:
            # Use a tiny ResNet-9
            if arch.lower() == 'vit_tiny':
                self.patch_rank = vit_tiny(patch_size=patch_size, drop_path_rate=dropout_rate, use_cls=False)
                rank_hidden_size = 192
            elif arch.lower() == 'vit_small':
                self.patch_rank = vit_small(patch_size=patch_size, drop_path_rate=dropout_rate, use_cls=False)
                rank_hidden_size = 384
            elif arch.lower() == 'vit_base':
                self.patch_rank = vit_base(patch_size=patch_size, drop_path_rate=dropout_rate, use_cls=False)
                rank_hidden_size = 768

        self.n_patches = (image_size // patch_size)**2
        self.ranker = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(rank_hidden_size, 1))

        # Symbols may or may not depend on the context
        self.pe_symbol = nn.Parameter(
            torch.zeros(1, self.n_patches, hidden_size))

        if self.use_context:
            self.pe_symbol.requires_grad = False
            if arch.lower() == 'vit_tiny':
                self.patch_symbol = vit_tiny(
                    patch_size=patch_size, drop_path_rate=dropout_rate, use_cls=False)
                symbol_hidden_size = 192
            elif arch.lower() == 'vit_small':
                self.patch_symbol = vit_small(
                    patch_size=patch_size, drop_path_rate=dropout_rate, use_cls=False)
                symbol_hidden_size = 384
            elif arch.lower() == 'vit_base':
                self.patch_symbol = vit_base(
                    patch_size=patch_size, drop_path_rate=dropout_rate, use_cls=False)
                symbol_hidden_size = 768
            else:
                self.patch_symbol = vit_tiny(
                    patch_size=patch_size, drop_path_rate=dropout_rate, use_cls=False)
                symbol_hidden_size = 192
            #     replace_stride = patch_size==16
            #     self.patch_symbol = ResNetNoPool(
            #         torchvision.models.resnet.BasicBlock, 
            #         [1, 1, 1, 1], replace_stride_with_dilation=[False, False, replace_stride])
            #     symbol_hidden_size = 512
            self.lin_symbol = nn.Sequential(nn.Linear(symbol_hidden_size, hidden_size), nn.ReLU())
        else:
            self.patch_symbol = nn.Conv2d(
                in_channels=3,
                out_channels=hidden_size,
                kernel_size=patch_size,
                stride=patch_size
            )
            # self.dropout_symbol = nn.Dropout(dropout_rate)

    def forward(self, x, rank_t=1.0, straight_through=False):
        # First convert the image to symbols
        # [B, 3, 224, 224] => [B, hidden_size, image_size/patch_size, image_size/patch_size]
        symbol_embed = self.patch_symbol(x)
        # if self.arch.lower() not in ['vit_tiny', 'vit_small', 'vit_base'] or not self.use_context:
        if not self.use_context:
            # [B, hidden_size, image_size/patch_size, image_size/patch_size] => [B, hidden_size, n_patches]
            symbol_embed = symbol_embed.flatten(2)
            
            # [B, hidden_size, n_patches] => [B, n_patches, hidden_size]
            symbol_embed = symbol_embed.transpose(-1, -2)

        if self.use_context:
            symbol_embed = self.lin_symbol(symbol_embed)
        else:
            symbol_embed = symbol_embed + self.pe_symbol

        # Convert the image to patch importances
        # [B, 3, 224, 224] => [B, hidden_size, image_size/patch_size, image_size/patch_size]
        rank_embed = self.patch_rank(x)

        if self.arch.lower() not in ['vit_tiny', 'vit_small', 'vit_base']:
            # [B, hidden_size, image_size/patch_size, image_size/patch_size] => [B, hidden_size, n_patches]
            rank_embed = rank_embed.flatten(2)
            # [B, 576, 49] => [B, 49, 576]
            rank_embed = rank_embed.transpose(-1, -2)

        # if not self.use_context:
        #     rank_embed = rank_embed + self.pe_rank

        # [B, n_patches, hidden_size] => [B, n_patches, 1]
        ranks = self.ranker(rank_embed)
        # [B, 49, 1] => [B, 49]
        ranks = ranks.squeeze(2)

        if self.norm == 'sort':
            # Ranking act as normalizing the importance values between [1/n_patches, 1.]
            ranks = soft_rank(ranks.float(), regularization_strength=rank_t)
            ranks /= self.n_patches
        elif self.norm == 'l2':
            ranks = torch.nn.functional.normalize(ranks)
        else:
            ranks = ranks - ranks.min().detach()
            ranks = ranks / ranks.max().detach()

        # selected = gumbel_softmax(ranks, gumbel_t, straight_through)
        if self.topk is None:
            # hack; gumbel softmax doesn't take binary outputs; so we provide x, 1-x for sampling
            ranks = torch.stack([ranks, 1-ranks], dim=2)
            patches = gumbel_softmax(ranks, rank_t, straight_through)
            patches = patches[:, :, 0]
        else:
            patches = straight_through_topk(ranks, self.topk)

        return symbol_embed, patches


class GumbelWrapper(nn.Module):
    """
    Gumbel-Softmax Wrapper for the sender agent. 
    """

    def __init__(
        self, vocab_size=3, max_len=2, hidden_size=768,
        temperature=1.0,
        trainable_temperature=False,
        straight_through=False,
    ):
        super(GumbelWrapper, self).__init__()
        self.straight_through = straight_through

        if not trainable_temperature:
            self.temperature = temperature
        else:
            self.temperature = torch.nn.Parameter(
                torch.tensor([temperature]), requires_grad=True
            )

        self.max_len = max_len
        self.vocab_size = vocab_size
        self.logits = nn.Linear(hidden_size, vocab_size * max_len)

    def forward(self, x, temperature=None):
        T = temperature if temperature is not None else self.temperature
        logits = self.logits(x)
        *dims, _ = logits.shape     # Reshape only the last dimension
        logits = logits.reshape(*dims, self.max_len, self.vocab_size)
        sample = gumbel_softmax(logits, T, self.straight_through)
        return sample


class Listener(nn.Module):
    """
    """
    def __init__(
        self, arch='resnet18', 
        dim=128, patch_size=32, image_size=224, 
        hidden_size=192, num_heads=3, num_layers=12, 
        attn_dropout=0., vocab_size=128, max_len=1):

        super(Listener, self).__init__()

        n_patches = (image_size // patch_size)**2

        # Vision module
        if arch.lower() == 'vit_tiny':
            self.vision = vit_tiny(
                patch_size=patch_size,
                drop_path_rate=0.1,  # stochastic depth
            )
            dim_mlp = 192
        elif arch.lower() == 'vit_small':
            self.vision = vit_small(
                patch_size=patch_size,
                drop_path_rate=0.1,  # stochastic depth
            )
            dim_mlp = 384
        elif arch.lower() == 'vit_base':
            self.vision = vit_base(
                patch_size=patch_size,
                drop_path_rate=0.1,  # stochastic depth
            )
            dim_mlp = 768
        else:
            self.vision = torchvision.models.__dict__[arch](num_classes=dim, zero_init_residual=True)
            dim_mlp = self.vision.fc.weight.shape[1]
            self.vision.fc = nn.Identity()

        self.vision_fc = Head(dim_mlp, dim, use_bn=False, norm_last_layer=False)

        # Language module
        self.embedding = RelaxedEmbedding(vocab_size, hidden_size)
        self.position_encoding = nn.Parameter(torch.zeros(1, n_patches*max_len+1, hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.position_encoding, std=0.01)
        nn.init.normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dropout=attn_dropout, dim_feedforward=4*hidden_size)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # self.text_fc = nn.Linear(hidden_size, dim)
        self.text_fc = Head(hidden_size, dim, use_bn=False, norm_last_layer=False)

    def forward(self, m, x):
        B, P, L, V = m.shape
        m = m.reshape(B, P*L, V)

        # Embed the symbols
        m_emb = self.embedding(m)

        # Append cls embedding at the beginning of each message
        cls_tokens = self.cls_token.expand(B, -1, -1)
        m_emb = torch.cat((cls_tokens, m_emb), dim=1)

        # Add position encoding
        m_emb = m_emb + self.position_encoding

        # PyTorch transformer expects batch to be the second dimension
        m_emb = m_emb.transpose(0, 1)
        m_emb = self.encoder(m_emb)
        m_emb = m_emb.transpose(0, 1)[:, 0]  # Just use the embedding of cls_token
        m_emb = self.text_fc(m_emb)

        # Embed images
        x_emb = self.vision(x)
        x_emb = self.vision_fc(x_emb)
        return m_emb, x_emb


class PatchGame(nn.Module):
    """
    """
    def __init__(
        self, image_size=224, patch_size=32, 
        sender_arch='resnet', sender_hidden_size=768, sender_dropout=0.1,
        use_context=True, sender_norm='sort', 
        vocab_size=3, max_len=1, temperature=1.0, 
        trainable_temperature=False, hard=False,
        receiver_arch='resnet18', receiver_dim=128,
        receiver_hidden_size=768,
        num_heads=12, num_layers=12, topk=None
    ):
        super(PatchGame, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.sender_arch = sender_arch
        self.sender_hidden_size = sender_hidden_size
        self.sender_dropout = sender_dropout
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.temperature = temperature
        self.trainable_temperature = trainable_temperature
        self.hard = hard
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.sender = Speaker(
            patch_size=patch_size, image_size=image_size, 
            hidden_size=sender_hidden_size, dropout_rate=sender_dropout,
            arch=sender_arch, use_context=use_context, norm=sender_norm, topk=topk
        )
        self.gumbel = GumbelWrapper(
            vocab_size=vocab_size, max_len=max_len, hidden_size=sender_hidden_size,
            temperature=temperature, trainable_temperature=trainable_temperature,
            straight_through=hard
        )
        self.receiver = Listener(
            arch=receiver_arch, dim=receiver_dim, 
            patch_size=patch_size, image_size=image_size, 
            vocab_size=vocab_size, max_len=max_len,
            hidden_size=receiver_hidden_size,
            num_heads=num_heads, num_layers=num_layers,
        )

    def forward(self, im_sender, im_receiver, multigpu_loss=True, temperature=None):
        symbol_embed, selected = self.sender(im_sender)
        m = self.gumbel(symbol_embed, temperature)
        m_selected = m * selected.unsqueeze(-1).unsqueeze(-1)
        m_embed, im_embed = self.receiver(m_selected, im_receiver)

        im_embed = im_embed / im_embed.norm(dim=-1, keepdim=True)
        m_embed = m_embed / m_embed.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        bs = im_receiver.size(0)
        gpu_idx = torch.distributed.get_rank()

        if multigpu_loss:
            # image loss
            all_m_embed = concat_all_gather(m_embed)
            logits_per_image = logit_scale * im_embed @ all_m_embed.t()
            # image_loss = F.cross_entropy(logits_per_image, labels)

            # text loss
            all_im_embed = concat_all_gather(im_embed)
            logits_per_text = logit_scale * m_embed @ all_im_embed.t()
            # text_loss  = F.cross_entropy(logits_per_text, labels)

            logits = torch.cat([logits_per_image, logits_per_text], dim=0)
            labels = torch.arange(gpu_idx*bs, (gpu_idx+1)*bs).to(logits.device).repeat(2)
            return logits, labels

        else:
            logits_per_image = logit_scale * im_embed @ m_embed.t()
            logits_per_text = logit_scale * m_embed @ im_embed.t()

            labels = torch.arange(logits_per_image.size(0)).to(logits_per_image.device).repeat(2)

            return torch.cat([logits_per_image, logits_per_text], dim=0), labels


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def sample_gumbel(shape, eps=1e-20, device='cpu'):
    U = torch.rand(shape)
    U = U.to(device)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size(), device=logits.device)
    return nn.functional.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)

    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard


def straight_through_topk(x, k):
    """
    Assumes input size is batch_size x scores 
    """
    topk_hard = torch.zeros_like(x)
    topk = torch.topk(x, k, dim=1).indices
    batch_index = torch.arange(x.size(0)).repeat_interleave(k)
    topk_hard[batch_index, topk.reshape(-1)] = 1.
    return (topk_hard - x).detach() + x
