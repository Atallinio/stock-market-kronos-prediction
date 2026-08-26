"""Flax NNX port of the Kronos financial-market foundation model.

Contains the binary-spherical-quantized tokenizer (KronosTokenizer) and the
autoregressive predictor (Kronos), plus the pretrained configs for the
base / small / mini variants. Extracted from kronos_rag.ipynb.

Weights are not stored in this repo: use transfer.load_models() to load the
official PyTorch checkpoints from HuggingFace and transfer them into these
NNX modules.
"""

import jax
import jax.numpy as jnp
import optax
from einshape import jax_einshape as einshape
from flax import nnx


import jax
import jax.numpy as jnp
import optax
from einshape import jax_einshape as einshape
from flax import nnx


class BinarySphericalQuantizer(nnx.Module):
    def __init__(self, embed_dim, beta, gamma0, gamma, zeta, group_size, inv_temperature=1.0, l2_norm=True, rngs=nnx.Rngs(0)):
        self.embed_dim = embed_dim
        self.l2_norm = l2_norm
        self.beta = beta
        self.gamma0 = gamma0
        self.gamma = gamma
        self.zeta = zeta
        self.group_size = group_size
        self.inv_temperature = inv_temperature

        assert embed_dim % group_size == 0, f"embed_dim ({embed_dim}) must be divisible by group_size ({group_size})"
        self.num_groups = embed_dim // group_size

        # Precompute bit conversion bases
        self.basis = nnx.Variable(2 ** jnp.arange(embed_dim - 1, -1, -1, dtype=jnp.int32))
        self.group_basis = nnx.Variable(2 ** jnp.arange(group_size - 1, -1, -1, dtype=jnp.int32))

        # Precompute group codebook table of size (2^group_size, group_size) with values in {-1.0, +1.0}
        group_codes = jnp.arange(2 ** group_size)
        self.group_codebook = nnx.Variable((group_codes[:, None] // self.group_basis) % 2 * 2.0 - 1.0)

    def quantize(self, z):
        zhat = jnp.where(z > 0, 1.0, -1.0)
        return z + jax.lax.stop_gradient(zhat - z)

    def codes_to_indexes(self, zhat):
        zb = ((zhat > 0).astype(jnp.int32))
        return jnp.sum(zb * self.basis, axis=-1)

    def codes_to_group_indexes(self, zhat):
        zhat_groups = zhat.reshape(*zhat.shape[:-1], self.num_groups, self.group_size)
        zb_groups = ((zhat_groups > 0).astype(jnp.int32))
        return jnp.sum(zb_groups * self.group_basis, axis=-1)

    def soft_entropy_loss(self, z):
        scale = 1.0 / jnp.sqrt(self.embed_dim) if self.l2_norm else 1.0
        group_cb_norm = self.group_codebook * scale
        divided_z = z.reshape(*z.shape[:-1], self.num_groups, self.group_size)

        # Distance and softmax probabilities over group codebook entries
        distance = -2.0 * jnp.einsum('...gc,dc->...gd', divided_z, group_cb_norm)
        prob = jax.nn.softmax(-distance * self.inv_temperature, axis=-1)

        # Analytical per-sample entropy
        p = jax.nn.sigmoid(-4.0 * z * scale * self.inv_temperature)
        per_sample_entropy = -jnp.sum(p * jnp.log(p + 1e-8) + (1.0 - p) * jnp.log(1.0 - p + 1e-8), axis=-1).mean()

        # Average probability across batch and time dimensions
        reduce_axes = tuple(range(prob.ndim - 2))
        avg_prob = jnp.mean(prob, axis=reduce_axes)  # Shape: (num_groups, 2^group_size)

        # Codebook entropy H
        codebook_entropy = -jnp.sum(avg_prob * jnp.log(avg_prob + 1e-8), axis=-1).sum()

        return per_sample_entropy, codebook_entropy, avg_prob

    def __call__(self, z, collect_metrics=True):
        zq = self.quantize(z)
        q_scale = 1.0 / jnp.sqrt(self.embed_dim) if self.l2_norm else 1.0
        zq = zq * q_scale

        if not collect_metrics:
            return zq, jnp.array(0.0), {}

        commit_loss = self.beta * jnp.mean(jnp.sum((jax.lax.stop_gradient(zq) - z) ** 2, axis=-1))

        per_sample_entropy, cb_entropy, avg_prob = self.soft_entropy_loss(z)
        entropy_penalty = self.gamma0 * per_sample_entropy - self.gamma * cb_entropy

        total_bsq_loss = commit_loss + self.zeta * entropy_penalty / self.inv_temperature

        zq_detached = jax.lax.stop_gradient(zq)
        indices = self.codes_to_indexes(zq_detached)
        group_indices = self.codes_to_group_indexes(zq_detached)

        metrics = {
            "H": cb_entropy,
            "per_sample_entropy": per_sample_entropy,
            "commit_loss": commit_loss,
            "indices": indices,
            "group_indices": group_indices,
            "avg_prob": avg_prob
        }

        return zq, total_bsq_loss, metrics


class BSQuantizer(nnx.Module):
    def __init__(self, s1_bits, s2_bits, beta, gamma0, gamma, zeta, group_size, rngs=nnx.Rngs(0)):
        self.s1_bits = s1_bits
        self.s2_bits = s2_bits
        self.codebook_dim = s1_bits + s2_bits
        self.bsq = BinarySphericalQuantizer(self.codebook_dim, beta, gamma0, gamma, zeta, group_size, rngs=rngs)

    def bits_to_indices(self, bits):
        set_bits = (bits >= 0).astype(jnp.int32)
        powers = 2 ** jnp.arange(bits.shape[-1], dtype=jnp.int32)
        return jnp.einsum('...d,d->...', set_bits, powers)

    def __call__(self, z, half=False, collect_metrics=True):
        norm = jnp.linalg.norm(z, axis=-1, keepdims=True)
        z_norm = z / jnp.maximum(norm, 1e-12)
        quantized, bsq_loss, metrics = self.bsq(z_norm, collect_metrics=collect_metrics)
        if half:
            q_pre = quantized[:, :, :self.s1_bits]
            q_post = quantized[:, :, self.s1_bits:]
            z_indices = [self.bits_to_indices(q_pre), self.bits_to_indices(q_post)]
        else:
            z_indices = self.bits_to_indices(quantized)
        return bsq_loss, quantized, z_indices, metrics


def make_linear(in_features, out_features, lora_rank=0, use_bias=True, rngs=nnx.Rngs(0)):
    if lora_rank > 0:
        return nnx.LoRALinear(in_features, out_features, lora_rank=lora_rank, use_bias=use_bias, rngs=rngs)
    else:
        return nnx.Linear(in_features, out_features, use_bias=use_bias, rngs=rngs)


class FeedForward(nnx.Module):
    def __init__(self, d_model, ff_dim, ffn_dropout_p=0.0, lora_rank=0, rngs=nnx.Rngs(0)):
        self.w1 = make_linear(d_model, ff_dim, lora_rank=lora_rank, use_bias=False, rngs=rngs)
        self.w3 = make_linear(d_model, ff_dim, lora_rank=lora_rank, use_bias=False, rngs=rngs)
        self.w2 = make_linear(ff_dim, d_model, lora_rank=lora_rank, use_bias=False, rngs=rngs)
        self.ffn_dropout = nnx.Dropout(ffn_dropout_p, rngs=rngs)

    def __call__(self, x):
        h = nnx.silu(self.w1(x)) * self.w3(x)
        return self.ffn_dropout(self.w2(h))


class RotaryPositionalEmbedding(nnx.Module):
    def __init__(self, dim):
        self.dim = dim
        self.inv_freq = nnx.Variable(1.0 / (10000.0 ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim)))

    def _rotate_half(self, x):
        x1, x2 = jnp.split(x, 2, axis=-1)
        return jnp.concatenate([-x2, x1], axis=-1)

    def _get_embed(self, seq_len):
        t = jnp.arange(seq_len, dtype=jnp.float32)
        freqs = jnp.einsum('i,j->ij', t, self.inv_freq)
        emb = jnp.concatenate([freqs, freqs], axis=-1)
        cos = einshape('sd->11sd', jnp.cos(emb))
        sin = einshape('sd->11sd', jnp.sin(emb))
        return cos, sin

    def __call__(self, q, k):
        q_len = q.shape[-2]
        k_len = k.shape[-2]

        q_cos, q_sin = self._get_embed(q_len)
        q_out = q * q_cos + self._rotate_half(q) * q_sin

        if q_len == k_len:
            k_out = k * q_cos + self._rotate_half(k) * q_sin
        else:
            k_cos, k_sin = self._get_embed(k_len)
            k_out = k * k_cos + self._rotate_half(k) * k_sin

        return q_out, k_out


class MultiHeadAttentionWithRoPE(nnx.Module):
    def __init__(self, d_model, n_heads, attn_dropout_p, resid_dropout_p, lora_rank, rngs=nnx.Rngs(0)):
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = make_linear(d_model, d_model, lora_rank=lora_rank, rngs=rngs)
        self.k_proj = make_linear(d_model, d_model, lora_rank=lora_rank, rngs=rngs)
        self.v_proj = make_linear(d_model, d_model, lora_rank=lora_rank, rngs=rngs)
        self.out_proj = make_linear(d_model, d_model, lora_rank=lora_rank, rngs=rngs)
        self.rotary = RotaryPositionalEmbedding(self.head_dim)
        self.attn_dropout = nnx.Dropout(attn_dropout_p, rngs=rngs)
        self.resid_dropout = nnx.Dropout(resid_dropout_p, rngs=rngs)

    def __call__(self, x, key_padding_mask=None):
        split_heads = lambda t: einshape('bs(hd)->bhsd', t, h=self.n_heads)
        q = split_heads(self.q_proj(x))
        k = split_heads(self.k_proj(x))
        v = split_heads(self.v_proj(x))

        q, k = self.rotary(q, k)
        scores = jnp.einsum('bhqd,bhkd->bhqk', q, k) / jnp.sqrt(self.head_dim)

        seq_len = x.shape[1]
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
        scores = jnp.where(einshape('qk->11qk', causal_mask), scores, -jnp.inf)

        if key_padding_mask is not None:
            scores = jnp.where(einshape('bk->b11k', key_padding_mask.astype(jnp.bool_)), scores, -jnp.inf)

        attn_weights = nnx.softmax(scores, axis=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = jnp.einsum('bhqk,bhkd->bhqd', attn_weights, v)
        attn_output = einshape('bhsd->bs(hd)', attn_output)
        return self.resid_dropout(self.out_proj(attn_output))


class MultiHeadCrossAttentionWithRoPE(nnx.Module):
    def __init__(self, d_model, n_heads, attn_dropout_p, resid_dropout_p, lora_rank, deterministic=False, rngs=nnx.Rngs(0)):
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.deterministic = deterministic
        self.q_proj = make_linear(d_model, d_model, lora_rank=lora_rank, rngs=rngs)
        self.k_proj = make_linear(d_model, d_model, lora_rank=lora_rank, rngs=rngs)
        self.v_proj = make_linear(d_model, d_model, lora_rank=lora_rank, rngs=rngs)
        self.out_proj = make_linear(d_model, d_model, lora_rank=lora_rank, rngs=rngs)
        self.rotary = RotaryPositionalEmbedding(self.head_dim)
        self.attn_dropout = nnx.Dropout(attn_dropout_p, rngs=rngs)
        self.resid_dropout = nnx.Dropout(resid_dropout_p, rngs=rngs)

    def __call__(self, query, key, value, key_padding_mask=None):
        q_len = query.shape[1]
        seq_len = key.shape[1]
        q = einshape('bq(hd)->bhqd', self.q_proj(query), h=self.n_heads)
        k = einshape('bk(hd)->bhkd', self.k_proj(key), h=self.n_heads)
        v = einshape('bk(hd)->bhkd', self.v_proj(value), h=self.n_heads)

        q, k = self.rotary(q, k)
        scores = jnp.einsum('bhqd,bhkd->bhqk', q, k) / jnp.sqrt(self.head_dim)
        
        if not self.deterministic:
            causal_mask = jnp.tril(jnp.ones((q_len, seq_len), dtype=jnp.bool_))
            scores = jnp.where(einshape('qk->11qk', causal_mask), scores, -jnp.inf)

        if key_padding_mask is not None:
            scores = jnp.where(einshape('bk->b11k', key_padding_mask), scores, -jnp.inf)

        attn_weights = nnx.softmax(scores, axis=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = jnp.einsum('bhqk,bhkd->bhqd', attn_weights, v)
        attn_output = einshape('bhqd->bq(hd)', attn_output)
        return self.resid_dropout(self.out_proj(attn_output))



class HierarchicalEmbedding(nnx.Module):
    def __init__(self, s1_bits, s2_bits, d_model, rngs=nnx.Rngs(0)):
        self.s1_bits = s1_bits
        self.s2_bits = s2_bits
        self.d_model = d_model

        self.emb_s1 = nnx.Embed(2 ** s1_bits, d_model, rngs=rngs)
        self.emb_s2 = nnx.Embed(2 ** s2_bits, d_model, rngs=rngs)
        self.fusion_proj = nnx.Linear(d_model * 2, d_model, rngs=rngs)


    def __call__(self, token_ids):
        s1_ids, s2_ids = token_ids
        s1_emb = self.emb_s1(s1_ids) * jnp.sqrt(self.d_model)
        s2_emb = self.emb_s2(s2_ids) * jnp.sqrt(self.d_model)
        return self.fusion_proj(jnp.concatenate([s1_emb, s2_emb], axis=-1))


class DependencyAwareLayer(nnx.Module):
    def __init__(self, d_model, n_heads=4, attn_dropout_p=0.0, resid_dropout_p=0.0, lora_rank=0, rngs=nnx.Rngs(0)):
        self.cross_attn = MultiHeadCrossAttentionWithRoPE(d_model, n_heads, attn_dropout_p, resid_dropout_p, lora_rank=lora_rank, rngs=rngs)
        self.norm = nnx.RMSNorm(d_model, epsilon=1e-5, rngs=rngs)

    def __call__(self, hidden_states, sibling_embed, key_padding_mask=None):
        attn_out = self.cross_attn(
            query=sibling_embed,
            key=hidden_states,
            value=hidden_states,
            key_padding_mask=key_padding_mask,
        )
        return self.norm(hidden_states + attn_out)


class TransformerBlock(nnx.Module):
    def __init__(self, d_model, n_heads, ff_dim, ffn_dropout_p, attn_dropout_p, resid_dropout_p, lora_rank, rngs=nnx.Rngs(0)):
        self.norm1 = nnx.RMSNorm(d_model, epsilon=1e-5, rngs=rngs)
        self.self_attn = MultiHeadAttentionWithRoPE(d_model, n_heads, attn_dropout_p, resid_dropout_p, lora_rank=lora_rank, rngs=rngs)
        self.norm2 = nnx.RMSNorm(d_model, epsilon=1e-5, rngs=rngs)
        self.ffn = FeedForward(d_model, ff_dim, ffn_dropout_p, lora_rank=lora_rank, rngs=rngs)

    def __call__(self, x, key_padding_mask=None):
        residual = x
        x_norm = self.norm1(x)
        attn_out = self.self_attn(x_norm, key_padding_mask=key_padding_mask)
        x = residual + attn_out

        residual = x
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        return residual + ffn_out


class DualHead(nnx.Module):
    def __init__(self, s1_bits, s2_bits, d_model, rngs=nnx.Rngs(0)):
        self.proj_s1 = nnx.Linear(d_model, 2 ** s1_bits, rngs=rngs)
        self.proj_s2 = nnx.Linear(d_model, 2 ** s2_bits, rngs=rngs)

    def __call__(self, x):
        return self.proj_s1(x)

    def cond_forward(self, x2):
        return self.proj_s2(x2)


class TemporalEmbedding(nnx.Module):
    def __init__(self, d_model, rngs=nnx.Rngs(0)):
        minute_size = 60
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        self.minute_embed = nnx.Embed(minute_size, d_model, rngs=rngs)
        self.hour_embed = nnx.Embed(hour_size, d_model, rngs=rngs)
        self.weekday_embed = nnx.Embed(weekday_size, d_model, rngs=rngs)
        self.day_embed = nnx.Embed(day_size, d_model, rngs=rngs)
        self.month_embed = nnx.Embed(month_size, d_model, rngs=rngs)

    def __call__(self, x):
        x = x.astype(jnp.int32)
        minute_x = self.minute_embed(x[:, :, 0])
        hour_x = self.hour_embed(x[:, :, 1])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 3])
        month_x = self.month_embed(x[:, :, 4])
        return hour_x + weekday_x + day_x + month_x + minute_x


class KronosTokenizer(nnx.Module):
    def __init__(self, config, rngs=nnx.Rngs(0)):
        self.d_in = config["d_in"]
        self.d_model = config["d_model"]
        self.n_heads = config["n_heads"]
        self.ff_dim = config["ff_dim"]
        self.enc_layers = config["n_enc_layers"]
        self.dec_layers = config["n_dec_layers"]
        self.ffn_dropout_p = config["ffn_dropout_p"]
        self.attn_dropout_p = config["attn_dropout_p"]
        self.resid_dropout_p = config["resid_dropout_p"]
        self.lora_rank = config["lora_rank"]
        self.s1_bits = config["s1_bits"]
        self.s2_bits = config["s2_bits"]
        self.beta = config["beta"]
        self.gamma0 = config["gamma0"]
        self.gamma = config["gamma"]
        self.zeta = config["zeta"]
        self.group_size = config["group_size"]

        self.codebook_dim = self.s1_bits + self.s2_bits

        self.embed = make_linear(self.d_in, self.d_model, lora_rank=self.lora_rank, rngs=rngs)
        self.head = make_linear(self.d_model, self.d_in, lora_rank=self.lora_rank, rngs=rngs)

        self.encoder = nnx.List([
            TransformerBlock(self.d_model, self.n_heads, self.ff_dim, self.ffn_dropout_p, self.attn_dropout_p, self.resid_dropout_p, self.lora_rank, rngs=rngs)
            for _ in range(self.enc_layers - 1)
        ])
        self.decoder = nnx.List([
            TransformerBlock(self.d_model, self.n_heads, self.ff_dim, self.ffn_dropout_p, self.attn_dropout_p, self.resid_dropout_p, self.lora_rank, rngs=rngs)
            for _ in range(self.dec_layers - 1)
        ])

        self.quant_embed = make_linear(self.d_model, self.codebook_dim, lora_rank=self.lora_rank, rngs=rngs)
        self.post_quant_embed_pre = make_linear(self.s1_bits, self.d_model, lora_rank=self.lora_rank, rngs=rngs)
        self.post_quant_embed = make_linear(self.codebook_dim, self.d_model, lora_rank=self.lora_rank, rngs=rngs)
        self.tokenizer = BSQuantizer(self.s1_bits, self.s2_bits, self.beta, self.gamma0, self.gamma, self.zeta, self.group_size, rngs=rngs)

    def indices_to_bits(self, x, half=False):
        if half:
            x1, x2 = x[0], x[1]
            half_dim = self.codebook_dim // 2
            mask = 2 ** jnp.arange(half_dim, dtype=jnp.int32)
            b1 = (einshape('bs->bs1', x1.astype(jnp.int32)) & mask) != 0
            b2 = (einshape('bs->bs1', x2.astype(jnp.int32)) & mask) != 0
            bits = jnp.concatenate([b1, b2], axis=-1)
        else:
            mask = 2 ** jnp.arange(self.codebook_dim, dtype=jnp.int32)
            bits = (einshape('bs->bs1', x.astype(jnp.int32)) & mask) != 0

        bits = bits.astype(jnp.float32) * 2 - 1 
        q_scale = 1.0 / (self.codebook_dim ** 0.5)
        return bits * q_scale

    def decode(self, x, half=True):
        quantized = self.indices_to_bits(x, half=half)
        z = self.post_quant_embed(quantized)
        for layer in self.decoder:
            z = layer(z)
        return self.head(z)

    def encode(self, x, half=True, collect_metrics=False):
        z = self.embed(x)
        for layer in self.encoder:
            z = layer(z)
        z = self.quant_embed(z)
        _, _, z_indices, _ = self.tokenizer(z, half=half, collect_metrics=collect_metrics)
        return z_indices[0], z_indices[1]
    
    def compute_loss(self, x, z_pre, z, bsq_loss):
        recon_loss_pre = jnp.mean((z_pre - x) ** 2)
        recon_loss_all = jnp.mean((z - x) ** 2)
        recon_loss = recon_loss_pre + recon_loss_all
        loss = 0.5 * (recon_loss + bsq_loss)
        return loss

    def __call__(self, x, collect_metrics=True):
        z = self.embed(x)
        for layer in self.encoder:
            z = layer(z)
        z = self.quant_embed(z)

        bsq_loss, quantized, z_indices, metrics = self.tokenizer(z, collect_metrics=collect_metrics)

        quantized_pre = quantized[:, :, :self.s1_bits]
        z_pre = self.post_quant_embed_pre(quantized_pre)
        z = self.post_quant_embed(quantized)

        for layer in self.decoder:
            z_pre = layer(z_pre)
        z_pre = self.head(z_pre)

        for layer in self.decoder:
            z = layer(z)
        z = self.head(z)
        
        loss = self.compute_loss(x, z_pre, z, bsq_loss)
        return loss, quantized, z_indices, metrics



class Kronos(nnx.Module):
    def __init__(self, config, rngs=nnx.Rngs(0)):
        self.s1_bits = config["s1_bits"]
        self.s2_bits = config["s2_bits"]
        self.n_layers = config["n_layers"]
        self.d_model = config["d_model"]
        self.n_heads = config["n_heads"]
        self.ff_dim = config["ff_dim"]
        self.lora_rank = config["lora_rank"]
        self.ffn_dropout_p = config["ffn_dropout_p"]
        self.attn_dropout_p = config["attn_dropout_p"]
        self.resid_dropout_p = config["resid_dropout_p"]
        self.token_dropout_p = config["token_dropout_p"]
        self.s1_vocab_size = 2 ** self.s1_bits
        self.rngs = rngs

        self.token_drop = nnx.Dropout(self.token_dropout_p, rngs=rngs)
        self.embedding = HierarchicalEmbedding(self.s1_bits, self.s2_bits, self.d_model, rngs=rngs)
        self.time_emb = TemporalEmbedding(self.d_model, rngs=rngs)
        self.transformer = nnx.List([
            TransformerBlock(self.d_model, self.n_heads, self.ff_dim, self.ffn_dropout_p, self.attn_dropout_p, self.resid_dropout_p, lora_rank=self.lora_rank, rngs=rngs)
            for _ in range(self.n_layers)
        ])
        self.norm = nnx.RMSNorm(self.d_model, epsilon=1e-5, rngs=rngs)
        self.dep_layer = DependencyAwareLayer(self.d_model, lora_rank=self.lora_rank, rngs=rngs)
        self.head = DualHead(self.s1_bits, self.s2_bits, self.d_model, rngs=rngs)

    def compute_loss(self, s1_logits, s2_logits, s1_targets, s2_targets, padding_mask=None):
        if padding_mask is not None:
            valid = (padding_mask == 0)
            ce_s1 = optax.softmax_cross_entropy_with_integer_labels(s1_logits[valid], s1_targets[valid]).mean()
            ce_s2 = optax.softmax_cross_entropy_with_integer_labels(s2_logits[valid], s2_targets[valid]).mean()
        else:
            ce_s1 = optax.softmax_cross_entropy_with_integer_labels(einshape('bsv->(bs)v', s1_logits), einshape('bs->(bs)', s1_targets)).mean()
            ce_s2 = optax.softmax_cross_entropy_with_integer_labels(einshape('bsv->(bs)v', s2_logits), einshape('bs->(bs)', s2_targets)).mean()
        return (ce_s1 + ce_s2) / 2.0, ce_s1, ce_s2

    def __call__(self, s1_ids, s2_ids, stamp=None, padding_mask=None, s1_targets=None, s2_targets=None):
        x = self.embedding([s1_ids, s2_ids])
        if stamp is not None:
            time_embedding = self.time_emb(stamp)
            x = x + time_embedding
        x = self.token_drop(x)

        for layer in self.transformer:
            x = layer(x, key_padding_mask=padding_mask)

        x = self.norm(x)

        s1_logits = self.head(x)
        sample_s1_ids = self.rngs.categorical(s1_logits, axis=-1)
        sibling_embed = self.embedding.emb_s1(sample_s1_ids)
        x2 = self.dep_layer(x, sibling_embed, key_padding_mask=padding_mask)
        s2_logits = self.head.cond_forward(x2)

        if s1_targets is not None and s2_targets is not None:
            loss, _, _ = self.compute_loss(s1_logits, s2_logits, s1_targets, s2_targets, padding_mask=padding_mask)
            return loss

        return s1_logits, s2_logits, x


TOKENIZER_BASE_CONFIG = {
    "s1_bits": 10,
    "s2_bits": 10,
    "d_in": 6,
    "d_model": 256,
    "ff_dim": 512,
    "n_dec_layers": 4,
    "n_enc_layers": 4,
    "n_heads": 4,
    "attn_dropout_p": 0.0,
    "ffn_dropout_p": 0.0,
    "resid_dropout_p": 0.0,
    "beta": 0.05,
    "zeta": 0.05,
    "gamma": 1.1,
    "gamma0": 1.0,
    "group_size": 4,
    "lora_rank": 0
}

KRONOS_BASE_CONFIG = {
    "s1_bits": 10,
    "s2_bits": 10,
    "d_model": 832,
    "ff_dim": 2048,
    "n_heads": 16,
    "n_layers": 12,
    "attn_dropout_p": 0.0,
    "ffn_dropout_p": 0.2,
    "resid_dropout_p": 0.2,
    "token_dropout_p": 0.0,
    "lora_rank": 0
}

KRONOS_SMALL_CONFIG = {
    "s1_bits": 10,
    "s2_bits": 10,
    "d_model": 512,
    "ff_dim": 1024,
    "n_heads": 8,
    "n_layers": 8,
    "attn_dropout_p": 0.1,
    "ffn_dropout_p": 0.25,
    "resid_dropout_p": 0.25,
    "token_dropout_p": 0.1,
    "lora_rank": 0
}

KRONOS_MINI_CONFIG = {
    "s1_bits": 10,
    "s2_bits": 10,
    "d_model": 256,
    "ff_dim": 512,
    "n_heads": 4,
    "n_layers": 4,
    "attn_dropout_p": 0.0,
    "ffn_dropout_p": 0.2,
    "resid_dropout_p": 0.2,
    "token_dropout_p": 0.0,
    "lora_rank": 0
}
