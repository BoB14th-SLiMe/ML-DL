#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from typing import Any

import numpy as np

from train_common import run_training_common


def build_bayesian_model(T: int, D: int, hidden_dim: int, latent_dim: int, bidirectional: bool, n_train: int):
    import tensorflow_probability as tfp
    import tf_keras as keras

    tfd = tfp.distributions
    kl_scale = float(max(int(n_train), 1))

    def kld(q, p, _):
        return tfd.kl_divergence(q, p) / kl_scale

    layers = keras.layers
    models = keras.models

    x_in = layers.Input(shape=(T, D), name="encoder_input")

    if bidirectional:
        h = layers.Bidirectional(layers.LSTM(hidden_dim, return_sequences=False), name="enc_bi")(x_in)
    else:
        h = layers.LSTM(hidden_dim, return_sequences=False, name="enc")(x_in)

    z = tfp.layers.DenseFlipout(latent_dim, kernel_divergence_fn=kld, name="z_flipout")(h)

    rep = layers.RepeatVector(T, name="repeat")(z)
    dec = layers.LSTM(hidden_dim, return_sequences=True, name="dec")(rep)

    out = tfp.layers.DenseFlipout(D, kernel_divergence_fn=kld, name="out_flipout")
    y_out = layers.TimeDistributed(out, name="out_td")(dec)

    return models.Model(inputs=x_in, outputs=y_out, name="bayesian_lstm_autoencoder")


def _as_tensor_output(y: Any):
    if isinstance(y, (tuple, list)):
        return y[0]
    return y


def make_predict_bayesian(mc_samples: int):
    mc_samples = max(int(mc_samples), 1)

    def predict_fn(model, X: np.ndarray, batch_size: int) -> np.ndarray:
        n = int(X.shape[0])
        preds_sum = None

        for _ in range(mc_samples):
            outs = []
            for i in range(0, n, batch_size):
                xb = X[i : i + batch_size]
                yb = model(xb, training=True)
                yb = _as_tensor_output(yb)
                outs.append(yb.numpy())
            pred = np.concatenate(outs, axis=0)
            preds_sum = pred if preds_sum is None else (preds_sum + pred)

        return preds_sum / float(mc_samples)

    return predict_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input_jsonl", required=True)
    ap.add_argument("-o", "--output_dir", required=True)

    ap.add_argument("--window_size", type=int, default=None)
    ap.add_argument("--pad_value", type=float, default=-1.0)

    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--latent_dim", type=int, default=64)
    ap.add_argument("--bidirectional", action="store_true")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--feature-policy-file", type=str, default=None)
    ap.add_argument("--mc_samples", type=int, default=20)

    args = ap.parse_args()

    run_training_common(
        input_jsonl=args.input_jsonl,
        output_dir=args.output_dir,
        window_size=args.window_size,
        pad_value=args.pad_value,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        bidirectional=args.bidirectional,
        lr=args.lr,
        val_ratio=args.val_ratio,
        seed=args.seed,
        feature_policy_file=args.feature_policy_file,
        model_type="bayesian",
        build_model_fn=build_bayesian_model,
        predict_fn=make_predict_bayesian(args.mc_samples),
        extra_config={"mc_samples": int(args.mc_samples)},
    )


if __name__ == "__main__":
    main()
