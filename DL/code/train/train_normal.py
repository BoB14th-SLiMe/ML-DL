#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np

from train import run_training_common


def build_normal_model(T: int, D: int, hidden_dim: int, latent_dim: int, bidirectional: bool, _n_train: int):
    import tensorflow as tf
    from tensorflow.keras import layers, models

    x_in = layers.Input(shape=(T, D), name="encoder_input")

    if bidirectional:
        enc = layers.Bidirectional(layers.LSTM(hidden_dim, return_sequences=False), name="enc_bi")(x_in)
    else:
        enc = layers.LSTM(hidden_dim, return_sequences=False, name="enc")(x_in)

    z = layers.Dense(latent_dim, name="latent")(enc)

    def repeat_latent(x):
        x = tf.expand_dims(x, axis=1)
        return tf.tile(x, [1, T, 1])

    rep = layers.Lambda(repeat_latent, name="repeat")(z)
    dec = layers.LSTM(hidden_dim, return_sequences=True, name="dec")(rep)
    y_out = layers.TimeDistributed(layers.Dense(D), name="out")(dec)

    return models.Model(inputs=x_in, outputs=y_out, name="lstm_autoencoder")


def predict_normal(model, X: np.ndarray, batch_size: int) -> np.ndarray:
    return model.predict(X, batch_size=batch_size, verbose=1)


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
        model_type="normal",
        build_model_fn=build_normal_model,
        predict_fn=predict_normal,
        extra_config={"mc_samples": 0},
    )


if __name__ == "__main__":
    main()
