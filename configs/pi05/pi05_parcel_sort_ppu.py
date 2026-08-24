_base_ = ['./pi05_parcel_sort.py']

# Use the optimized scaled-dot-product attention path on PPU.
model = dict(
    attention_implementation='sdpa',
)

train_dataloader = dict(
    per_device_batch_size=20,
)

runner = dict(
    max_epochs=7,
    optimizer=dict(lr=3e-5),
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    max_keep_ckpts=5,
    save_epoch_interval=1,
    save_iter_interval=5000,
    # Keep the current project's checkpoint policy: safetensors is always
    # saved, while the legacy .pt checkpoint is opt-in from the launcher.
    save_pt_checkpoints=False,
)
