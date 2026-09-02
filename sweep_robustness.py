"""ClinicDB robustness sweep for batch size, resolution, AMP, and seed."""
import argparse
import itertools
import subprocess


def main(args):
    for bs, sz, amp, seed in itertools.product(args.batch_sizes, args.img_sizes, args.amp, args.seeds):
        cmd = [
            "python", "-W", "ignore", "train_polyp.py",
            "--network", args.network,
            "--dataset_name", "ClinicDB",
            "--batchsize", str(bs),
            "--test_batchsize", str(args.test_batchsize),
            "--img_size", str(sz),
            "--epoch", str(args.epochs),
            "--num_runs", "1",
            "--seed", str(seed),
            "--amp", str(amp).lower(),
            "--augmentation", "false",
        ]
        print("\nRUN:", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print("FAILED:", e)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--network", default="MK_UNet")
    p.add_argument("--batch_sizes", nargs="+", type=int, default=[4,8,16])
    p.add_argument("--img_sizes", nargs="+", type=int, default=[256,352])
    p.add_argument("--amp", nargs="+", default=["false","true"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42,43,44])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--test_batchsize", type=int, default=8)
    main(p.parse_args())
