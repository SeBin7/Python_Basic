import argparse
from .attention_pattern.attention import identity_attention, uniform_attention, band_attention
from .attention_pattern.visualize import show_attention

def main():
    parser = argparse.ArgumentParser(description="Generate simple attention patterns")
    parser.add_argument("--pattern", choices=["identity", "uniform", "band"], required=True)
    parser.add_argument("--n", type=int, default=4, help="Sequence length")
    parser.add_argument("--k", type=int, default=1, help="Band width (for band pattern)")
    parser.add_argument("--save", default="", help="Optional path to save the heatmap image")
    parser.add_argument("--show", action="store_true", help="Force GUI show (not for WSL2)")
    args = parser.parse_args()

    # --- generate matrix depending on pattern ---
    if args.pattern == "identity":
        mat = identity_attention(args.n)
    elif args.pattern == "uniform":
        mat = uniform_attention(args.n)
    elif args.pattern == "band":
        mat = band_attention(args.n, args.k)
    else:
        raise ValueError(f"Unknown pattern: {args.pattern}")

    # --- visualize / save ---
    show_attention(mat, title=args.pattern, save=args.save, show=args.show)


if __name__ == "__main__":
    main()