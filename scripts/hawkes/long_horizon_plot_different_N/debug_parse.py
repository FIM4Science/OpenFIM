from create_long_horizon_plots import HORIZON_FILES, parse_latex_table


def main():
    for N, path in sorted(HORIZON_FILES.items()):
        df = parse_latex_table(path)
        print("File N=%s: %s" % (N, path))
        if df.empty:
            print("  Parsed 0 rows")
            continue
        methods = sorted(df["Method"].unique())
        print("  Parsed rows: %s, methods (%s): %s" % (len(df), len(methods), methods))
        counts = df["Method"].value_counts().to_dict()
        print("  Counts per method: %s" % counts)
    print("Done")


if __name__ == "__main__":
    main()
