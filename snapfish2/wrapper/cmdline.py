import argparse



def main():
    parser = argparse.ArgumentParser(description="CLI for SnapFISH2.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    args = parser.parse_args()
    args.func(args)