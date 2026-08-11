#!/usr/bin/env python3
import os
import argparse

from plotting import run_post_flight_analysis


def main():
    parser = argparse.ArgumentParser(description="Analyze and Plot Flight CSV")
    parser.add_argument("csv_path", type=str, help="Path to the flight data CSV file")
    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
        print(f"[!] Error: File not found at {args.csv_path}")
        return

    run_post_flight_analysis(args.csv_path)


if __name__ == "__main__":
    main()
