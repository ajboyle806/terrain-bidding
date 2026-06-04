"""Entry point: python -m terrain_bidding.grandtour [n_missions]"""
import sys
from terrain_bidding.grandtour import process_missions

n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
process_missions(n_missions=n)
