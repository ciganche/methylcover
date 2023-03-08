#!/usr/bin/env python3

import importlib
import sys

VERSION = "0.1.0"
FUNCTIONALITIES = ["prepare_bam", "read_level_deconv", "beta_deconv"]

def print_global_help(funs):
    printout = "\nUsage: methylcover <functionality> [<arguments>]"
    printout += "\nRun methylcover <functionality> -h for more information"
    printout += "\n\nFunctionalities:\n"
    print(printout)
    print(*funs, sep='\n')
    print("\n")

def print_version(ver):
    print("Methylcover version: " + ver)


if __name__ == "__main__":
    if len(sys.argv) < 2 or (sys.argv[1] in ("-h", "--help")):
        print_global_help(FUNCTIONALITIES)
        exit(0)
    elif sys.argv[1] in ("-v", "--version"):
        print_version(VERSION)
        exit(0)
    elif sys.argv[1] not in FUNCTIONALITIES:
        print("Unknown command.")
        print_global_help(FUNCTIONALITIES)
        exit(1)

    module = sys.argv[1]
    importlib.import_module(module).main(sys.argv[2:])



