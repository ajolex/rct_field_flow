from power_analysis import main

if __name__ == "__main__":
    # Pass through CLI args to allow --data/--cluster-var/--id-var
    import sys
    main(sys.argv[1:])
