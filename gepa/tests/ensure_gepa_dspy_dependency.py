import sys

try:
    import datasets
except ImportError:
    print("Pass: The `datasets` package was not installed, as expected.")
    sys.exit(0)
else:
    print("Fail: The `datasets` package was unexpectedly installed.")
    sys.exit(1)
