name: Diagnose RLP Raw Response

on:
  workflow_dispatch:

jobs:
  diagnose:
    runs-on: ubuntu-latest
    timeout-minutes: 3

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install requests
        run: python -m pip install --upgrade requests

      - name: Run RLP raw diagnostic
        run: python diagnose_rlp_raw.py
