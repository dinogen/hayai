#!/bin/bash
# every day 6:00 am
cd /opt/hayai
venv/bin/python hayai.py -p medium_tech_usa -r
venv/bin/python hayai.py -p eu -r
venv/bin/python hayai.py -p asia -r
