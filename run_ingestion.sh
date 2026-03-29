#!/bin/bash
# monday 12:00 am
cd /opt/hayai
venv/bin/python hayai.py -p medium_tech_usa -i
venv/bin/python hayai.py -p eu -i
venv/bin/python hayai.py -p asia -i

venv/bin/python hayai.py -p medium_tech_usa -s
venv/bin/python hayai.py -p eu -s
venv/bin/python hayai.py -p asia -s
