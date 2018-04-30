#!/usr/bin/env bash
python replicator.py prisoners --phase;
python replicator.py stag --phase;
python replicator.py matching --phase;
python replicator.py symmetric_random --phase;
python replicator.py symmetric_random --size=5;