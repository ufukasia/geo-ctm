"""
Fetch the upstream CTM repository into ./third_party/.

We deliberately do not vendor it (no LICENSE upstream; its bundled YOLOv5 is
GPL-3.0). This script clones it, then checks that the pieces the benchmark
needs are actually present.

    python tools/setup_upstream.py

The 60 cropped UFPR-ALPR test tracks are NOT in the git repository. The upstream
README links them from Google Drive; download the archive and unpack it as
third_party/Character-Time-series-Matching/test_tracks/. This script prints the
link and verifies the result.
"""
import argparse
import os
import subprocess
import sys

REPO = 'https://github.com/chequanghuy/Character-Time-series-Matching.git'
TRACKS_URL = ('https://drive.google.com/file/d/'
              '1U429KxS6SvoOMGoAGw0yQaDlLmBrLFgE/view?usp=sharing')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(ROOT, 'third_party', 'Character-Time-series-Matching')

REQUIRED_FILES = [
    ('process_plate.py', 'CTM association and voting (baseline B2/B3)'),
    ('Char_detection_yolo.py', 'character detector wrapper'),
    ('character_name.txt', 'class names'),
    ('exp/weights/best.pt', 'trained character detector weights'),
    ('yolov5/models/experimental.py', 'bundled YOLOv5'),
]


def clone():
    if os.path.isdir(os.path.join(DEST, '.git')):
        print(f'[skip] already cloned: {DEST}')
        return True
    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    print(f'[clone] {REPO}\n     -> {DEST}')
    try:
        subprocess.check_call(['git', 'clone', '--depth', '1', REPO, DEST])
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f'[fail] git clone failed: {e}')
        print('       Clone it manually to the path above and re-run this script.')
        return False


def verify():
    ok = True
    print('\nChecking required files:')
    for rel, what in REQUIRED_FILES:
        present = os.path.exists(os.path.join(DEST, rel))
        print(f'  [{"ok" if present else "--"}] {rel:<38} {what}')
        ok &= present

    tracks = os.path.join(DEST, 'test_tracks')
    n = len([d for d in os.listdir(tracks)
             if os.path.isdir(os.path.join(tracks, d))]) if os.path.isdir(tracks) else 0
    print(f'\n  [{"ok" if n else "--"}] test_tracks/{" " * 27}{n} tracks '
          f'(expected 60)')
    if n != 60:
        print('\n  The cropped test tracks are not in the git repo. Download them:')
        print(f'    {TRACKS_URL}')
        print(f'  and unpack so that {os.path.join(DEST, "test_tracks", "track0091")}'
              ' exists.')
        ok = False
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--verify-only', action='store_true')
    args = p.parse_args()

    if not args.verify_only and not clone():
        return 1

    if verify():
        print('\nReady. Next:\n  python tools/eval_ufpr.py')
        return 0

    print('\nSome pieces are missing — see above. The pure-fusion parts of the'
          '\npackage (geoctm.fusion / decode / degrade) work without them:'
          '\n  python -m pytest tests/')
    return 1


if __name__ == '__main__':
    sys.exit(main())
