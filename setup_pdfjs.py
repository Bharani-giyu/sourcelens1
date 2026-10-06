from __future__ import annotations

import io
import shutil
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / 'static' / 'pdfjs'

VERSION = '5.4.624'
URL = (
    f'https://github.com/mozilla/pdf.js/releases/download/'
    f'v{VERSION}/pdfjs-{VERSION}-dist.zip'
)


def main() -> None:
    marker = TARGET / '.version'

    if (
        marker.exists()
        and marker.read_text().strip() == VERSION
        and (TARGET / 'web' / 'viewer.html').exists()
    ):
        print(f'PDF.js {VERSION} is already installed.')
        return

    print(f'Downloading PDF.js {VERSION}...')
    request = urllib.request.Request(
        URL,
        headers={'User-Agent': 'SourceLens-Setup'},
    )

    with urllib.request.urlopen(request, timeout=120) as response:
        archive_bytes = response.read()

    TARGET.mkdir(parents=True, exist_ok=True)
    root = TARGET.resolve()

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        # Validate archive paths before extracting.
        for member in archive.infolist():
            destination = (root / member.filename).resolve()
            if not destination.is_relative_to(root):
                raise RuntimeError('Unsafe path in PDF.js archive.')

        archive.extractall(root)

    if not (TARGET / 'web' / 'viewer.html').exists():
        raise RuntimeError('PDF.js distribution did not contain viewer.html.')

    # Remove the sample PDF; retain distribution license files.
    sample = TARGET / 'web' / 'compressed.tracemonkey-pldi-09.pdf'
    sample.unlink(missing_ok=True)

    marker.write_text(VERSION)
    print(f'Installed PDF.js to {TARGET}')


if __name__ == '__main__':
    main()