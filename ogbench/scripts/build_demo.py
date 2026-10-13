from pathlib import Path
root=Path(__file__).resolve().parents[1];p=root/'docs';(p/'index.html').write_text((p/'template.html').read_text().replace('__DATA__',(p/'results.json').read_text()))
