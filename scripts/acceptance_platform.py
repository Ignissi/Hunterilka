"""Проверка поставки на чистом временном компьютере GitHub Actions."""
from datetime import datetime, timedelta
from hashlib import file_digest
import json
import os
from pathlib import Path
import platform
import re
import shutil
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from urllib.request import urlopen
from zipfile import ZipFile

import psutil
from playwright.sync_api import sync_playwright

from src import db
from src.contracts import feature_contract
from src.incoming import consume_batch
from src.models import load_models
from src.service import Service
from src.storage import ROOT, load_config


URL = 'http://127.0.0.1:8527'
RELEASE_SHA = '95d51b5f764c0320ec592dad81cf0c93899f678fd25ca84f379ea348c4efa4f7'


def prepare():
    package = Path('release-test.zip')
    with package.open('rb') as stream:
        assert file_digest(stream, 'sha256').hexdigest() == RELEASE_SHA
    dump = Path(os.environ['RUNNER_TEMP']) / 'release-example.dump'
    with ZipFile(package) as archive:
        for name in archive.namelist():
            relative = name.removeprefix('hunterilka-v2-ready/')
            if not relative.startswith('artifacts/') or relative.endswith(('/', '.gitkeep')):
                continue
            target = ROOT / 'artifacts' / Path(relative).name
            with archive.open(name) as src, target.open('wb') as dst:
                shutil.copyfileobj(src, dst)
        with archive.open('hunterilka-v2-ready/_local/transfer/hunterilka_v2.dump') as src, dump.open('wb') as dst:
            shutil.copyfileobj(src, dst)
    package.unlink()
    for name in ('hunter_test', 'hunter_bench', 'hunter_demo'):
        subprocess.run(['createdb', '-h', '127.0.0.1', '-U', 'postgres', name], check=True)
    subprocess.run(['pg_restore', '--exit-on-error', '--no-owner', '--no-privileges', '--dbname', os.environ['HUNTERILKA_DB'], str(dump)], check=True)
    dump.unlink()
    with db.connect() as connection:
        db.init_schema(connection)
        assert db.count_transactions(connection) == 4754476


def process_minutes(result):
    config = load_config()
    models = load_models(config=config)
    start = datetime(2026, 10, 21, 12, 7)
    example = json.loads((ROOT / 'examples/incoming/20261021T120700.json').read_text(encoding='utf-8'))['events'][0]
    result['model_id'] = models['meta']['model_id']
    with db.connect(os.environ['HUNTERILKA_BENCH_DB']) as connection, TemporaryDirectory() as folder:
        db.init_schema(connection)
        assert db.stream_state(connection) is None
        db.ensure_stream(connection, 'platform:1000', result['model_id'], feature_contract(config), start.isoformat(), kind='inbox')
        service = Service(config, models, connection)
        service.restore()
        for i in range(5):
            minute = start + timedelta(minutes=i)
            rows = []
            for j in range(1000):
                group, position = divmod(j, 8)
                amount = f'{30 + 13*j}.00'
                rows.append(dict(example, transaction_id=i*1000+j+1, timestamp=minute.isoformat(), source_bank='01', target_bank='01', source_account=f'G{group:03}-{position}', target_account=f'G{group:03}-{(position+1)%8}', amount_paid=amount, amount_received=amount))
            pending = Path(folder) / 'minute.tmp'
            pending.write_text(json.dumps({'events':rows}), encoding='utf-8')
            path = Path(folder) / (minute.strftime('%Y%m%dT%H%M%S') + '.json')
            tick = time.perf_counter()
            pending.replace(path)
            consume_batch(path, connection, service)
            elapsed = time.perf_counter() - tick
            assert (path.parent / 'processed' / path.name).is_file()
            assert db.count_transactions(connection) == (i+1)*1000
            assert db.stream_state(connection)['last_time'] == minute.isoformat()
            result['minutes_s'].append(round(elapsed, 4))


def open_cards(result):
    with Path('platform-streamlit.log').open('w', encoding='utf-8') as log:
        server = subprocess.Popen([sys.executable, '-m', 'streamlit', 'run', 'app.py', '--server.port=8527', '--server.address=127.0.0.1', '--server.headless=true'], stdout=log, stderr=subprocess.STDOUT)
        try:
            for _ in range(100):
                if server.poll() is not None:
                    raise RuntimeError('Streamlit exited during startup')
                try:
                    with urlopen(URL + '/_stcore/health', timeout=1) as response:
                        if response.status == 200:
                            break
                except OSError:
                    time.sleep(0.3)
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch()
                page = browser.new_page(viewport={'width':1420, 'height':1100})
                page.goto(URL, wait_until='domcontentloaded')
                caption = page.get_by_text(re.compile(r'Обнаружено .* · случай .* · версия ')).first
                caption.wait_for(timeout=90000)
                frame = page.locator('iframe').first
                table = page.get_by_role('tabpanel', name='Операции', exact=True).locator('[data-testid="stDataFrame"]').first
                table.wait_for()
                for _ in range(5):
                    old_caption = caption.inner_text()
                    previous = frame.get_attribute('srcdoc')
                    tick = time.perf_counter()
                    page.get_by_role('button', name=re.compile('Далее')).click()
                    page.wait_for_function("old=>Array.from(document.querySelectorAll('[data-testid=stCaptionContainer]')).some(e=>e.innerText.includes(' · случай ') && e.innerText.trim()!==old)", arg=old_caption)
                    page.wait_for_function("prev=>Array.from(document.querySelectorAll('iframe')).some(e=>e.getBoundingClientRect().width>0 && e.srcdoc && e.srcdoc!==prev && e.contentDocument?.querySelector('canvas'))", arg=previous)
                    table.wait_for(state='visible')
                    result['cards_s'].append(round(time.perf_counter()-tick, 4))
                    assert caption.inner_text() != old_caption
                page.get_by_role('tab', name='Рассмотрение и экспорт', exact=True).click()
                page.get_by_role('button', name='Сохранить решение').wait_for()
                result['review_form'] = True
                browser.close()
        finally:
            server.terminate()
            server.wait(timeout=30)


def run():
    result = dict(os=platform.platform(), runner_image=os.getenv('ImageOS'), machine=platform.machine(), native_arch=os.getenv('PROCESSOR_ARCHITEW6432', platform.machine()), python_bits=struct.calcsize('P')*8, physical_cores=psutil.cpu_count(logical=False), logical_cpus=psutil.cpu_count(), memory_bytes=psutil.virtual_memory().total, free_disk_bytes=shutil.disk_usage(ROOT).free, minutes_s=[], cards_s=[])
    try:
        process_minutes(result)
        open_cards(result)
        assert max(result['minutes_s']) <= 60 and max(result['cards_s']) <= 5
    finally:
        report = json.dumps(result, ensure_ascii=False, indent=2)
        Path('platform-result.json').write_text(report, encoding='utf-8')
        print('PLATFORM_RESULT ' + json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['prepare']:
        prepare()
    elif sys.argv[1:] == ['run']:
        run()
    else:
        raise SystemExit('Use prepare or run')
