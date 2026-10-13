"""Optional browser QA; requires Playwright and its Chromium install."""
import json,itertools
from pathlib import Path
from playwright.sync_api import sync_playwright
root=Path(__file__).resolve().parents[1]
with sync_playwright() as p:
 b=p.chromium.launch(headless=True,args=['--no-sandbox']);page=b.new_page(viewport={'width':1280,'height':1000});errors=[];page.on('pageerror',lambda e:errors.append(str(e)));page.goto((root/'docs/index.html').as_uri());assert page.inner_text('#runs')=='18'
 n=0
 for env,alg,seed,task in itertools.product(['cube-single-play-v0','cube-double-play-v0'],['gcbc','gciql','hiql'],['0','1','2'],['1','2','3','4','5']):
  for key,val in [('env',env),('alg',alg),('seed',seed),('task',task)]:page.select_option('#'+key,val)
  assert 'No recorded' not in page.inner_text('#status');assert '250 episodes' in page.inner_text('#evidence');n+=1
 page.click('#success');assert page.inner_text('#status')=='Reached goal';page.click('#play');page.wait_for_timeout(1100);page.click('#play')
 page.locator('#scrub').evaluate('(e)=>{e.value=e.max;e.dispatchEvent(new Event("input"))}')
 with page.expect_download() as info:page.click('#download')
 download=info.value;payload=json.loads(Path(download.path()).read_text());assert payload['replay']['success'] is True
 page.mouse.move(300,650);page.mouse.down();page.mouse.move(420,650);page.mouse.up();page.click('#resetview')
 desktop_overflow=page.evaluate('document.documentElement.scrollWidth>innerWidth');page.set_viewport_size({'width':390,'height':844});mobile_overflow=page.evaluate('document.documentElement.scrollWidth>innerWidth');assert not desktop_overflow and not mobile_overflow;assert not errors
 report={'selector_combinations_checked':n,'playback_scrubbing_download_camera':'passed','success_replay':'passed','desktop_horizontal_overflow':desktop_overflow,'mobile_horizontal_overflow':mobile_overflow,'javascript_errors':errors}
 (root/'docs/browser_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report));b.close()
