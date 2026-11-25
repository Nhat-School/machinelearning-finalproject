// Real Chrome screenshots and end-to-end checks. Reuses installed Chrome, no browser download.
const fs=require('fs'),path=require('path');
const root=path.resolve(__dirname,'..');
const bundled=path.join(process.env.HOME,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||bundled);
(async()=>{const dir=path.join(root,'reports/screenshots');fs.mkdirSync(dir,{recursive:true});const browser=await chromium.launch({executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true,args:['--disable-background-networking'],downloadsPath:path.join(root,'.cache/downloads')});
try{const page=await browser.newPage({viewport:{width:1440,height:1100},deviceScaleFactor:1});await page.goto(process.env.PROJECT_APP_URL||'http://127.0.0.1:8765');await page.waitForFunction(()=>document.querySelector('#model').options.length>=8);await page.screenshot({path:path.join(dir,'01_home.png'),fullPage:true});
const checks=[];
for(const [example,file] of [['urgent','02_urgent.png'],['thanks','03_thanks.png']]){await page.locator(`[data-example="${example}"]`).click();const request=page.waitForResponse(r=>r.url().endsWith('/api/predict')&&r.request().method()==='POST');await page.locator('#predict').click();const result=await (await request).json();await page.waitForFunction(score=>document.querySelector('#result').dataset.score===String(score),result.items[0].score);checks.push({case:example,input:result.items[0].text,model:result.model_key,score:result.items[0].score,label:result.items[0].priority,threshold:result.items[0].threshold,api_matches_ui:true});await page.screenshot({path:path.join(dir,file),fullPage:true})}
const previousModel=await page.locator('#model').inputValue();
if(await page.locator('#model option[value="stanford__finetuned_minilm"]').count()){
await page.locator('#model').selectOption('stanford__finetuned_minilm');await page.locator('[data-example="urgent"]').click();
let response=page.waitForResponse(r=>r.url().endsWith('/api/predict')&&r.request().method()==='POST');await page.locator('#predict').click();let data=await (await response).json();await page.waitForFunction(score=>document.querySelector('#result').dataset.score===String(score),data.items[0].score);
checks.push({case:'finetuned_urgent',model:data.model_key,label:data.items[0].priority,score:data.items[0].score,threshold:data.items[0].threshold,api_matches_ui:true});await page.screenshot({path:path.join(dir,'07_finetuned.png'),fullPage:true});
await page.locator('#model').selectOption(previousModel);await page.locator('[data-example="thanks"]').click();response=page.waitForResponse(r=>r.url().endsWith('/api/predict')&&r.request().method()==='POST');await page.locator('#predict').click();data=await (await response).json();await page.waitForFunction(score=>document.querySelector('#result').dataset.score===String(score),data.items[0].score);
}
await page.locator('#message').fill('');await page.locator('#predict').click();await page.waitForFunction(()=>document.querySelector('#error').textContent.length>0);checks.push({case:'empty_input',rejected:true});
if(await page.locator('#model option[value="stanford__minilm_tfidf_blend"]').count()){
await page.locator('#model').selectOption('stanford__minilm_tfidf_blend');await page.locator('[data-example="urgent"]').click();
const response=page.waitForResponse(r=>r.url().endsWith('/api/predict')&&r.request().method()==='POST');await page.locator('#predict').click();const data=await (await response).json();await page.waitForFunction(score=>document.querySelector('#result').dataset.score===String(score),data.items[0].score);
checks.push({case:'hybrid_urgent',model:data.model_key,label:data.items[0].priority,score:data.items[0].score,threshold:data.items[0].threshold,api_matches_ui:true});await page.screenshot({path:path.join(dir,'08_hybrid.png'),fullPage:true});
await page.locator('#model').selectOption(previousModel);await page.locator('[data-example="thanks"]').click();
const restoredResponse=page.waitForResponse(r=>r.url().endsWith('/api/predict')&&r.request().method()==='POST');await page.locator('#predict').click();const restored=await (await restoredResponse).json();await page.waitForFunction(score=>document.querySelector('#result').dataset.score===String(score),restored.items[0].score);
}
if(await page.locator('#model option[value="joint__reliable_deberta"]').count()){
await page.locator('#model').selectOption('joint__reliable_deberta');await page.locator('[data-example="urgent"]').click();
let response=page.waitForResponse(r=>r.url().endsWith('/api/predict')&&r.request().method()==='POST');await page.locator('#predict').click();let data=await (await response).json();await page.waitForFunction(score=>document.querySelector('#result').dataset.score===String(score),data.items[0].score);
checks.push({case:'deberta_new_post',model:data.model_key,action:data.items[0].action,score:data.items[0].score,api_matches_ui:true});await page.screenshot({path:path.join(dir,'09_deberta.png'),fullPage:true});
const audit=JSON.parse(fs.readFileSync(path.join(root,'reports/reliability/deberta_real_only/browser_review_input.json'),'utf8'));
await page.locator('#message').fill(audit.text);response=page.waitForResponse(r=>r.url().endsWith('/api/predict')&&r.request().method()==='POST');await page.locator('#predict').click();data=await (await response).json();
await page.waitForFunction(score=>document.querySelector('#result').dataset.score===String(score),data.items[0].score);
if(data.items[0].action!=='REVIEW'||await page.locator('#result').getAttribute('data-action')!=='REVIEW')throw Error('Review action was not displayed');
checks.push({case:'deberta_review',input:audit.text,model:data.model_key,action:data.items[0].action,score:data.items[0].score,api_matches_ui:true});await page.screenshot({path:path.join(dir,'10_deberta_review.png'),fullPage:true});
await page.locator('[data-tab="queue"]').click();await page.locator('#demo-queue').click();await page.waitForFunction(()=>document.querySelector('#queue-list').dataset.model===document.querySelector('#model').value&&!document.querySelector('#queue-status').textContent.includes('Đang'));await page.waitForSelector('.queue-row:nth-child(8)');
const downloadPromise=page.waitForEvent('download');await page.locator('#export').click();const download=await downloadPromise;const reviewExport=path.join(dir,'exported_deberta_queue.csv');await download.saveAs(reviewExport);
if(!fs.readFileSync(reviewExport,'utf8').split('\n')[0].includes('action'))throw Error('Review action missing from exported CSV');checks.push({case:'deberta_csv_has_action',passed:true});
await page.locator('[data-tab="classify"]').click();await page.locator('#model').selectOption(previousModel);await page.locator('[data-example="thanks"]').click();response=page.waitForResponse(r=>r.url().endsWith('/api/predict')&&r.request().method()==='POST');await page.locator('#predict').click();data=await (await response).json();await page.waitForFunction(score=>document.querySelector('#result').dataset.score===String(score),data.items[0].score);
}
for(const key of ['joint__minilm_real_control','joint__minilm_custom_trial']){
if(!await page.locator(`#model option[value="${key}"]`).count())continue;
await page.locator('#model').selectOption(key);
for(const [state,input] of [['blocked','I cannot submit my assignment and the deadline is tonight. Please help!'],['resolved','The submit button works now and my assignment has a receipt. No further help is needed.'],['still_blocked','Update: I tried a different browser, but the assignment still will not upload. I still need help.']]){
await page.locator('#message').fill(input);const pending=page.waitForResponse(r=>r.url().endsWith('/api/predict')&&r.request().method()==='POST');await page.locator('#predict').click();const result=await (await pending).json();
await page.waitForFunction(data=>document.querySelector('#result').dataset.model===data.model_key&&document.querySelector('#result').dataset.score===String(data.items[0].score),result);
const item=result.items[0];if(await page.locator('#result').getAttribute('data-label')!==String(item.priority))throw Error('Final MiniLM label differs from API');
checks.push({case:'final_minilm_'+state,input,model:key,score:item.score,label:item.priority,action:item.action,api_matches_ui:true});
if(state==='resolved')await page.locator('.workspace').screenshot({path:path.join(dir,key.endsWith('custom_trial')?'11_final_minilm.png':'12_final_minilm_control.png')});
}}
await page.locator('#model').selectOption(previousModel);
await page.locator('[data-tab="queue"]').click();await page.locator('#demo-queue').click();await page.waitForFunction(()=>document.querySelector('#queue-list').dataset.model===document.querySelector('#model').value&&!document.querySelector('#queue-status').textContent.includes('Đang'));await page.waitForSelector('.queue-row:nth-child(8)');await page.screenshot({path:path.join(dir,'04_queue.png'),fullPage:true});const values=await page.locator('.queue-score').allTextContents();if(values.some((x,i)=>i&&Number(x)>Number(values[i-1])))throw Error('Queue not sorted');checks.push({case:'batch_8_sorted',passed:true});
const downloadPromise=page.waitForEvent('download');await page.locator('#export').click();const download=await downloadPromise;await download.saveAs(path.join(dir,'exported_queue.csv'));checks.push({case:'csv_export',passed:true});
await page.locator('[data-tab="metrics"]').click();await page.screenshot({path:path.join(dir,'05_metrics.png'),fullPage:true});
await page.setViewportSize({width:390,height:844});await page.locator('[data-tab="classify"]').click();await page.screenshot({path:path.join(dir,'06_mobile.png'),fullPage:true});
const overflow=await page.evaluate(()=>document.documentElement.scrollWidth>window.innerWidth);if(overflow)throw Error('Mobile horizontal overflow');checks.push({case:'mobile_390',no_horizontal_overflow:true});
fs.writeFileSync(path.join(dir,'browser_checks.json'),JSON.stringify(checks,null,2)+'\n');console.log('PASS: app prediction, validation, queue sorting, export, metrics and mobile; screenshots saved.');
}finally{await browser.close()}})().catch(e=>{console.error(e);process.exitCode=1});
