"use strict";
const $ = id => document.getElementById(id);
const palette = ["#5ed5be", "#edc786", "#82b2f2", "#dc9ee0", "#f09a9d", "#95cb88", "#c7b1f8", "#76cadf", "#dda671", "#b9cd70", "#a7c3dc", "#a495e8", "#e0b2aa", "#69b8a5", "#b6bfe4", "#dfc9aa"];
const phaseLabels = {bootstrap:"初始化",gpu_setup:"GPU 环境准备",tuning:"CPU 并行校准",freezing:"冻结教师",mining:"难例挖掘",sampling:"采样状态",labeling:"标注收益",training:"训练候选",evaluating:"独立评估",auditing:"独立审计",promoting:"更新教师",paused:"已暂停",stopped:"已停止",error:"运行错误",complete:"已完成"};
const stageIndices = {bootstrap:0,gpu_setup:0,tuning:0,freezing:0,mining:1,sampling:1,labeling:1,training:2,evaluating:3,auditing:3,promoting:4};
let token = "", latest = {}, history = [], lastHistorySignature = "", connectionFailures = 0;
let tablePage = 0, tableRows = [], presetResultsVersion = null;
const finite = x => x !== null && x !== undefined && x !== "" && Number.isFinite(Number(x));
const num = (v, digits=3) => finite(v) ? Number(v).toFixed(digits) : "—";
const signed = (v, digits=3) => finite(v) ? (Number(v)>0?"+":"") + Number(v).toFixed(digits) : "—";
function duration(seconds) {
  if (!finite(seconds) || seconds < 0) return "—";
  seconds = Math.floor(seconds);
  const days = Math.floor(seconds/86400), hours = Math.floor(seconds/3600)%24;
  return (days ? days+"天 " : "") + [hours,Math.floor(seconds/60)%60,seconds%60].map(v=>String(v).padStart(2,"0")).join(":");
}
function text(id,value){$(id).textContent = value;}
function colorBySign(el,value){el.classList.toggle("positive",finite(value)&&value>0);el.classList.toggle("negative",finite(value)&&value<0);}
function numberOr(...values){return values.find(finite);}
function displayModel(value){return value===null||value===undefined ? "—" : String(value).replace(/^model_/,"M").replace(/^round_/,"R");}
function usesGpuGames(status){return status.phase==="sampling"?status.sampling_simulation_backend==="cuda_games":status.phase==="labeling"?status.simulation_backend_actual==="cuda_games":["evaluating","auditing"].includes(status.phase)&&status.evaluation_simulation_backend==="cuda_games";}
function notice(message, error=false){const e=$("notice");e.hidden=!message;e.textContent=message||"";e.classList.toggle("error",error);}
async function api(path, options){const r=await fetch(path,{cache:"no-store",...options});if(!r.ok){let message=`HTTP ${r.status}`;try{message=(await r.json()).error||message;}catch(_){}throw new Error(message);}return r.json();}

class LineChart {
  constructor(canvasId,tooltipId,emptyId){
    this.canvas=$(canvasId);this.tooltip=$(tooltipId);this.empty=$(emptyId);this.series=[];
    this.options={zero:true,xTitle:"轮次"};this.points=[];
    new ResizeObserver(()=>this.draw()).observe(this.canvas.parentElement);
    this.canvas.addEventListener("mousemove", e=>this.hover(e));
    this.canvas.addEventListener("mouseleave", ()=>this.tooltip.hidden=true);
  }
  update(series,options={}){this.series=series;this.options={zero:true,xTitle:"轮次",...options};this.draw();}
  draw(){
    const rect=this.canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1;
    const w=rect.width,h=rect.height;if(w<20||h<20)return;
    this.canvas.width=Math.round(w*dpr);this.canvas.height=Math.round(h*dpr);
    const ctx=this.canvas.getContext("2d");ctx.scale(dpr,dpr);ctx.clearRect(0,0,w,h);
    const all=this.series.flatMap(s=>s.data.filter(p=>finite(p.x)&&finite(p.y)));
    this.empty.hidden=all.length>0;this.points=[];
    let minX=all.length?all.reduce((m,p)=>Math.min(m,p.x),Infinity):0,maxX=all.length?all.reduce((m,p)=>Math.max(m,p.x),-Infinity):5;
    let ys=all.flatMap(p=>[p.y,p.low,p.high].filter(finite).map(Number));
    let minY=ys.length?ys.reduce((a,b)=>Math.min(a,b),Infinity):0,maxY=ys.length?ys.reduce((a,b)=>Math.max(a,b),-Infinity):100;
    if(this.options.zero){minY=Math.min(0,minY);maxY=Math.max(0,maxY);}
    if(finite(this.options.yMin))minY=Number(this.options.yMin);
    if(finite(this.options.yMax))maxY=Number(this.options.yMax);
    const niceStep = range => {const rough=range/5||.2,base=10**Math.floor(Math.log10(rough));return [1,2,5,10].find(v=>v*base>=rough)*base;};
    const ystep=niceStep(maxY-minY);minY=Math.floor(minY/ystep)*ystep;maxY=Math.ceil(maxY/ystep)*ystep;
    if(minY===maxY)maxY=minY+ystep;
    minX=Math.min(0,minX);maxX=Math.max(1,maxX);
    const xstep=Math.max(1,niceStep(maxX-minX));maxX=Math.ceil(maxX/xstep)*xstep;
    const box={left:55,right:w-19,top:17,bottom:h-34};
    const X=x=>box.left+(x-minX)/(maxX-minX)*(box.right-box.left);
    const Y=y=>box.bottom-(y-minY)/(maxY-minY)*(box.bottom-box.top);
    ctx.font="10px Segoe UI, Microsoft YaHei, sans-serif";ctx.lineWidth=1;
    for(let v=minY;v<=maxY+ystep*.01;v+=ystep){
      const y=Y(v);ctx.strokeStyle="#2a3a4e77";ctx.beginPath();ctx.moveTo(box.left,y);ctx.lineTo(box.right,y);ctx.stroke();
      ctx.fillStyle="#7e95b0";ctx.textAlign="right";if(all.length)ctx.fillText(String(Number(v.toFixed(Math.max(0,-Math.floor(Math.log10(ystep)))))),box.left-10,y+3);
    }
    for(let v=minX;v<=maxX+xstep*.01;v+=xstep){
      ctx.fillStyle="#7e95b0";ctx.textAlign="center";
      if(all.length)ctx.fillText(String(v),X(v),h-13);
    }
    if(this.options.zero&&minY<0&&maxY>0){ctx.strokeStyle="#75869c";ctx.setLineDash([4,5]);ctx.beginPath();ctx.moveTo(box.left,Y(0));ctx.lineTo(box.right,Y(0));ctx.stroke();ctx.setLineDash([]);}
    ctx.save();ctx.beginPath();ctx.rect(box.left,box.top-4,box.right-box.left,box.bottom-box.top+8);ctx.clip();
    for(const series of this.series){
      const pts=series.data.filter(p=>finite(p.x)&&finite(p.y));if(!pts.length)continue;
      ctx.strokeStyle=series.color;ctx.lineWidth=1.7;ctx.globalAlpha=series.dashed?.7:1;ctx.setLineDash(series.dashed?[5,4]:[]);
      ctx.beginPath();pts.forEach((p,i)=>i&&p.segment===pts[i-1].segment?ctx.lineTo(X(p.x),Y(p.y)):ctx.moveTo(X(p.x),Y(p.y)));ctx.stroke();ctx.setLineDash([]);
      for(const p of pts){
        if(finite(p.low)&&finite(p.high)){ctx.globalAlpha=.35;ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(X(p.x),Y(p.low));ctx.lineTo(X(p.x),Y(p.high));ctx.stroke();ctx.globalAlpha=1;}
        if(pts.length<80){ctx.beginPath();ctx.arc(X(p.x),Y(p.y),2.5,0,Math.PI*2);ctx.fillStyle=series.color;ctx.fill();}
        this.points.push({px:X(p.x),py:Y(p.y),point:p,name:series.name,color:series.color});
      }
    }
    ctx.restore();ctx.globalAlpha=1;
  }
  hover(e){
    if(!this.points.length)return;const rect=this.canvas.getBoundingClientRect(),x=e.clientX-rect.left,y=e.clientY-rect.top;
    const closest=this.points.reduce((a,b)=>Math.hypot(b.px-x,b.py-y)<Math.hypot(a.px-x,a.py-y)?b:a);
    if(Math.abs(closest.px-x)>80){this.tooltip.hidden=true;return;}
    const same=this.points.filter(p=>p.point.x===closest.point.x).slice(0,14);
    this.tooltip.textContent=`${this.options.xTitle} ${closest.point.x}\n`+same.map(p=>`${p.name}  ${num(p.point.y,3)}${finite(p.point.low)?`  [${num(p.point.low)}, ${num(p.point.high)}]`:""}`).join("\n");
    this.tooltip.hidden=false;this.tooltip.style.left=Math.min(Math.max(0,x+15),rect.width-this.tooltip.offsetWidth-4)+"px";
    this.tooltip.style.top=Math.max(0,Math.min(y+12,rect.height-this.tooltip.offsetHeight))+"px";
  }
}
const expectationChart=new LineChart("expectation-chart","expectation-tooltip","expectation-empty");
const deltaChart=new LineChart("delta-chart","delta-tooltip","delta-empty");
const lossChart=new LineChart("loss-chart","loss-tooltip","loss-empty");

function presetRows(row){return row.audit_presets || row.audit?.presets || [];}
function adoptedMean(row,preset){return numberOr(preset.adopted_mean,preset.current_mean,preset.mean,row.promoted?preset.candidate_mean:preset.teacher_mean);}
let teacherMapColumns=[];
function teacherAgeColor(age){
  const stops=[[82,183,131],[155,185,85],[225,185,85],[237,98,94]],t=Math.max(0,Math.min(32,age));
  const segment=t<8?0:t<16?1:2,from=[0,8,16][segment],span=[8,8,16][segment],fraction=(t-from)/span;
  return stops[segment].map((v,i)=>Math.round(v+(stops[segment+1][i]-v)*fraction));
}
function renderTeacherMap(){
  teacherMapColumns=history.filter(r=>r.teacher_age_sample?.teacher_rounds?.length).slice(-128).map(column=>{
    const sample=column.teacher_age_sample;
    const sorted=sample.teacher_rounds.map((birth,i)=>({birth,id:sample.preset_ids[i]})).sort((a,b)=>Number(b.birth)-Number(a.birth)||a.id-b.id);
    return {...column,teacher_age_sample:{teacher_rounds:sorted.map(row=>row.birth),preset_ids:sorted.map(row=>row.id)}};
  });
  const canvas=$("teacher-map"),rows=teacherMapColumns[0]?.teacher_age_sample.teacher_rounds.length||512;
  canvas.width=Math.max(1,teacherMapColumns.length);canvas.height=rows;
  const ctx=canvas.getContext("2d"),pixels=ctx.createImageData(canvas.width,rows);
  teacherMapColumns.forEach((column,x)=>column.teacher_age_sample.teacher_rounds.forEach((birth,y)=>{
    const color=teacherAgeColor(Number(column.round)-Number(birth)),offset=(y*canvas.width+x)*4;
    pixels.data.set([...color,255],offset);
  }));ctx.putImageData(pixels,0,0);
  const first=teacherMapColumns[0],last=teacherMapColumns.at(-1);
  text("teacher-map-start",first?`第 ${first.round} 轮 · ${rows} 个预设采样`:"等待教师记录");
  text("teacher-map-end",last?`第 ${last.round} 轮`:"—");
}
$("teacher-map").onmousemove=e=>{
  if(!teacherMapColumns.length)return;
  const canvas=$("teacher-map"),rect=canvas.getBoundingClientRect(),x=Math.min(teacherMapColumns.length-1,Math.floor((e.clientX-rect.left)/rect.width*teacherMapColumns.length)),column=teacherMapColumns[x];
  const sample=column.teacher_age_sample,y=Math.min(sample.teacher_rounds.length-1,Math.floor((e.clientY-rect.top)/rect.height*sample.teacher_rounds.length)),birth=sample.teacher_rounds[y],tip=$("teacher-map-tooltip");
  tip.textContent=`第 ${column.round} 轮 · 预设 ${sample.preset_ids[y]}\n教师出生 R${birth} · 年龄 ${Number(column.round)-Number(birth)} 轮\n本轮更新 ${column.champion_updated_count??0} / 4,960`;
  tip.hidden=false;tip.style.left=Math.max(0,Math.min(e.clientX-rect.left+12,rect.width-tip.offsetWidth))+"px";tip.style.top=Math.max(0,Math.min(e.clientY-rect.top+12,rect.height-tip.offsetHeight))+"px";
};
$("teacher-map").onmouseleave=()=>$("teacher-map-tooltip").hidden=true;
function renderCharts(){
  renderTeacherMap();
  const bestData=history.filter(r=>finite(r.best_candidate_mean)&&finite(r.round)).map(r=>({x:Number(r.round),y:Number(r.best_candidate_mean)}));
  $("preset-legend").textContent="最佳候选平均期望";
  expectationChart.update([{name:"最佳候选",color:palette[0],data:bestData}],{zero:false,yMin:75,yMax:76});
  deltaChart.update([{name:"候选 − 教师",color:palette[0],data:history.filter(r=>finite(r.primary_delta)).map(r=>({x:Number(r.round),y:Number(r.primary_delta),low:r.ci_low,high:r.ci_high}))}],{zero:true});
  const trainData=[],valData=[];
  for(const row of history){const train=numberOr(row.training_loss,row.train_loss,row.train_metrics?.loss),val=numberOr(row.validation_loss,row.val_loss,row.train_metrics?.val_loss);if(finite(train))trainData.push({x:Number(row.round),y:Number(train)});if(finite(val))valData.push({x:Number(row.round),y:Number(val)});}
  if(finite(latest.round)&&latest.phase==="training"){
    if(finite(latest.training_loss))trainData.push({x:Number(latest.round),y:Number(latest.training_loss)});
    if(finite(latest.validation_loss))valData.push({x:Number(latest.round),y:Number(latest.validation_loss)});
  }
  lossChart.update([{name:"训练损失",color:palette[0],data:trainData},{name:"验证损失",color:palette[1],data:valData}]);
}
function renderTable(rows){
  tableRows=rows;
  const pages=Math.max(1,Math.ceil(rows.length/100));tablePage=Math.min(tablePage,pages-1);
  text("table-page",`${tablePage+1} / ${pages} · ${rows.length.toLocaleString()} 预设`);
  $("table-prev").disabled=tablePage===0;$("table-next").disabled=tablePage>=pages-1;
  const body=$("preset-table");body.replaceChildren();
  if(!rows.length){const tr=document.createElement("tr"),td=document.createElement("td");td.colSpan=8;td.className="empty-row";td.textContent="暂无已完成的评估";tr.append(td);body.append(tr);return;}
  for(const p of rows.slice(tablePage*100,(tablePage+1)*100)){
    const tr=document.createElement("tr"),values=[p.name||p.preset||p.preset_id||"—",p.cap??p.target_total??"—",num(p.teacher_mean),num(p.candidate_mean),signed(p.delta),finite(p.ci_low)?`[${num(p.ci_low)}, ${num(p.ci_high)}]`:"—",p.n??p.pairs??"—",p.candidate_id||"—"];
    values.forEach((v,i)=>{const td=document.createElement("td");td.textContent=String(v);if(i===4)colorBySign(td,p.delta);tr.append(td);});body.append(tr);
  }
}
function renderHistory(){
  const box=$("round-history");box.replaceChildren();
  if(!history.length){const p=document.createElement("p");p.className="muted";p.textContent="等待第一轮完成。";box.append(p);return;}
  for(const row of history.slice(-60).reverse()){
    const item=document.createElement("div");item.className="round-entry";const icon=document.createElement("div");icon.className="round-icon"+(row.promoted?" promoted":"");icon.textContent=row.rules_revision?"↻":row.promoted?"↑":"·";
    const content=document.createElement("div"),title=document.createElement("strong"),detail=document.createElement("p");
    title.textContent=row.initial?"初始教师":`第 ${row.round} 轮 · 更新 ${row.champion_updated_count??0} 个预设`;
    detail.textContent=`候选 ${num(row.candidate_mean)} · 最佳 ${num(row.best_candidate_mean)} · 组合教师 ${num(row.composite_teacher_mean)}`;content.append(title,detail);
    const delta=document.createElement("span");delta.className="round-delta";delta.textContent=row.rules_revision?"规则重测":signed(row.primary_delta);colorBySign(delta,row.primary_delta);item.append(icon,content,delta);box.append(item);
  }
}
function renderStatus(status){
  latest=status;const process=status.process||{},running=process.running!==false,setupStage=process.stage==="gpu_setup",inSetup=setupStage&&running,setup=status.gpu_setup||{},phase=inSetup?"gpu_setup":setupStage&&!running?"stopped":["paused","stopped","error","complete"].includes(status.status)?status.status:status.phase||status.status||"bootstrap";
  renderHardware(status);
  text("round",finite(status.round)?String(status.round).padStart(3,"0"):"—");text("best",displayModel(status.best_model_id));text("teacher","本轮教师 "+displayModel(status.teacher_id));text("candidate","候选 "+displayModel(status.candidate_id));
  text("elapsed",duration(status.elapsed_seconds??0));text("round-elapsed","本轮 "+duration(status.round_elapsed_seconds??0));text("phase-badge",phaseLabels[phase]||phase);
  const recent=history.at(-1)||{};
  text("candidate-mean",num(status.candidate_mean));
  text("best-candidate-mean",num(numberOr(status.best_candidate_mean,recent.best_candidate_mean)));
  text("composite-mean",num(numberOr(status.composite_teacher_mean,recent.composite_teacher_mean)));
  text("champion-label",recent.initial?"上一轮更新冠军预设":`第 ${recent.round} 轮更新冠军预设`);
  text("champion-count",recent.initial?"—":(recent.champion_updated_count??"—"));
  const updated=recent.initial?undefined:recent.champion_updated_count;
  text("teacher-update-rate",finite(updated)?`第 ${recent.round} 轮更新 ${Number(updated).toLocaleString()} / 4,960 · ${num(Number(updated)/4960*100,1)}%`:"等待首轮评测");
  text("teacher-count",numberOr(status.distinct_teacher_count,recent.distinct_teacher_count)??"—");
  const completed=numberOr(status.evaluation_completed_presets,status.evaluation_presets_completed),total=numberOr(status.evaluation_total_presets,status.evaluation_presets_total);
  text("evaluation-count",finite(completed)?`${Number(completed).toLocaleString()} / ${Number(total||4960).toLocaleString()} 预设`:"4,960 预设 × 256 局");
  const hits=status.cache_hits,misses=status.cache_misses,rate=numberOr(status.cache_hit_rate,finite(hits)&&finite(misses)&&hits+misses>0?hits/(hits+misses):undefined);
  text("cache",finite(rate)?`缓存命中 ${num(rate*100,1)}%${finite(hits)?` · ${Number(hits).toLocaleString()} 次`:""}`:finite(hits)?`缓存命中 ${Number(hits).toLocaleString()} 次`:"缓存 —");
  text("message",setupStage?(setup.message||"正在检查 GPU 环境，进度窗口保持响应。"):status.message||"进度将随任务自动刷新。");text("samples","样本 "+(finite(status.samples)?Number(status.samples).toLocaleString():"—"));text("workers",!inSetup&&usesGpuGames(status)?"1 个 GPU 协调进程":"工作进程 "+(status.workers??"—"));text("loss","训练损失 "+num(status.training_loss,5));text("validation","验证损失 "+num(status.validation_loss,5));
  const metrics=status.validation_metrics||{},validationDetails=[`验证损失 ${num(status.validation_loss,5)}`];
  if(finite(metrics.targeted_label_regret))validationDetails.push(`目标难例遗憾 ${num(metrics.targeted_label_regret,4)}`);
  if(finite(metrics.hard_label_regret))validationDetails.push(`高遗憾样本 ${num(metrics.hard_label_regret,4)}`);
  if(finite(metrics.advantage_confidence_mean))validationDetails.push(`优势标签平均可信权重 ${num(metrics.advantage_confidence_mean,3)}`);
  if(status.checkpoint_metric){const metricNames={targeted_label_regret:"目标难例遗憾",hard_label_regret:"高遗憾样本",selection_loss:"选模损失",label_regret:"动作遗憾",validation_loss:"验证损失"};validationDetails.push(`检查点 ${metricNames[status.checkpoint_metric]||status.checkpoint_metric} ${num(status.checkpoint_score,4)}`);}
  if(finite(status.checkpoint_epoch))validationDetails.push(`${status.evaluation_mode==="six_checkpoint_official_v1"?"正式评测":"开发整局比较"} ${status.checkpoint_candidate_index??1}/${status.checkpoint_candidates??1} · epoch ${status.checkpoint_epoch}`);
  if(finite(status.checkpoint_selection_epoch))validationDetails.push(`本轮选中 epoch ${status.checkpoint_selection_epoch}`);
  text("validation",validationDetails.join(" · "));
  const last=history.at(-1)||{};
  const previous=history.at(-2);
  const delta=!last.initial&&finite(last.candidate_mean)&&finite(previous?.best_candidate_mean)?Number(last.candidate_mean)-Number(previous.best_candidate_mean):undefined;
  text("delta",signed(delta));colorBySign($("delta"),delta);
  text("ci",finite(delta)?`第 ${last.round} 轮 · ${last.candidate_id} − ${last.teacher_id}`:"等待完整评测");
  let progress=inSetup?setup.progress:numberOr(status.phase_progress,finite(status.phase_total)&&status.phase_total>0?status.phase_completed/status.phase_total:undefined);
  if(finite(progress)){progress=Number(progress);if(progress>1)progress/=100;progress=Math.max(0,Math.min(1,progress));text("phase-percent",`${Math.round(progress*100)}%`);$("progress-bar").style.width=`${progress*100}%`;}else{text("phase-percent","—");$("progress-bar").style.width="0%";}
  const eta=inSetup?setup.eta_seconds:status.eta_seconds;text("eta",finite(eta)?"本阶段剩余约 "+duration(eta):phase==="stopped"?"已保存，等待下次启动":"预计剩余时间计算中");
  for(const el of document.querySelectorAll("[data-stage]")){const i=Number(el.dataset.stage),current=stageIndices[phase];el.classList.toggle("done",current!==undefined&&i<current);el.classList.toggle("active",i===current);}
  const paused=phase==="paused"||status.pause_requested;$("pause").textContent=!running?"继续运行":paused?"继续":"暂停";$("pause").disabled=inSetup||(running&&status.stop_requested);$("stop").disabled=!running||status.stop_requested;$("stop").textContent=status.stop_requested&&running?"正在安全停止…":"保存并停止";
  if(status.error)notice(String(status.error),true);
  else if(!running&&process.exit_code)notice(`训练进程异常结束（退出码 ${process.exit_code}）。日志保存在任务目录的 worker.log；修复后再次双击启动器继续。`,true);
  else if(!running)notice("任务已停止，模型与进度已保留。点击“继续运行”或再次双击启动器即可恢复。");
  else if(status.stop_requested)notice("已收到停止请求，正在等待安全保存点。请保留启动器窗口。");
  else if(paused)notice("训练已暂停或正在到达暂停点。点击“继续”恢复；当前教师和本轮进度保留。");
  else notice("");
  if(!status.preset_results_path){const rows=status.preset_results?.length?status.preset_results:last.presets||last.preset_results||[];renderTable(rows);}
  if(phase==="training")renderCharts();
}
function renderHardware(status){
  const hardware=status.hardware||{},cpu=hardware.cpu||{},processes=hardware.processes||{},gpu=hardware.gpu||{},device=gpu.devices?.[0],setup=status.gpu_setup||{};
  text("cpu-util",finite(cpu.utilization_percent)?num(cpu.utilization_percent,1)+"%":"—");text("cpu-name",cpu.name||"CPU 检测中");
  text("cpu-detail",[finite(cpu.physical_cores)?cpu.physical_cores+" 核":null,finite(cpu.logical_cores)?cpu.logical_cores+" 线程":null,cpu.reason].filter(Boolean).join(" · "));
  text("gpu-util",finite(device?.utilization_percent)?num(device.utilization_percent,0)+"%":"—");text("gpu-name",device?.name||"GPU 暂不可读取");
  text("gpu-memory",device?`显存 ${finite(device.memory_used_mib)?num(device.memory_used_mib/1024,2):"—"} / ${finite(device.memory_total_mib)?num(device.memory_total_mib/1024,2):"—"} GiB · 驱动 ${device.driver_version||"—"}`:gpu.reason||"正在检测 NVIDIA 驱动");
  const preparing=status.process?.stage==="gpu_setup"&&status.process?.running!==false,sampling=status.phase==="sampling",labeling=status.phase==="labeling",evaluating=["evaluating","auditing"].includes(status.phase),gpuGames=!preparing&&usesGpuGames(status),cpuPhase=status.phase==="tuning",actual=preparing?null:sampling?status.sampling_backend_actual||"numpy":labeling?status.label_backend_actual:evaluating?status.evaluation_backend_actual:cpuPhase?"numpy":status.training_backend_actual;
  text("actual-backend",actual==="torch_cuda"?"CUDA / PyTorch":actual==="numpy"?"CPU / NumPy":actual||((setup.state==="ready")?"GPU 已就绪":setup.state==="cpu"?"CPU 已选择":"等待确认"));
  text("actual-device",(!preparing&&(sampling?status.sampling_device||(actual==="numpy"?"CPU / NumPy FP32":null):labeling?status.label_device:evaluating?status.evaluation_device:cpuPhase?"CPU / NumPy FP32":status.training_device))||((setup.state==="ready")?"已通过 CUDA 运算，等待计算阶段":"尚未收到计算后端报告"));
  const gpuDecision=labeling?status.label_decision_mode:status.evaluation_decision_mode;
  const gpuReason=sampling?"完整轨迹 GPU 生成；GPU 原生策略 / 独立探索流"+(finite(status.sampling_batch_families)?`；每批 ${status.sampling_batch_families} 条轨迹`:""):(evaluating?"GPU 完整配对对局":"GPU 完整对局生成")+"；"+(gpuDecision==="gpu"?"GPU 原生 FP32 决策":"FP32 策略推理")+(evaluating&&finite(status.evaluation_batch_pairs)?`；每批 ${status.evaluation_batch_pairs} 配对`:"");
  text("backend-reason",gpuGames?gpuReason:evaluating?actual==="torch_cuda"?"原 Python 路径：CPU 模拟 + 双模型 GPU 推理；近似平局由 CPU 复核":"CPU / NumPy FP32 配对整局评测":labeling&&actual==="torch_cuda"?"原 Python 路径：CPU 模拟 + 共享 GPU 推理；近似平局由 CPU 复核":!preparing&&sampling?"来源轨迹采样使用 CPU FP32 策略":!preparing&&cpuPhase?"CPU 工作进程吞吐校准":(!preparing&&(labeling?status.label_backend_reason:status.backend_reason))||setup.reason||"CUDA 用于来源轨迹、完整根标签、配对评测与网络训练。");
  if(!preparing&&status.phase==="training"&&(status.training_precision||status.training_tf32!==undefined)){
    const precision=[status.training_precision,status.training_tf32===undefined?null:`TF32 ${status.training_tf32?"开":"关"}`].filter(Boolean).join(" · ");
    text("backend-reason",[$("backend-reason").textContent,precision].filter(Boolean).join(" · "));
  }
  if(!preparing&&status.phase==="mining")text("backend-reason","从历史训练标签挖掘误选类型，供本轮困难根采样使用");
  const producing=status.process?.running!==false&&status.process?.stage!=="gpu_setup"&&status.status!=="paused";
  text("throughput",producing&&finite(status.phase_units_per_second)?num(status.phase_units_per_second,2)+(sampling?" 轨迹/秒":labeling?" 根/秒":evaluating?(status.catalogue_mode?" 单模型局/秒":" 配对/秒"):" 项/秒"):"—");
  text("process-count",`训练进程树 ${finite(processes.training_process_count)?processes.training_process_count:"—"} · ${gpuGames?"1 个 GPU 协调进程":"配置工作进程 "+(status.workers??"—")}`);
  text("throughput-note",processes.reason||"本阶段实际完成单位 / 秒；不含 GPU 安装时间");
  text("hardware-updated",hardware.sampled_at?new Date(hardware.sampled_at*1000).toLocaleTimeString():"正在检测");
  const active=["checking","downloading","verifying","installing"].includes(setup.state),failed=setup.state==="cpu"&&setup.reason&&!String(setup.reason).includes("CPU mode requested");
  $("gpu-setup-panel").hidden=!active&&!failed&&setup.state!=="cancelled";
  text("gpu-setup-title",active?"首次 GPU 准备 · 完成后自动训练":setup.state==="cancelled"?"GPU 准备已停止 · 分段下载已保留":"当前使用 CPU");text("gpu-setup-time",finite(setup.elapsed_seconds)?"准备耗时 "+duration(setup.elapsed_seconds):"");
  text("gpu-setup-message",setup.message||"");
  $("gpu-setup-progress").style.width=finite(setup.progress)?`${Math.max(0,Math.min(1,setup.progress))*100}%`:"0%";
  let detail=setup.reason||setup.last_output||"安装或验证时无法预估剩余时间；可使用“保存并停止”。";
  if(setup.state==="downloading")detail=`${num((setup.downloaded_bytes||0)/1e9,2)} / ${num((setup.total_bytes||0)/1e9,2)} GB · ${num((setup.bytes_per_second||0)/1e6,2)} MB/s${finite(setup.eta_seconds)?" · 剩余约 "+duration(setup.eta_seconds):""}`;
  text("gpu-setup-detail",detail);
}
function renderEvents(rows){
  const box=$("events"),atBottom=box.scrollHeight-box.scrollTop-box.clientHeight<35;box.replaceChildren();
  if(!rows.length){const p=document.createElement("p");p.className="muted";p.textContent="等待进程日志。";box.append(p);return;}
  for(const row of rows){const item=document.createElement("div");item.className="event";const tm=document.createElement("time"),msg=document.createElement("span");let date;if(typeof row.time==="number")date=new Date(row.time*1000);else date=new Date(row.time||"");tm.textContent=Number.isNaN(date.valueOf())?String(row.time||"").slice(-8):date.toLocaleTimeString();msg.className=["error","warn","warning"].includes(row.level)?(row.level==="error"?"error":"warn"):"info";msg.textContent=String(row.message||"");item.append(tm,msg);box.append(item);}if(atBottom)box.scrollTop=box.scrollHeight;
}
$("table-prev").onclick=()=>{tablePage--;renderTable(tableRows);};
$("table-next").onclick=()=>{tablePage++;renderTable(tableRows);};
async function control(action){
  try{await api("/api/control",{method:"POST",headers:{"Content-Type":"application/json","X-Stella-Token":token},body:JSON.stringify({action})});await refresh();}catch(e){notice(e.message,true);}
}
$("pause").onclick=()=>control(latest.process?.running===false?"restart":latest.pause_requested||latest.status==="paused"?"resume":"pause");$("stop").onclick=()=>control("stop");
let refreshing=false;
async function refresh(){
  if(refreshing)return;refreshing=true;
  try{
    if(!token){const session=await api("/api/session");token=session.token;text("run-path",session.run_dir);$("run-path").title=session.run_dir;}
    const [status,hist,events]=await Promise.all([api("/api/status"),api("/api/history"),api("/api/events")]);
    history=(hist.rows||[]).filter(row=>row.rules_version===status.rules_version);
    const completedVersion=history.at(-1)?.round??(status.rules_effective_round?status.rules_effective_round-1:undefined);
    if(status.preset_results_path&&completedVersion!==undefined&&completedVersion!==presetResultsVersion){
      const presets=await api("/api/presets");
      if(presets.round===completedVersion&&presets.rules_version===status.rules_version){presetResultsVersion=completedVersion;renderTable(presets.presets||[]);}
    }
    renderStatus(status);const signature=JSON.stringify(history);if(signature!==lastHistorySignature){lastHistorySignature=signature;renderCharts();renderHistory();}renderEvents(events.rows||[]);
    connectionFailures=0;text("connection",status.process?.running===false?"进程已结束":"实时连接");$("connection-dot").className="online";text("updated",new Date().toLocaleTimeString());
  }catch(e){connectionFailures++;text("connection","连接中断");$("connection-dot").className="offline";if(connectionFailures>2)notice("无法连接本地进度窗口。请查看启动器是否仍在运行；再次双击启动器可恢复任务。",true);}finally{refreshing=false;}
}
refresh();setInterval(refresh,2000);
