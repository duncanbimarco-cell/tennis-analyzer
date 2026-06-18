/**
 * 网球动作分析器 · 小程序版
 * 流程：VKSession 追踪 → 录制 → 分析引擎 → 图表报告
 */

// ==================== 身体关键点索引 ====================
const KP = {
  NOSE:0, NECK:1,
  RIGHT_SHOULDER:2, RIGHT_ELBOW:3, RIGHT_WRIST:4,
  LEFT_SHOULDER:5,  LEFT_ELBOW:6,  LEFT_WRIST:7,
  RIGHT_HIP:8, RIGHT_KNEE:9, RIGHT_ANKLE:10,
  LEFT_HIP:11,  LEFT_KNEE:12,  LEFT_ANKLE:13,
  RIGHT_EYE:14, LEFT_EYE:15, RIGHT_EAR:16, LEFT_EAR:17,
};

const CONNECTIONS = [
  [KP.RIGHT_SHOULDER,KP.LEFT_SHOULDER], [KP.RIGHT_SHOULDER,KP.RIGHT_HIP],
  [KP.LEFT_SHOULDER,KP.LEFT_HIP],       [KP.RIGHT_HIP,KP.LEFT_HIP],
  [KP.RIGHT_SHOULDER,KP.RIGHT_ELBOW],   [KP.RIGHT_ELBOW,KP.RIGHT_WRIST],
  [KP.LEFT_SHOULDER,KP.LEFT_ELBOW],     [KP.LEFT_ELBOW,KP.LEFT_WRIST],
  [KP.RIGHT_HIP,KP.RIGHT_KNEE],         [KP.RIGHT_KNEE,KP.RIGHT_ANKLE],
  [KP.LEFT_HIP,KP.LEFT_KNEE],           [KP.LEFT_KNEE,KP.LEFT_ANKLE],
  [KP.NECK,KP.NOSE], [KP.NOSE,KP.RIGHT_EYE], [KP.NOSE,KP.LEFT_EYE],
];

// ==================== 工具函数 ====================
function calcAngle(a,b,c){
  if(!a||!b||!c)return null;
  const ba={x:a.x-b.x,y:a.y-b.y};
  const bc={x:c.x-b.x,y:c.y-b.y};
  const dot=ba.x*bc.x+ba.y*bc.y;
  const ma=Math.sqrt(ba.x*ba.x+ba.y*ba.y);
  const mc=Math.sqrt(bc.x*bc.x+bc.y*bc.y);
  if(ma===0||mc===0)return null;
  return Math.round(Math.acos(Math.max(-1,Math.min(1,dot/(ma*mc))))*(180/Math.PI)*10)/10;
}

Page({
  // ===== 数据 =====
  data: {
    state: 'init',       // init | ready | recording | result
    recording: false,
    canRecord: false,
    recTimer: '00:00',
    elbowAngle: '--°',
    angleClass: 'neut',
    frameCount: '0',
    statusLabel: '初始化中…',
    statusClass: 'neut',
    report: {
      score: 0, grade: '', minAngle: 0, minAngleText: '--',
      bentRatio: 0, bentRatioText: '--', bentFrames: 0, totalFrames: 0,
      stdDev: 0, stdDevText: '--',
    },
    verdictItems: [],
  },

  // ===== 内部状态 =====
  vkSession: null,
  canvasCtx: null,
  chartCtx: null,
  canvasW: 0, canvasH: 0,
  lastPoints: null,
  recordedFrames: [],
  recStartTime: 0,
  recTimerInterval: null,
  animFrameId: null,

  // ===== 生命周期 =====
  onReady(){
    const info=wx.getSystemInfoSync();
    this.canvasW=info.windowWidth;
    this.canvasH=info.windowHeight;
    this.initSkeletonCanvas();
    this.initVKSession();
  },

  onUnload(){
    this.stopAll();
  },

  // ===== 骨架 Canvas =====
  initSkeletonCanvas(){
    const q=wx.createSelectorQuery();
    q.select('#skeletonCanvas').fields({node:true,size:true}).exec(res=>{
      if(!res||!res[0]){console.error('Canvas 获取失败');return;}
      const c=res[0].node;
      c.width=this.canvasW;c.height=this.canvasH;
      this.canvasCtx=c.getContext('2d');
    });
  },

  // ===== 图表 Canvas =====
  initChartCanvas(frames){
    const q=wx.createSelectorQuery();
    q.select('#chartCanvas').fields({node:true,size:true}).exec(res=>{
      if(!res||!res[0])return;
      const c=res[0].node;
      const rect=res[0];
      // 获取 chart-wrap 实际宽度
      const q2=wx.createSelectorQuery();
      q2.select('.chart-wrap').boundingClientRect(r=>{
        const cw=r?r.width-20:this.canvasW-60;
        c.width=cw;c.height=220;
        this.chartCtx=c.getContext('2d');
        this.drawChartToCanvas(frames);
      }).exec();
    });
  },

  // ===== VKSession 初始化 =====
  initVKSession(){
    if(typeof wx.createVKSession!=='function'){
      this.setData({statusLabel:'微信版本过低',statusClass:'bad'});
      return;
    }
    try{
      const session=wx.createVKSession({track:{body:{mode:1}}});
      session.on('update',r=>this.onBodyUpdate(r));
      session.on('error',e=>{
        console.error('VKSession error:',e);
        this.setData({statusLabel:'追踪出错',statusClass:'bad'});
      });
      session.start().then(()=>{
        console.log('✅ VKSession 已启动');
        this.setData({state:'ready',canRecord:true,statusLabel:'就绪',statusClass:'good'});
      }).catch(e=>this.handleError(e));
      this.vkSession=session;
    }catch(e){this.handleError(e);}
  },

  handleError(e){
    const msg=(e&&e.errMsg)||String(e);
    console.error('初始化失败:',msg);
    this.setData({statusLabel:'设备不支持',statusClass:'bad'});
  },

  // ===== 身体追踪回调 =====
  onBodyUpdate(result){
    const anchors=result.anchors;
    if(!anchors||anchors.length===0){
      this.drawSkeleton(null);
      return;
    }
    const pts=anchors[0].points||[];
    if(pts.length===0){this.drawSkeleton(null);return;}
    this.lastPoints=pts;

    // 绘制骨架
    this.drawSkeleton(pts);

    // 计算右肘角度
    const sh=pts[KP.RIGHT_SHOULDER];
    const el=pts[KP.RIGHT_ELBOW];
    const wr=pts[KP.RIGHT_WRIST];
    let angle=null;

    if(sh&&el&&wr){
      angle=calcAngle(sh,el,wr);
      this.setData({
        elbowAngle: angle!==null?angle+'°':'--°',
        angleClass: angle!==null?(angle<100?'bad':'good'):'neut',
      });
    } else {
      this.setData({elbowAngle:'--°',angleClass:'neut'});
    }

    // 录制中保存帧
    if(this.data.recording&&angle!==null&&pts.length>0){
      this.recordedFrames.push({
        ts: Date.now()-this.recStartTime,
        points: pts.map(p=>({x:p.x,y:p.y})),
        elbowAngle: angle,
      });
      this.setData({frameCount: this.recordedFrames.length});
    }
  },

  // ===== 骨架绘制 =====
  drawSkeleton(pts){
    const ctx=this.canvasCtx;
    if(!ctx)return;
    const w=this.canvasW,h=this.canvasH;
    ctx.clearRect(0,0,w,h);
    if(!pts||pts.length===0)return;

    // 连线
    for(const[i,j]of CONNECTIONS){
      const a=pts[i],b=pts[j];
      if(!a||!b)continue;
      const ra=(i===KP.RIGHT_SHOULDER&&j===KP.RIGHT_ELBOW)||(i===KP.RIGHT_ELBOW&&j===KP.RIGHT_WRIST);
      ctx.beginPath();ctx.moveTo(a.x*w,a.y*h);ctx.lineTo(b.x*w,b.y*h);
      ctx.strokeStyle=ra?'rgba(255,140,0,0.9)':'rgba(86,171,47,0.5)';
      ctx.lineWidth=ra?4:2.5;ctx.stroke();
    }

    // 关键点
    for(let i=0;i<pts.length;i++){
      const p=pts[i];if(!p)continue;
      const cx=p.x*w,cy=p.y*h;
      const ra=i===KP.RIGHT_SHOULDER||i===KP.RIGHT_ELBOW||i===KP.RIGHT_WRIST;
      ctx.beginPath();ctx.arc(cx,cy,ra?7:4.5,0,Math.PI*2);
      if(i===KP.RIGHT_SHOULDER)ctx.fillStyle='#ff4b4b';
      else if(i===KP.RIGHT_ELBOW)ctx.fillStyle='#ff8c00';
      else if(i===KP.RIGHT_WRIST)ctx.fillStyle='#ffd700';
      else ctx.fillStyle='#56ab2f';
      ctx.fill();
      if(ra){
        ctx.beginPath();ctx.arc(cx,cy,10,0,Math.PI*2);
        ctx.globalAlpha=0.5;ctx.strokeStyle=ctx.fillStyle;ctx.lineWidth=2;ctx.stroke();ctx.globalAlpha=1;
      }
    }

    // 角度标注
    const sh=pts[KP.RIGHT_SHOULDER],el=pts[KP.RIGHT_ELBOW],wr=pts[KP.RIGHT_WRIST];
    if(sh&&el&&wr){
      const ang=calcAngle(sh,el,wr);
      if(ang!==null){
        ctx.font='bold 18px sans-serif';
        ctx.fillStyle=ang<100?'#ff6b6b':'#a8e063';
        ctx.shadowColor='rgba(0,0,0,0.7)';ctx.shadowBlur=4;
        ctx.fillText(ang+'°',el.x*w+14,el.y*h-14);
        ctx.shadowBlur=0;
      }
    }
  },

  // ===== 录制开关 =====
  toggleRecording(){
    if(!this.data.recording){this.startRecording();}
    else{this.stopAndAnalyze();}
  },

  startRecording(){
    this.recordedFrames=[];
    this.recStartTime=Date.now();
    this.setData({recording:true,frameCount:'0',recTimer:'00:00',state:'recording',statusLabel:'🔴 录制中',statusClass:'bad'});
    let sec=0;
    this.recTimerInterval=setInterval(()=>{
      sec++;
      const m=Math.floor(sec/60),s=sec%60;
      this.setData({recTimer:String(m).padStart(2,'0')+':'+String(s).padStart(2,'0')});
    },1000);
  },

  stopAndAnalyze(){
    this.setData({recording:false,statusLabel:'分析中…',statusClass:'neut'});
    if(this.recTimerInterval){clearInterval(this.recTimerInterval);this.recTimerInterval=null;}

    if(this.recordedFrames.length<5){
      this.setData({statusLabel:'数据不足',statusClass:'bad'});
      return;
    }

    const report=this.analyze(this.recordedFrames);
    const verdictItems=this.buildVerdict(report);

    this.setData({
      state:'result',
      canRecord:false,
      statusLabel:'分析完成',
      statusClass:'good',
      report:{
        score:report.score,
        grade:report.grade,
        minAngle:report.minAngle,
        minAngleText:report.minAngle+'°',
        bentRatio:report.bentRatio,
        bentRatioText:report.bentRatio+'%',
        bentFrames:report.bentFrames,
        totalFrames:report.totalFrames,
        stdDev:report.stdDev,
        stdDevText:String(report.stdDev),
      },
      verdictItems,
    });

    // 延迟初始化图表 Canvas（等 DOM 渲染）
    setTimeout(()=>this.initChartCanvas(this.recordedFrames),300);
  },

  // ===== 分析引擎 =====
  analyze(frames){
    const angles=frames.map(f=>f.elbowAngle);
    const minAngle=Math.min(...angles);
    const maxAngle=Math.max(...angles);
    const avgAngle=angles.reduce((a,b)=>a+b,0)/angles.length;
    const bentFrames=angles.filter(a=>a<100).length;
    const bentRatio=bentFrames/angles.length;
    const variance=angles.reduce((s,a)=>s+(a-avgAngle)**2,0)/angles.length;
    const stdDev=Math.round(Math.sqrt(variance)*10)/10;

    // 找挥拍阶段（手腕速度最大处）
    let maxSpeed=0,swingCenter=0;
    for(let i=1;i<frames.length;i++){
      const p=frames[i-1].points[KP.RIGHT_WRIST];
      const c=frames[i].points[KP.RIGHT_WRIST];
      if(!p||!c)continue;
      const dx=c.x-p.x,dy=c.y-p.y;
      const speed=Math.sqrt(dx*dx+dy*dy);
      if(speed>maxSpeed){maxSpeed=speed;swingCenter=i;}
    }
    const ss=Math.max(0,swingCenter-10),se=Math.min(frames.length-1,swingCenter+10);
    const swingAngles=angles.slice(ss,se+1);
    const swingMin=swingAngles.length>0?Math.min(...swingAngles):minAngle;

    // 评分
    let score=100;
    if(bentRatio>0.8)score-=40;else if(bentRatio>0.5)score-=25;else if(bentRatio>0.2)score-=10;
    if(minAngle<70)score-=20;else if(minAngle<85)score-=10;else if(minAngle<100)score-=5;
    if(stdDev>20)score-=15;else if(stdDev>12)score-=8;
    if(swingMin<100)score-=15;
    score=Math.max(0,Math.min(100,Math.round(score)));
    const grade=score>=80?'great':score>=55?'ok':'poor';

    return{
      minAngle,maxAngle,avgAngle:Math.round(avgAngle*10)/10,
      bentFrames,totalFrames:frames.length,
      bentRatio:Math.round(bentRatio*1000)/10,
      stdDev,swingMinAngle:Math.round(swingMin*10)/10,
      score,grade,
      duration:frames[frames.length-1].ts/1000,
    };
  },

  // ===== 诊断建议生成 =====
  buildVerdict(r){
    const items=[];
    if(r.score>=80)items.push({cls:'good',txt:'✅ 整体动作质量优秀，肘关节延展充分'});
    else if(r.score>=55)items.push({cls:'ok',txt:'⚠️ 动作质量一般，存在改善空间'});
    else items.push({cls:'bad',txt:'❌ 动作质量较差，需重点改进肘部延展'});

    if(r.bentRatio>50)items.push({cls:'bad',txt:'⚠️ 超过一半时间肘关节弯曲，击球时请刻意锁死并延展手肘'});
    if(r.minAngle<85)items.push({cls:'bad',txt:'🔴 最小肘角度仅 '+r.minAngle+'°，严重弯曲——击球效率大幅降低'});
    if(r.stdDev>15)items.push({cls:'bad',txt:'📊 肘角度波动大（±'+r.stdDev+'°），动作不够稳定'});
    else if(r.stdDev<8)items.push({cls:'good',txt:'📊 动作角度稳定（±'+r.stdDev+'°），挥拍一致性良好'});
    if(r.swingMinAngle<100)items.push({cls:'bad',txt:'🎯 触球阶段肘角度仅 '+r.swingMinAngle+'°，确保击球瞬间手臂充分延展'});
    items.push({cls:'good',txt:'💡 录制时长 '+r.duration.toFixed(1)+' 秒，共分析 '+r.totalFrames+' 帧'});
    return items;
  },

  // ===== 图表绘制 =====
  drawChartToCanvas(frames){
    const ctx=this.chartCtx;
    if(!ctx||!frames||frames.length<2)return;
    const w=ctx.canvas.width,h=ctx.canvas.height;
    ctx.clearRect(0,0,w,h);

    const angles=frames.map(f=>f.elbowAngle);
    const ml=50,mr=20,mt=30,mb=40;
    const pw=w-ml-mr,ph=h-mt-mb;
    const minA=Math.min(...angles),maxA=Math.max(...angles);
    const rangeA=Math.max(maxA-minA,20);
    const yMin=Math.max(0,minA-rangeA*0.3);
    const yMax=Math.min(180,maxA+rangeA*0.3);

    // 网格
    ctx.strokeStyle='#222';ctx.lineWidth=1;
    for(let i=0;i<=5;i++){const y=mt+(ph/5)*i;ctx.beginPath();ctx.moveTo(ml,y);ctx.lineTo(w-mr,y);ctx.stroke();}
    for(let i=0;i<=4;i++){const x=ml+(pw/4)*i;ctx.beginPath();ctx.moveTo(x,mt);ctx.lineTo(x,h-mb);ctx.stroke();}

    // 100° 阈值线
    const y100=mt+ph-(100-yMin)/(yMax-yMin)*ph;
    ctx.strokeStyle='rgba(255,75,75,0.5)';ctx.lineWidth=1.5;ctx.setLineDash([6,4]);
    ctx.beginPath();ctx.moveTo(ml,y100);ctx.lineTo(w-mr,y100);ctx.stroke();ctx.setLineDash([]);
    ctx.fillStyle='#ff6b6b';ctx.font='10px sans-serif';ctx.fillText('100°',ml+4,y100-4);

    // Y轴
    ctx.fillStyle='#888';ctx.font='10px sans-serif';ctx.textAlign='right';
    for(let i=0;i<=5;i++){const y=mt+(ph/5)*i;ctx.fillText(Math.round(yMax-(yMax-yMin)/5*i)+'°',ml-6,y+4);}
    ctx.textAlign='start';
    ctx.fillText('0s',ml,h-6);
    ctx.fillText((frames[frames.length-1].ts/1000).toFixed(1)+'s',w-mr-30,h-6);

    // 曲线
    ctx.strokeStyle='#ff8c00';ctx.lineWidth=2.5;ctx.shadowColor='rgba(255,140,0,0.4)';ctx.shadowBlur=6;
    ctx.beginPath();
    for(let i=0;i<frames.length;i++){
      const x=ml+(i/(frames.length-1))*pw;
      const y=mt+ph-(angles[i]-yMin)/(yMax-yMin)*ph;
      if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);
    }
    ctx.stroke();ctx.shadowBlur=0;

    // 标题
    ctx.fillStyle='#ccc';ctx.font='12px sans-serif';ctx.textAlign='center';
    ctx.fillText('右肘关节角度曲线 —— 红色虚线 = 100° 阈值',w/2,18);
    ctx.textAlign='start';
  },

  // ===== 再录一次 =====
  recordAgain(){
    this.setData({state:'ready',canRecord:true,frameCount:'0',recTimer:'00:00',elbowAngle:'--°',angleClass:'neut',statusLabel:'就绪',statusClass:'good'});
    this.recordedFrames=[];
  },

  // ===== 重置 =====
  resetAll(){
    if(this.data.recording){
      this.setData({recording:false});
      if(this.recTimerInterval){clearInterval(this.recTimerInterval);this.recTimerInterval=null;}
    }
    this.recordedFrames=[];
    this.setData({
      state:'ready',canRecord:true,recTimer:'00:00',
      frameCount:'0',elbowAngle:'--°',angleClass:'neut',
      statusLabel:'就绪',statusClass:'good',
      report:{score:0,grade:'',minAngle:0,minAngleText:'--',bentRatio:0,bentRatioText:'--',bentFrames:0,totalFrames:0,stdDev:0,stdDevText:'--'},
      verdictItems:[],
    });
  },

  // ===== 停止所有 =====
  stopAll(){
    if(this.vkSession){try{this.vkSession.stop();this.vkSession.destroy();}catch(e){}this.vkSession=null;}
    if(this.animFrameId){cancelAnimationFrame(this.animFrameId);}
    if(this.recTimerInterval){clearInterval(this.recTimerInterval);}
  },

  // ===== 摄像头错误 =====
  onCameraError(e){
    console.error('摄像头错误:',e);
    this.setData({statusLabel:'摄像头失败',statusClass:'bad'});
  },
});
