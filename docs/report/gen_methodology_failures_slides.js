const pptxgen = require("pptxgenjs");
const p = new pptxgen();
p.layout = "LAYOUT_WIDE";            // 13.33 x 7.5
const W = 13.33, H = 7.5;

const NAVY="1E2761", INK="2B2B2B", MUTED="6B7280", LIGHT="F4F6F9", LINE="E3E7ED";
const CG="0072B2", MAD="D55E00", LCF="009E73", WIN="1E8449", FAIL="C0392B", AMBER="B9770E";
const HEAD="Cambria", BODY="Calibri";

function bg(s){ s.background={color:"FFFFFF"}; }
function title(s,t){
  s.addText(t,{x:0.55,y:0.33,w:12.2,h:0.7,fontFace:HEAD,fontSize:30,bold:true,color:NAVY,align:"left",isTextBox:true,margin:0});
}
function chip(s,x,y,label,col){
  s.addShape(p.ShapeType.roundRect,{x,y,w:0.42,h:0.42,rectRadius:0.21,fill:{color:col}});
  s.addText(label,{x,y,w:0.42,h:0.42,fontFace:HEAD,fontSize:13,bold:true,color:"FFFFFF",align:"center",valign:"middle",isTextBox:true,margin:0});
}
function card(s,x,y,w,h){
  s.addShape(p.ShapeType.roundRect,{x,y,w,h,rectRadius:0.06,fill:{color:LIGHT},line:{color:LINE,width:1}});
}
function bullets(s,items,x,y,w,h,sz){
  const runs=items.map((t,i)=>({text:t,options:{bullet:{indent:14},breakLine:true,paraSpaceAfter:5}}));
  s.addText(runs,{x,y,w,h,fontFace:BODY,fontSize:sz||12.5,color:INK,align:"left",valign:"top",isTextBox:true,margin:4,lineSpacingMultiple:1.0});
}

// ---------- SLIDE 1 : where each monitor fails ----------
let s=p.addSlide(); bg(s); title(s,"Where each monitor fails and why");
const cols=[
 {h:"ConfGuard",sub:"reads output confidence only",col:CG,x:0.55,b:[
   "Clean code and driving output also lock, so benign inputs raise false alarms",
   "Every output can lock, so the score loses its range",
   "A short or variable payload locks less than clean code, so the score inverts",
   "A short wrong answer leaves no lock, so semantic defection stays invisible"]},
 {h:"MAD",sub:"reads one decision-token activation",col:MAD,x:4.79,b:[
   "The observation trigger re-converges by the decision token, so Watch-OA fails",
   "The one blind fixed layer misses the best layer",
   "The threshold sits on the reference, so the false-alarm rate blows up on driving",
   "The signal is present at a different token, not the decision token"]},
 {h:"LCF",sub:"reads all-layer velocity at the decision token",col:LCF,x:9.03,b:[
   "The threshold calibrates on the reference, so the false-alarm rate blows up",
   "It fingerprints the trigger input, so a clean model earns the same score",
   "Saturated driving output makes the aggregate score explode",
   "It ranks the border group above the trigger and misses semantic defection"]},
];
for(const c of cols){
  card(s,c.x,1.25,3.75,5.6);
  chip(s,c.x+0.22,1.5,c.h[0],c.col);
  s.addText(c.h,{x:c.x+0.78,y:1.5,w:2.8,h:0.42,fontFace:HEAD,fontSize:19,bold:true,color:c.col,valign:"middle",isTextBox:true,margin:0});
  s.addText(c.sub,{x:c.x+0.22,y:2.0,w:3.35,h:0.5,fontFace:BODY,fontSize:11.5,italic:true,color:MUTED,isTextBox:true,margin:0});
  bullets(s,c.b,c.x+0.18,2.55,3.4,4.1,12.5);
}

// ---------- SLIDE 2 : MAD improvements tried ----------
s=p.addSlide(); bg(s); title(s,"MAD improvements we tried");
const mi=[
 {n:"1",h:"Better threshold calibration",d:["MAD sets the cutoff on the clean reference","The runtime normal inputs score higher than the reference","I set the cutoff on the runtime normal inputs"]},
 {n:"2",h:"Label-free layer selection",d:["MAD picks one fixed layer","The best layer changes with each install","I compare the fixed layer against the best layer"]},
 {n:"3",h:"Context-span pooling",d:["MAD reads only the decision token","The observation signal re-converges by that token","I pool the last 64 tokens against the same reference"]},
];
let y=1.35;
for(const m of mi){
  card(s,0.55,y,12.2,1.68);
  chip(s,0.82,y+0.28,m.n,MAD);
  s.addText(m.h,{x:1.42,y:y+0.22,w:4.3,h:0.5,fontFace:HEAD,fontSize:17,bold:true,color:NAVY,valign:"middle",isTextBox:true,margin:0});
  bullets(s,m.d,5.9,y+0.18,6.6,1.35,13);
  y+=1.86;
}

// ---------- SLIDE 3 : MAD results ----------
s=p.addSlide(); bg(s); title(s,"MAD results");
const mr=[
 {big:"1.00 to 0.05",col:WIN,h:"Threshold calibration",d:["Driving false-alarm rate falls from 1.00 to 0.05","Detection stays at 1.00","It fixes the calibration class and nothing else"]},
 {big:"0.68 to 0.81",col:AMBER,h:"Layer selection",d:["Watch-OA-3b rises from 0.34 to 0.67","AgentGhost rises from 0.68 to 0.81","The best layer is an oracle, so a label-free selector is the open step"]},
 {big:"0.15 to 0.66",col:WIN,h:"Context-span pooling",d:["Watch-OA 7B blind score rises from 0.15 to 0.66","The missing signal is at a different token, not a different layer"]},
];
let x=0.55;
for(const m of mr){
  card(s,x,1.3,3.95,5.3);
  s.addText(m.h,{x:x+0.22,y:1.55,w:3.5,h:0.5,fontFace:HEAD,fontSize:16,bold:true,color:NAVY,isTextBox:true,margin:0});
  s.addText(m.big,{x:x+0.2,y:2.15,w:3.6,h:0.9,fontFace:HEAD,fontSize:30,bold:true,color:m.col,align:"left",isTextBox:true,margin:0});
  s.addText("blind AUROC",{x:x+0.22,y:3.0,w:3.5,h:0.3,fontFace:BODY,fontSize:10.5,italic:true,color:MUTED,isTextBox:true,margin:0});
  bullets(s,m.d,x+0.18,3.5,3.6,2.9,13);
  x+=4.26;
}

// ---------- SLIDE 4 : ConfGuard improvements tried ----------
s=p.addSlide(); bg(s); title(s,"ConfGuard improvements we tried");
const ci=[
 {t:"I0",h:"Hold the token budget constant",d:"A longer output gives a longer run, so I score every run at one budget"},
 {t:"I1",h:"Calibrate the threshold per model",d:"Some clean models lock on their own, so I set the threshold at each model's own benign floor"},
 {t:"I2",h:"Confidence contrast against a clean model",d:"I subtract the clean model confidence from the audited model confidence"},
 {t:"I3",h:"Use the full output distribution",d:"I lock on the top-1 to top-2 gap and on low entropy"},
 {t:"I4",h:"Use the position of the lock",d:"I add the onset position and the count of locked spans"},
];
y=1.3;
for(const c of ci){
  card(s,0.55,y,12.2,1.0);
  chip(s,0.8,y+0.29,c.t,CG);
  s.addText(c.h,{x:1.4,y:y+0.1,w:4.9,h:0.8,fontFace:HEAD,fontSize:15.5,bold:true,color:NAVY,valign:"middle",isTextBox:true,margin:0});
  s.addText(c.d,{x:6.5,y:y+0.1,w:6.0,h:0.8,fontFace:BODY,fontSize:12.5,color:INK,valign:"middle",isTextBox:true,margin:0});
  y+=1.1;
}

// ---------- SLIDE 5 : ConfGuard results ----------
s=p.addSlide(); bg(s); title(s,"ConfGuard results");
const cr=[
 {badge:"hygiene",bc:AMBER,t:"I0",d:"The gemma benign false-alarm rate rose from 0.07 to 0.88 with the budget, so I fix the budget before I compare families"},
 {badge:"works",bc:WIN,t:"I1",d:"The mean false-alarm rate fell from 0.70 to 0.21 on every model, strong backdoors stayed visible and weak backdoors became honest misses"},
 {badge:"fails",bc:FAIL,t:"I2",d:"The contrast reached 0.64 against the 0.95 baseline, and the border group matches the triggered group, so this is a behavioral ceiling"},
 {badge:"no gain",bc:MUTED,t:"I3",d:"The gap test and the entropy test both scored 0.951 against the 0.952 baseline, because clean driving output is degenerate"},
 {badge:"no gain",bc:MUTED,t:"I4",d:"The onset scored 0.58 and the span count scored 0.45 against the 0.50 chance line, so I drop it"},
];
y=1.3;
for(const c of cr){
  card(s,0.55,y,12.2,1.0);
  s.addText(c.t,{x:0.8,y:y+0.1,w:0.8,h:0.8,fontFace:HEAD,fontSize:18,bold:true,color:CG,valign:"middle",isTextBox:true,margin:0});
  s.addShape(p.ShapeType.roundRect,{x:1.6,y:y+0.3,w:1.25,h:0.4,rectRadius:0.2,fill:{color:c.bc}});
  s.addText(c.badge,{x:1.6,y:y+0.3,w:1.25,h:0.4,fontFace:BODY,fontSize:11,bold:true,color:"FFFFFF",align:"center",valign:"middle",isTextBox:true,margin:0});
  s.addText(c.d,{x:3.05,y:y+0.1,w:9.5,h:0.8,fontFace:BODY,fontSize:12.5,color:INK,valign:"middle",isTextBox:true,margin:0});
  y+=1.1;
}

p.writeFile({fileName:"/Users/muahmed/Desktop/Projects/backdoor-coding-agents/docs/report/Methodology_Failures_Slides.pptx"})
 .then(f=>console.log("wrote",f));
