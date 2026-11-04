const pptxgen = require("pptxgenjs");
const fs = require("fs");
const p = new pptxgen();
p.layout = "LAYOUT_WIDE";            // 13.33 x 7.5
const DIR = "/Users/muahmed/Desktop/Projects/backdoor-coding-agents/docs/report/";

const NAVY="1E2761", INK="2B2B2B", MUTED="6B7280", LIGHT="F4F6F9", LINE="E3E7ED";
const CG="0072B2", MAD="D55E00", LCF="009E73", WIN="1E8449", FAIL="C0392B", AMBER="B9770E";
const HEAD="Cambria", BODY="Calibri";

function bg(s){ s.background={color:"FFFFFF"}; }
function title(s,t){
  s.addText(t,{x:0.55,y:0.3,w:12.2,h:0.7,fontFace:HEAD,fontSize:28,bold:true,color:NAVY,align:"left",isTextBox:true,margin:0});
}
function chip(s,x,y,label,col){
  s.addShape(p.ShapeType.roundRect,{x,y,w:0.42,h:0.42,rectRadius:0.21,fill:{color:col}});
  s.addText(label,{x,y,w:0.42,h:0.42,fontFace:HEAD,fontSize:13,bold:true,color:"FFFFFF",align:"center",valign:"middle",isTextBox:true,margin:0});
}
function card(s,x,y,w,h,fill){
  s.addShape(p.ShapeType.roundRect,{x,y,w,h,rectRadius:0.06,fill:{color:fill||LIGHT},line:{color:LINE,width:1}});
}
function bullets(s,items,x,y,w,h,sz){
  const runs=items.map(t=>({text:t,options:{bullet:{indent:14},breakLine:true,paraSpaceAfter:5}}));
  s.addText(runs,{x,y,w,h,fontFace:BODY,fontSize:sz||12.5,color:INK,align:"left",valign:"top",isTextBox:true,margin:4});
}
// place image inside box preserving aspect, centered
function placeImg(s,file,aspect,bx,by,bw,bh){
  let w=bw, h=w/aspect;
  if(h>bh){ h=bh; w=h*aspect; }
  const x=bx+(bw-w)/2, y=by+(bh-h)/2;
  s.addImage({path:DIR+file,x,y,w,h});
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

// ---------- improvement slide helper ----------
function impSlide(method,mcol,tag,name,what,result,resultHi,fig,aspect){
  const s=p.addSlide(); bg(s);
  // title row: method chip + tag + name
  chip(s,0.55,0.34,tag,mcol);
  s.addText(name,{x:1.1,y:0.3,w:11.7,h:0.66,fontFace:HEAD,fontSize:26,bold:true,color:NAVY,valign:"middle",isTextBox:true,margin:0});
  s.addText(method+" improvement",{x:1.12,y:0.92,w:6,h:0.3,fontFace:BODY,fontSize:12,italic:true,color:mcol,isTextBox:true,margin:0});
  // left text column
  s.addText("What we did",{x:0.55,y:1.4,w:4.8,h:0.35,fontFace:HEAD,fontSize:15,bold:true,color:mcol,isTextBox:true,margin:0});
  bullets(s,what,0.5,1.78,4.9,2.1,13);
  s.addText("Result",{x:0.55,y:4.0,w:4.8,h:0.35,fontFace:HEAD,fontSize:15,bold:true,color:NAVY,isTextBox:true,margin:0});
  bullets(s,result,0.5,4.38,4.9,2.3,13);
  // result highlight badge
  card(s,0.55,6.55,4.75,0.62,resultHi.bg);
  s.addText(resultHi.text,{x:0.55,y:6.55,w:4.75,h:0.62,fontFace:HEAD,fontSize:14,bold:true,color:"FFFFFF",align:"center",valign:"middle",isTextBox:true,margin:0});
  // figure right
  card(s,5.65,1.4,7.1,5.78,"FFFFFF");
  placeImg(s,fig,aspect,5.8,1.55,6.8,5.5);
}

// ---------- MAD improvements ----------
impSlide("MAD",MAD,"M1","Threshold calibration",
 ["MAD sets the cutoff on the clean reference","The runtime normal inputs score higher than the reference","I set the cutoff on the runtime normal inputs"],
 ["The driving false-alarm rate falls from 1.00 to 0.05","The detection rate stays at 1.00","The fix clears the calibration class and does nothing for the no-signal cases"],
 {bg:WIN,text:"FPR 1.00 to 0.05, detection 1.00"},"mad_imp1_threshold.png",2.81);

impSlide("MAD",MAD,"M2","Label-free layer selection",
 ["MAD reads one fixed-depth layer","The best layer changes with each install","I compare the fixed layer against the best layer in the network"],
 ["Watch-OA-3b rises from 0.34 to 0.67","Watch-QA rises to 0.83 and AgentGhost rises from 0.68 to 0.81","The best layer is an oracle, so a label-free selector is the open step"],
 {bg:AMBER,text:"AgentGhost 0.68 to 0.81, needs a selector"},"mad_imp2_layer.png",2.11);

impSlide("MAD",MAD,"M3","Context-span pooling",
 ["MAD reads only the decision token","The observation trigger sits in the prefill and re-converges by that token","I pool the last 64 tokens against the same clean reference"],
 ["The Watch-OA 7B blind score rises from 0.15 to 0.66","The decision-token score is at chance even at the best layer","The missing signal is at a different token, not a different layer"],
 {bg:WIN,text:"Watch-OA 7B 0.15 to 0.66"},"watch_oa_madspan.png",2.31);

// ---------- ConfGuard improvements ----------
impSlide("ConfGuard",CG,"I0","Hold the token budget constant",
 ["ConfGuard counts the longest run of confident tokens","A longer output gives a longer run","I score every run at one token budget"],
 ["The gemma benign false-alarm rate rose from 0.07 to 0.88 with the budget","The backdoor did not change","I fix the budget before I compare families"],
 {bg:AMBER,text:"Budget hygiene, do it first"},"confguard_I0_length.png",1.18);

impSlide("ConfGuard",CG,"I1","Calibrate the threshold per model",
 ["The fixed threshold flags clean code and driving output","I measure each model benign run lengths","I set the threshold at the 95th percentile of those lengths"],
 ["The mean false-alarm rate fell from 0.70 to 0.21 on every model","Strong backdoors stayed visible, with AgentGhost at 0.95 and gemma scene at 1.00","Weak backdoors became honest misses"],
 {bg:WIN,text:"FPR 0.70 to 0.21, the robust fix"},"confguard_I1_calibration.png",1.20);

impSlide("ConfGuard",CG,"I2","Confidence contrast against a clean model",
 ["I run the audited model and a clean reference model on the same input","The contrast is the audited confidence minus the reference confidence"],
 ["The contrast reached 0.64 against the 0.95 baseline","The border group scored 0.48 and the triggered group scored 0.51","A near-trigger moves the model as much as a real trigger"],
 {bg:FAIL,text:"0.64 against 0.95, behavioral ceiling"},"confguard_I2_contrast.png",1.35);

impSlide("ConfGuard",CG,"I3","Use the full output distribution",
 ["The baseline locks on the top-1 probability","I add a lock on the top-1 to top-2 gap","I add a lock on low entropy"],
 ["The gap test scored 0.951 and the entropy test scored 0.951 against the 0.952 baseline","The clean driving output is degenerate","A degeneracy test fires in the same place as the top-1 lock"],
 {bg:MUTED,text:"0.951 against 0.952, no gain"},"confguard_I3_degeneracy.png",1.24);

impSlide("ConfGuard",CG,"I4","Use the position of the lock",
 ["The baseline uses only the run length","I add the onset position of the longest run","I add the count of locked spans"],
 ["The onset scored 0.58 and the span count scored 0.45 against the 0.50 chance line","The payload lock has no fixed position","Position carries no usable signal"],
 {bg:MUTED,text:"0.58 and 0.45 near chance, dropped"},"confguard_I4_localization.png",1.27);

p.writeFile({fileName:DIR+"Methodology_Failures_Slides.pptx"}).then(f=>console.log("wrote",f));
