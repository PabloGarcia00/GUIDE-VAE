#import "@preview/fletcher:0.5.1" as fletcher: diagram, node, edge

#set page(width: auto, height: auto, margin: 1cm)

#diagram(
  node-stroke: 1pt,
  edge-stroke: 1pt,
  node-corner-radius: 5pt,
  label-sep: 2pt,
  spacing: (2cm, 1.5cm), // Horizontal, Vertical spacing

  // Configuration
  node((0,0), [config.json], shape: rect, fill: blue.lighten(90%), name: <config>),
  
  // Preprocessing
  node((0,1), [preprocess_lib
.prepare_data], shape: rect, name: <prep>),
  edge(<config>, <prep>, "->", [load]),

  // Data & Condition Objects
  node((-1,2), [Datasets
(train/val)], shape: rect, fill: green.lighten(90%), name: <data>),
  node((1,2), [Conditioner], shape: rect, fill: green.lighten(90%), name: <cond>),
  
  edge(<prep>, <data>, "->"),
  edge(<prep>, <cond>, "->"),

  // Model Initialization
  node((0,2), [CVAE Init], shape: rect, fill: orange.lighten(90%), name: <init>),
  edge(<data>, <init>, "->", [dims]),
  edge(<cond>, <init>, "->", [dims]),

  // Training Process
  node((0,3), [train_model
(Loop)], shape: rect, stroke: 2pt, name: <train>),
  edge(<init>, <train>, "->", [model]),
  edge(<data>, <train>, "->", [loaders], label-pos: 0.2),

  // Internal Training Logic (Sub-graph visualization)
  node((2,3), [Encoder], shape: rect, name: <enc>),
  node((3,3), [Latent z], shape: circle, name: <z>),
  node((4,3), [Decoder], shape: rect, name: <dec>),
  
  edge(<train>, <enc>, "->", [input + cond], label-pos: 0.5, label-side: center),
  edge(<enc>, <z>, "->", [mu, logvar]),
  edge(<z>, <dec>, "->", [sample]),
  
  node((3,4), [Loss Calc
(ELBO)], shape: diamond, fill: red.lighten(90%), name: <loss>),
  edge(<dec>, <loss>, "->", [recon]),
  edge(<enc>, <loss>, "->", [KL]),
  
  node((2,4), [Optimizer], shape: rect, name: <optim>),
  edge(<loss>, <optim>, "->", [backward]),
  edge(<optim>, <train>, "-->", [step]),

  // Outputs
  node((-1,4), [SummaryWriter
(Logs)], shape: rect, fill: purple.lighten(90%), name: <logs>),
  edge(<train>, <logs>, "->", [metrics]),

  node((0,5), [Saved Model
(.pt)], shape: rect, fill: purple.lighten(90%), name: <save>),
  edge(<train>, <save>, "->", [model.save()])
)
