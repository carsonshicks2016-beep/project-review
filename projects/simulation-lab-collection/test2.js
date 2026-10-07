import { GAApp } from './src/main_v2.js';
// mock DOM
global.document = {
  getElementById: () => ({ appendChild: ()=>{}, addEventListener: ()=>{}, classList: {toggle: ()=>{}, remove: ()=>{}, add: ()=>{}} }),
  createElement: () => ({ style: {}, getContext: ()=>{} }),
  body: { appendChild: ()=>{}, removeChild: ()=>{} }
};
global.window = { innerWidth: 800, innerHeight: 600, devicePixelRatio: 1, addEventListener: ()=>{} };
global.localStorage = { getItem: ()=>null, setItem: ()=>{}, removeItem: ()=>{} };

const app = new GAApp();
console.log(app.circuitData);
