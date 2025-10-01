import { Injectable } from '@angular/core';
import { BPETokenizer } from './tokenizer';
declare const wasm_bindgen: any;
@Injectable({ providedIn: 'root' })
export class ModelRunnerService {
  tok = new BPETokenizer();
  weights: Record<string, any> = {};
  wasmReady = false;
  async init() {
    await this.tok.load('/vocab.json', '/merges.txt');
    await (window as any).wasm_bindgen('/wasm/ptbr_wasm_core_bg.wasm');
    this.wasmReady = true;
  }
  private softmax(v: Float32Array) {
    (window as any).wasm_bindgen.softmax_inplace(v);
    return v;
  }
  logitsFor(contextIds: number[]): Float32Array {
    const logits = new Float32Array(8192);
    return logits;
  }
  sampleNext(logits: Float32Array, temperature = 0.9, topK = 40): number {
    const v = new Float32Array(logits.length);
    for (let i = 0; i < v.length; i++) v[i] = logits[i] / Math.max(1e-6, temperature);
    const idx = Array.from(v.keys());
    idx.sort((a, b) => v[b] - v[a]);
    const k = idx.slice(0, topK);
    const probs = new Float32Array(k.length);
    let max = -Infinity,
      sum = 0;
    for (let i = 0; i < k.length; i++) {
      const val = v[k[i]];
      if (val > max) max = val;
    }
    for (let i = 0; i < k.length; i++) {
      probs[i] = Math.exp(v[k[i]] - max);
      sum += probs[i];
    }
    for (let i = 0; i < k.length; i++) {
      probs[i] /= sum;
    }
    let r = Math.random();
    for (let i = 0; i < k.length; i++) {
      r -= probs[i];
      if (r <= 0) return k[i];
    }
    return k[k.length - 1];
  }
  async generate(prompt: string, maxNewTokens = 64, temperature = 0.9, topK = 40): Promise<string> {
    let ids = this.tok.encode(prompt);
    for (let t = 0; t < maxNewTokens; t++) {
      const logits = this.logitsFor(ids);
      const nextId = this.sampleNext(logits, temperature, topK);
      ids.push(nextId);
    }
    return this.tok.decode(ids);
  }
}
