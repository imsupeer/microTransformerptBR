export class BPETokenizer {
  private tokenToId: Record<string, number> = {};
  private idToToken: string[] = [];
  private merges = new Set<string>();
  async load(vocabUrl: string, mergesUrl: string) {
    this.tokenToId = await (await fetch(vocabUrl)).json();
    const mergesText = await (await fetch(mergesUrl)).text();
    mergesText.split('\n').forEach((line) => {
      const t = line.trim();
      if (t) this.merges.add(t);
    });
    this.idToToken = [];
    Object.entries(this.tokenToId).forEach(([tok, id]) => {
      this.idToToken[Number(id)] = tok;
    });
  }
  encode(text: string): number[] {
    const out: number[] = [];
    const wordEncode = (w: string) => {
      const syms = [...w, '</w>'];
      let i = 0;
      while (i < syms.length - 1) {
        const pair = syms[i] + ' ' + syms[i + 1];
        if (this.merges.has(pair)) {
          syms.splice(i, 2, syms[i] + syms[i + 1]);
          i = Math.max(0, i - 1);
        } else i++;
      }
      syms.forEach((s) => out.push(this.tokenToId[s] ?? this.tokenToId['<unk>']));
    };
    text.split(/\s+/).forEach(wordEncode);
    return out;
  }
  decode(ids: number[]): string {
    return ids
      .map((id) => this.idToToken[id] || '<unk>')
      .join('')
      .replace(/<\/w>/g, ' ')
      .trim();
  }
}
