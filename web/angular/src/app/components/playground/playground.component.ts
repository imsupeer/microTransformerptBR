import { Component } from '@angular/core';
import { ModelRunnerService } from '../../services/model-runner.service';
@Component({
  selector: 'app-playground',
  templateUrl: './playground.component.html',
  styleUrls: ['./playground.component.scss'],
})
export class PlaygroundComponent {
  prompt = 'No Brasil,';
  temperature = 0.9;
  topK = 40;
  tokens = 64;
  output = '';
  busy = false;
  constructor(private runner: ModelRunnerService) {
    this.runner.init().then(() => {});
  }
  async run() {
    this.busy = true;
    try {
      this.output = await this.runner.generate(this.prompt, this.tokens, this.temperature, this.topK);
    } finally {
      this.busy = false;
    }
  }
}
