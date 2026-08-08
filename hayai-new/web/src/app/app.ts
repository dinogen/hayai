import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { NavbarComponent } from './core/components/navbar/navbar.component';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet, NavbarComponent],
  template: `
    <div class="app-shell">
      <app-navbar></app-navbar>
      <main class="app-content">
        <router-outlet></router-outlet>
      </main>
      <footer class="app-footer">
        HAYAI v2 // Personal Quant Terminal & DeepSeek AI Analyst — 5,000 EUR Experiment
      </footer>
    </div>
  `,
  styles: [`
    .app-shell {
      display: flex;
      flex-direction: column;
      min-height: 100vh;
      font-family: 'Rajdhani', sans-serif;
      background: transparent;
    }

    .app-content {
      flex: 1;
      width: 100%;
      max-width: 1200px;
      margin: 0 auto;
      padding: 2rem 1.5rem;
      box-sizing: border-box;
    }

    .app-footer {
      background: #ffffff;
      border-top: 1px solid #cbd5e1;
      padding: 1rem 1.5rem;
      text-align: center;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.75rem;
      color: #64748b;
    }
  `]
})
export class App {}
