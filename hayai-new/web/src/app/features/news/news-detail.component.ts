import { Component, OnInit, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ActivatedRoute, RouterLink } from '@angular/router';
import { ApiService } from '../../core/services/api.service';

@Component({
  selector: 'app-news-detail',
  standalone: true,
  imports: [CommonModule, RouterLink],
  template: `
    <div style="display: flex; flex-direction: column; gap: 1.5rem;">
      <div class="hud-card" style="padding: 2rem;">
        <a routerLink="/news" style="font-family: 'JetBrains Mono'; font-size: 0.8rem; color: #65a30d; text-decoration: none; text-transform: uppercase;">&larr; Torna alle notizie</a>

        <div *ngIf="loading()" style="padding: 2rem; text-align: center; font-family: 'Rajdhani'; color: #64748b; font-size: 1.1rem;">
          Caricamento notizia...
        </div>

        <ng-container *ngIf="!loading() && news()">
          <div style="display: flex; align-items: center; gap: 0.75rem; margin-top: 1rem;">
            <span style="padding: 0.25rem 0.75rem; background: #f7fee7; color: #365314; border: 1px solid #bef264; font-family: 'JetBrains Mono'; font-size: 0.75rem; font-weight: bold; text-transform: uppercase;">{{ news()!.symbol }}</span>
            <span *ngIf="news()!.sector" style="padding: 0.25rem 0.75rem; background: #f1f5f9; color: #334155; border: 1px solid #cbd5e1; font-family: 'JetBrains Mono'; font-size: 0.75rem; text-transform: uppercase;">{{ news()!.sector }}</span>
            <span *ngIf="news()!.area" style="padding: 0.25rem 0.75rem; background: #f1f5f9; color: #334155; border: 1px solid #cbd5e1; font-family: 'JetBrains Mono'; font-size: 0.75rem; text-transform: uppercase;">{{ news()!.area }}</span>
          </div>

          <h1 class="font-display" style="font-size: 1.8rem; font-weight: 800; color: #0f172a; margin-top: 1rem; margin-bottom: 0.5rem; line-height: 1.25;">{{ news()!.title }}</h1>

          <div style="font-family: 'JetBrains Mono'; font-size: 0.8rem; color: #64748b; margin-bottom: 1.5rem;">
            {{ news()!.publisher || 'Editore sconosciuto' }} · {{ formatDate(news()!.published_at) }}
          </div>

          <div *ngIf="news()!.summary" style="font-family: 'Inter'; font-size: 1.05rem; color: #1e293b; background: #f8fafc; border-left: 3px solid #65a30d; padding: 1rem 1.25rem; margin-bottom: 1.5rem; line-height: 1.6;">
            {{ news()!.summary }}
          </div>

          <div *ngIf="news()!.impact_score != null" style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1.5rem; padding: 1rem; border: 1px solid #cbd5e1; border-radius: 4px; background: #fff;">
            <span [style.background]="sentimentColor(news()!.impact_score)" style="display: inline-block; width: 12px; height: 12px; border-radius: 50%;"></span>
            <div>
              <div style="font-family: 'JetBrains Mono'; font-size: 0.85rem; font-weight: bold; color: #0f172a; text-transform: uppercase;">
                Impatto IA: {{ news()!.impact_score | number:'1.1-1' }} ({{ scoreLabel(news()!.impact_score) }})
                <span *ngIf="news()!.confidence"> · confidenza {{ (news()!.confidence * 100) | number:'1.0-0' }}%</span>
                <span *ngIf="news()!.catalyst"> · Catalizzatore: {{ news()!.catalyst }}</span>
                <span *ngIf="news()!.impact_duration"> · Durata: {{ durationLabel(news()!.impact_duration) }}</span>
              </div>
              <div *ngIf="news()!.impact_surface" style="font-family: 'JetBrains Mono'; font-size: 0.75rem; color: #64748b; margin-top: 0.25rem; text-transform: uppercase;">
                Aree colpite: {{ news()!.impact_surface }}
              </div>
              <div style="font-family: 'Rajdhani'; font-size: 1rem; color: #334155; margin-top: 0.25rem;">
                {{ news()!.ai_rationale }}
              </div>
            </div>
          </div>

          <a [href]="news()!.link" target="_blank" rel="noopener" class="hud-btn"
            style="display: inline-block; padding: 0.6rem 1.5rem; background: #65a30d; color: #fff; text-decoration: none; font-family: 'JetBrains Mono'; font-size: 0.85rem; font-weight: bold; text-transform: uppercase;">
            Leggi la notizia originale →
          </a>
        </ng-container>

        <div *ngIf="!loading() && !news()" style="padding: 2rem; text-align: center; font-family: 'Rajdhani'; color: #dc2626; font-size: 1.1rem;">
          Notizia non trovata.
        </div>
      </div>
    </div>
  `
})
export class NewsDetailComponent implements OnInit {
  news = signal<any>(null);
  loading = signal(true);

  constructor(private api: ApiService, private route: ActivatedRoute) {}

  ngOnInit() {
    const id = Number(this.route.snapshot.paramMap.get('id'));
    if (!id) {
      this.loading.set(false);
      return;
    }
    this.api.getNewsDetail(id).subscribe({
      next: (res) => {
        this.news.set(res);
        this.loading.set(false);
      },
      error: (err) => {
        console.error(err);
        this.loading.set(false);
      }
    });
  }

  formatDate(d: any) {
    if (!d) return '—';
    const dt = new Date(d);
    if (isNaN(dt.getTime())) return String(d);
    return dt.toLocaleDateString('it-IT', { day: '2-digit', month: 'long', year: 'numeric' }) + ' · ' + dt.toLocaleTimeString('it-IT', { hour: '2-digit', minute: '2-digit' });
  }

  sentimentColor(score: any) {
    if (score == null) return '#94a3b8';
    if (score > 0.5) return '#16a34a';
    if (score < -0.5) return '#dc2626';
    return '#eab308';
  }

  scoreLabel(score: any) {
    if (score == null) return 'Non analizzata';
    if (score > 0) return 'Rialzista';
    if (score < 0) return 'Ribassista';
    return 'Neutrale';
  }

  durationLabel(d: any) {
    const map: any = { brief: 'breve', medium: 'media', long: 'lunga' };
    return map[d] || 'media';
  }
}
