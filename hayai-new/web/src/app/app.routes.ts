import { Routes } from '@angular/router';
import { DashboardComponent } from './features/dashboard/dashboard.component';
import { RecommendationsComponent } from './features/recommendations/recommendations.component';
import { HoldingsComponent } from './features/holdings/holdings.component';
import { SignalsComponent } from './features/signals/signals.component';
import { NewsComponent } from './features/news/news.component';
import { NewsDetailComponent } from './features/news/news-detail.component';
import { ConfigComponent } from './features/config/config.component';

export const routes: Routes = [
  { path: '', component: DashboardComponent },
  { path: 'portfolio', component: HoldingsComponent },
  { path: 'recommendations', component: RecommendationsComponent },
  { path: 'signals', component: SignalsComponent },
  { path: 'news', component: NewsComponent },
  { path: 'news/:id', component: NewsDetailComponent },
  { path: 'config', component: ConfigComponent },
  { path: '**', redirectTo: '' }
];
