import { Routes } from '@angular/router';
import { DashboardComponent } from './features/dashboard/dashboard.component';
import { RecommendationsComponent } from './features/recommendations/recommendations.component';
import { SignalsComponent } from './features/signals/signals.component';
import { NewsComponent } from './features/news/news.component';

export const routes: Routes = [
  { path: '', component: DashboardComponent },
  { path: 'recommendations', component: RecommendationsComponent },
  { path: 'signals', component: SignalsComponent },
  { path: 'news', component: NewsComponent },
  { path: '**', redirectTo: '' }
];
