import { Routes } from '@angular/router';
import { DashboardComponent } from './features/dashboard/dashboard.component';
import { RecommendationsComponent } from './features/recommendations/recommendations.component';
import { HoldingsComponent } from './features/holdings/holdings.component';
import { WatchlistComponent } from './features/watchlist/watchlist.component';
import { SignalsComponent } from './features/signals/signals.component';
import { NewsComponent } from './features/news/news.component';
import { NewsDetailComponent } from './features/news/news-detail.component';
import { ConfigComponent } from './features/config/config.component';
import { LoginComponent } from './features/login/login.component';
import { authGuard } from './core/guards/auth.guard';

export const routes: Routes = [
  { path: 'login', component: LoginComponent },
  { path: '', component: DashboardComponent, canActivate: [authGuard] },
  { path: 'portfolio', component: HoldingsComponent, canActivate: [authGuard] },
  { path: 'watchlist', component: WatchlistComponent, canActivate: [authGuard] },
  {
    path: 'watchlist/:symbol',
    canActivate: [authGuard],
    loadComponent: () => import('./features/instrument/instrument-detail.component').then((m) => m.InstrumentDetailComponent),
  },
  { path: 'recommendations', component: RecommendationsComponent, canActivate: [authGuard] },
  { path: 'signals', component: SignalsComponent, canActivate: [authGuard] },
  { path: 'news', component: NewsComponent, canActivate: [authGuard] },
  { path: 'news/:id', component: NewsDetailComponent, canActivate: [authGuard] },
  { path: 'config', component: ConfigComponent, canActivate: [authGuard] },
  { path: '**', redirectTo: '' }
];
