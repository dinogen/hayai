import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../../../environments/environment';

@Injectable({
  providedIn: 'root'
})
export class ApiService {
  private baseUrl = environment.apiUrl;

  constructor(private http: HttpClient) {}

  getHealth(): Observable<any> {
    return this.http.get(`${this.baseUrl}/health`);
  }

  getPortfolios(): Observable<any> {
    return this.http.get(`${this.baseUrl}/portfolios`);
  }

  getPortfolioDetail(code: string): Observable<any> {
    return this.http.get(`${this.baseUrl}/portfolios/${code}`);
  }

  getLatestRecommendations(code: string): Observable<any> {
    return this.http.get(`${this.baseUrl}/portfolios/${code}/recommendations/latest`);
  }

  getPortfolioValue(code: string): Observable<any> {
    return this.http.get(`${this.baseUrl}/portfolios/${code}/value`);
  }

  getSignals(code: string): Observable<any> {
    return this.http.get(`${this.baseUrl}/portfolios/${code}/signals`);
  }

  getNews(code: string, params?: any): Observable<any> {
    const qs = params ? '?' + new URLSearchParams(params).toString() : '';
    return this.http.get(`${this.baseUrl}/portfolios/${code}/news${qs}`);
  }

  getNewsDetail(newsId: number): Observable<any> {
    return this.http.get(`${this.baseUrl}/news/${newsId}`);
  }

  getLatestSummary(code: string): Observable<any> {
    return this.http.get(`${this.baseUrl}/portfolios/${code}/summaries/latest`);
  }

  getHoldings(code: string): Observable<any> {
    return this.http.get(`${this.baseUrl}/portfolios/${code}/holdings`);
  }

  getWatchlist(code: string): Observable<any> {
    return this.http.get(`${this.baseUrl}/portfolios/${code}/watchlist`);
  }

  getInstrumentDetail(symbol: string, days?: number): Observable<any> {
    const qs = days ? `?days=${days}` : '';
    return this.http.get(`${this.baseUrl}/instruments/${encodeURIComponent(symbol)}${qs}`);
  }

  saveHoldings(code: string, positions: any[]): Observable<any> {
    return this.http.post(`${this.baseUrl}/portfolios/${code}/holdings/save`, { positions });
  }

  getPortfolioConfig(code: string): Observable<any> {
    return this.http.get(`${this.baseUrl}/portfolios/${code}/config`);
  }

  updatePortfolioConfig(code: string, maxAssets?: number, rebalanceThresholdEur?: number): Observable<any> {
    const payload: any = {};
    if (maxAssets !== undefined) payload.max_assets = maxAssets;
    if (rebalanceThresholdEur !== undefined) payload.rebalance_threshold_eur = rebalanceThresholdEur;
    return this.http.post(`${this.baseUrl}/portfolios/${code}/config`, payload);
  }

  resetPortfolio(code: string, initialCapital: number): Observable<any> {
    return this.http.post(`${this.baseUrl}/portfolios/${code}/reset`, { initial_capital: initialCapital });
  }

  getNewsLlmEnabled(): Observable<any> {
    return this.http.get(`${this.baseUrl}/config/news-llm`);
  }

  updateNewsLlmEnabled(enabled: boolean): Observable<any> {
    return this.http.put(`${this.baseUrl}/config/news-llm`, { news_llm_enabled: enabled });
  }

  getMarketsStatus(): Observable<any> {
    return this.http.get(`${this.baseUrl}/markets/status`);
  }
}
