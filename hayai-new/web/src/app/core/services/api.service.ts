import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class ApiService {
  private baseUrl = 'http://127.0.0.1:8000/api';

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

  saveHoldings(code: string, positions: any[]): Observable<any> {
    return this.http.post(`${this.baseUrl}/portfolios/${code}/holdings/save`, { positions });
  }

  getPortfolioConfig(code: string): Observable<any> {
    return this.http.get(`${this.baseUrl}/portfolios/${code}/config`);
  }

  updatePortfolioConfig(code: string, maxAssets: number): Observable<any> {
    return this.http.post(`${this.baseUrl}/portfolios/${code}/config`, { max_assets: maxAssets });
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
}
