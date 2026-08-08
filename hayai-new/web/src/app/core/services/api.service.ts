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

  getSignals(code: string): Observable<any> {
    return this.http.get(`${this.baseUrl}/portfolios/${code}/signals`);
  }

  getNews(code: string): Observable<any> {
    return this.http.get(`${this.baseUrl}/portfolios/${code}/news`);
  }

  getLatestSummary(code: string): Observable<any> {
    return this.http.get(`${this.baseUrl}/portfolios/${code}/summaries/latest`);
  }
}
