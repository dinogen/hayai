import { HttpInterceptorFn } from '@angular/common/http';
import { inject } from '@angular/core';
import { catchError, throwError } from 'rxjs';
import { AuthService } from '../services/auth.service';

export const authInterceptor: HttpInterceptorFn = (req, next) => {
  const auth = inject(AuthService);

  const request = req.clone({ withCredentials: true });

  return next(request).pipe(
    catchError((error) => {
      // Login failures (401) and session checks (/auth/me never 401s) must not
      // trigger a redirect, otherwise we'd loop or hide the error message.
      const isAuthCall = req.url.includes('/auth/login') || req.url.includes('/auth/me');
      if (error.status === 401 && !isAuthCall) {
        auth.handleUnauthorized();
      }
      return throwError(() => error);
    })
  );
};
