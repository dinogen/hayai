import { inject } from '@angular/core';
import { CanActivateFn, Router } from '@angular/router';
import { map } from 'rxjs';
import { AuthService } from '../services/auth.service';

export const authGuard: CanActivateFn = () => {
  const auth = inject(AuthService);
  const router = inject(Router);

  return auth.checkAuth().pipe(
    map((res) => {
      if (res.authenticated) {
        return true;
      }
      router.navigate(['/login']);
      return false;
    })
  );
};
