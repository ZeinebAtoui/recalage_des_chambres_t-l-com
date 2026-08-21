import { Component, inject } from '@angular/core';
import { RouterLink } from '@angular/router';
import { Auth } from '../../core/auth';
import { UploadCarteSig } from '../../components/upload-carte-sig/upload-carte-sig';

@Component({
  imports: [UploadCarteSig, RouterLink],
  selector: 'app-home',
  styleUrl: './home.scss',
  templateUrl: './home.html',
})
export class Home {
  protected readonly auth = inject(Auth);
}
