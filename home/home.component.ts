import { Component } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.css']
})
export class HomeComponent {
  private apiUrl = 'http://localhost:4200/detect';
  private selectedFile: File | null = null;
  public imageUrl: string | null = null;
  public isLoading = false;
  public result: any = null;
  public error: string | null = null;


  constructor(private http: HttpClient) {}

  onFileSelected(event: any) {
    this.selectedFile = event.target.files[0];
    console.log('错误信息如下：');
    if (this.selectedFile) {
  this.imageUrl = URL.createObjectURL(this.selectedFile);
}

  }

  onUpload() {
    if (!this.selectedFile) {
      return;
    }
    const formData = new FormData();
    formData.append('file', this.selectedFile, this.selectedFile.name);
    this.isLoading = true;
    console.log('错误信息如下：');
    this.http.post<any>(this.apiUrl, formData).subscribe(
      res => {
        this.result = res.result;
        this.isLoading = false;
      },
      error => {
      console.log('错误信息如下：');
        console.error(error);  // 打印错误信息
        this.error = '上传图片失败，请重试';
        this.isLoading = false;
      }
    );
  }

  onClear() {
    this.selectedFile = null;
    this.imageUrl = null;
    this.isLoading = false;
    this.result = null;
    this.error = null;
  }
}
