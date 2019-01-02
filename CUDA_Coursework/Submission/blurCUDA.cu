//blur cuda
#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/time.h>

#define TY 32
#define TX 32


//this kernel does the regular blurring process
__global__
void blurKernel (int *R, int *G, int *B, int *Rnew, int *Gnew, int *Bnew, int rowsize, int colsize)
{
	int row = (blockIdx.y * blockDim.y) + threadIdx.y;
	int col = (blockIdx.x * blockDim.x) + threadIdx.x;
	int x = (row * colsize) + col;
	int xu = x - colsize;
	int xd = x + colsize;
	int xr = x + 1;
	int xl = x - 1;

	if((row < rowsize) && (col < colsize)) {
		if (row != 0 && row != (rowsize-1) && col != 0 && col != (colsize-1)){
			Rnew[x] = (R[xr]+R[xl]+R[xd]+R[xu])/4;
			Gnew[x] = (G[xr]+G[xl]+G[xd]+G[xu])/4;
			Bnew[x] = (B[xr]+B[xl]+B[xd]+B[xu])/4;
		}
		else if (row == 0 && col != 0 && col != (colsize-1)){
			Rnew[x] = (R[xr]+R[xd]+R[xl])/3;
			Gnew[x] = (G[xr]+G[xd]+G[xl])/3;
			Bnew[x] = (B[xr]+B[xd]+B[xl])/3;
		}
		else if (row == (rowsize-1) && col != 0 && col != (colsize-1)){
			Rnew[x] = (R[xl]+R[xr]+R[xu])/3;
			Gnew[x] = (G[xl]+G[xr]+G[xu])/3;
			Bnew[x] = (B[xl]+B[xr]+B[xu])/3;
		}
		else if (col == 0 && row != 0 && row != (rowsize-1)){
			Rnew[x] = (R[xr]+R[xu]+R[xd])/3;
			Gnew[x] = (G[xr]+G[xu]+G[xd])/3;
			Bnew[x] = (B[xr]+B[xu]+B[xd])/3;
		}
		else if (col == (colsize-1) && row != 0 && row != (rowsize-1)){
			Rnew[x] = (R[xd]+R[xl]+R[xu])/3;
			Gnew[x] = (G[xd]+G[xl]+G[xu])/3;
			Bnew[x] = (B[xd]+B[xl]+B[xu])/3;
		}
		else if (row==0 &&col==0){
			Rnew[x] = (R[xd]+R[xr])/2;
			Gnew[x] = (G[xd]+G[xr])/2;
			Bnew[x] = (B[xd]+B[xr])/2;
		}
		else if (row==0 &&col==(colsize-1)){
			Rnew[x] = (R[xd]+R[xl])/2;
			Gnew[x] = (G[xd]+G[xl])/2;
			Bnew[x] = (B[xd]+B[xl])/2;
		}
		else if (row==(rowsize-1) &&col==0){
			Rnew[x] = (R[xu]+R[xr])/2;
			Gnew[x] = (G[xu]+G[xr])/2;
			Bnew[x] = (B[xu]+B[xr])/2;
		}
		else if (row==(rowsize-1) &&col==(colsize-1)){
			Rnew[x] = (R[xu]+R[xl])/2;
			Gnew[x] = (G[xu]+G[xl])/2;
			Bnew[x] = (B[xu]+B[xl])/2;
		}
	}
}


__global__
void copyKernel(int *R, int *B, int *G, int *Rnew, int *Gnew, int *Bnew, int rowsize, int colsize)
{
	int row = (blockIdx.y * blockDim.y) + threadIdx.y;
	int col = (blockIdx.x * blockDim.x) + threadIdx.x;
	int x = (row * colsize) + col;

	if((col < colsize) && (row < rowsize)){
		R[x] = Rnew[x];
		G[x] = Gnew[x];
		B[x] = Bnew[x];
	}
}

int main (int argc, const char * argv[]) {
	static int const maxlen = 200, rowsize = 521, colsize = 428, linelen = 12;
	char str[maxlen], lines[5][maxlen];
	FILE *fp, *fout;
	int nlines = 0;
	unsigned int h1, h2, h3;
	char *sptr;
	int R[rowsize][colsize], G[rowsize][colsize], B[rowsize][colsize];
	int row = 0, col = 0, nblurs, lineno=0, k;
	struct timeval tim;
	
	// 5a. timing reading of image

	gettimeofday(&tim, NULL);
	double t1=tim.tv_sec+(tim.tv_usec/1000000.0);
	fp = fopen ("David.ps", "r");

	while(! feof(fp))
	{
		fscanf(fp, "\n%[^\n]", str);
		if (nlines < 5) {strcpy((char *)lines[nlines++],(char *)str);}
		else{
			for (sptr=&str[0];*sptr != '\0';sptr+=6){
				sscanf(sptr,"%2x",&h1);
				sscanf(sptr+2,"%2x",&h2);
				sscanf(sptr+4,"%2x",&h3);
				
				if (col==colsize){
					col = 0;
					row++;
				}
				if (row < rowsize) {
					R[row][col] = h1;
					G[row][col] = h2;
					B[row][col] = h3;
				}
				col++;
			}
		}
	}
	fclose(fp);

	gettimeofday(&tim, NULL);
	double t2=tim.tv_sec+(tim.tv_usec/1000000.0);
	printf("part 5a Reading Image File took :%.6lf seconds elapsed\n", t2-t1);
	//printf("%.6lf,", t2-t1 ); easy csv output

	nblurs = 160;

	//5b :timing allocation of device memory
	gettimeofday(&tim, NULL);
	double t1b=tim.tv_sec+(tim.tv_usec/1000000.0);

	//intialise int variables
	int *d_R;
	int *d_G;
	int *d_B;

	int *d_Rnew , *d_Gnew , *d_Bnew;

	int sizei = sizeof(int) * (rowsize*colsize);

	//memory allocation

	cudaMalloc((void **)&d_R, sizei);
	cudaMalloc((void **)&d_G, sizei);
	cudaMalloc((void **)&d_B, sizei);
	cudaMalloc((void **)&d_Rnew, sizei);
	cudaMalloc((void **)&d_Gnew, sizei);
	cudaMalloc((void **)&d_Bnew, sizei);


	gettimeofday(&tim, NULL);
	double t2b=tim.tv_sec+(tim.tv_usec/1000000.0);
	printf("part 5b Allocation of device memory took :%.6lf seconds elapsed\n", t2b-t1b);
	//printf("%.6lf,", t2b-t1b );


	//5c : timing Transferring data between host and device mem
	gettimeofday(&tim, NULL);
	double t3=tim.tv_sec+(tim.tv_usec/1000000.0);

	cudaMemcpy(d_R, R, sizei, cudaMemcpyHostToDevice);
	cudaMemcpy(d_G, G, sizei, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, sizei, cudaMemcpyHostToDevice);
	gettimeofday(&tim, NULL);
	double t4=tim.tv_sec+(tim.tv_usec/1000000.0);
	printf("part 5c Transfer of data took :%.6lf seconds elapsed\n", t4-t3);
	//printf("%.6lf,", t4-t3 );


	//5d: time doing the blurring
	gettimeofday(&tim, NULL);
	double t5=tim.tv_sec+(tim.tv_usec/1000000.0);

	dim3 dimGrid(ceil(colsize/(float)TX), ceil(rowsize/(float)TY), 1);
	dim3 dimBlock(32, 32, 1);
	//run blurring
	for (k=0; k < nblurs; k++){
		blurKernel<<<dimGrid,dimBlock>>>(d_R, d_G, d_B, d_Rnew, d_Gnew, d_Bnew, rowsize, colsize);
		copyKernel<<<dimGrid,dimBlock>>>(d_R, d_G, d_B, d_Rnew, d_Gnew, d_Bnew, rowsize, colsize);
	}

	//return data back to host
	cudaMemcpy(R, d_R, sizei, cudaMemcpyDeviceToHost);
	cudaMemcpy(G, d_G, sizei, cudaMemcpyDeviceToHost);
	cudaMemcpy(B, d_B, sizei, cudaMemcpyDeviceToHost);

	cudaFree(d_R); cudaFree(d_G); cudaFree(d_B);
	cudaFree(d_Rnew); cudaFree(d_Gnew); cudaFree(d_Bnew);

	gettimeofday(&tim, NULL);
	double t6=tim.tv_sec+(tim.tv_usec/1000000.0);
	printf("part 5d Blurring took :%.6lf seconds elapsed\n", t6-t5);
	//printf("%.6lf,", t6-t5 );
	


	//5e: time to output blurred image
	gettimeofday(&tim, NULL);
	double t7=tim.tv_sec+(tim.tv_usec/1000000.0);

	fout= fopen("DavidBlur.ps", "w");
	for (k=0;k<nlines;k++) fprintf(fout,"\n%s", lines[k]);
	fprintf(fout,"\n");
	for(row=0;row<rowsize;row++){
		for (col=0;col<colsize;col++){
			fprintf(fout,"%02x%02x%02x",R[row][col],G[row][col],B[row][col]);
			lineno++;
			if (lineno==linelen){
				fprintf(fout,"\n");
				lineno = 0;
			}
		}
	}
	fclose(fout);
    //return 0;


	gettimeofday(&tim, NULL);
	double t8=tim.tv_sec+(tim.tv_usec/1000000.0);
	printf("part 5e Outputting blurre image took :%.6lf seconds elapsed\n", t8-t7);
	//printf("%.6lf,\n", t8-t7 );
	return 0;
}