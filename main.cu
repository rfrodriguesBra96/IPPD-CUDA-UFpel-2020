#include "cuda_runtime_api.h"
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include "cuda_runtime.h"
#include "device_functions.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stdlib.h"
#include <string>
#include <cassert>
#include <iostream>
using namespace std;
#include<opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;
#include <ctime>
#include <sys/time.h>



cv::Mat imageRGBA;
cv::Mat imageGrey;

//Return the number of rows of RGBA photos
size_t numLinhas()
{
	return imageRGBA.rows;
}
//Return the number of photo columns
size_t numColunas()
{
	return imageRGBA.cols;
}

void criaMatriz(uchar4 **inputImage, unsigned char **greyImage, uchar4 **dev_rgbaImage, unsigned char **dev_greyImage, const string &filename)
{
	cv::Mat image;
	image = cv::imread(filename); //lê a imagem 
	if (image.empty())
	{
		cerr << "falha ao ler a imagem" << filename << endl;
		exit(1);
	}
	cv::cvtColor(image, imageRGBA, cv::COLOR_BGR2BGRA);

	//cria uma imagem que vai ser a crinza, com largura e altura igual a imagem de entrada
	imageGrey.create(image.rows, image.cols, CV_8UC1);
	
	//converte Mat em um array
	*inputImage = (uchar4*)imageRGBA.ptr<unsigned char>(0);
	*greyImage = imageGrey.ptr<unsigned char>(0);

	const size_t numPixels = numColunas()*numLinhas(); //tamnho da imagem

	//aloca memoria no device
	cudaMalloc(dev_rgbaImage, sizeof(uchar4)*numPixels);
	cudaMalloc(dev_greyImage, sizeof(unsigned char)*numPixels);
	

	//atribuiu os valores em dev_rgbaImage com os valores da imagem de entrada
	 cudaMemcpy(*dev_rgbaImage, *inputImage, sizeof(uchar4)*numPixels, cudaMemcpyHostToDevice);
}

__global__ void rgba_to_greyscale(uchar4* const rgbaImage, unsigned char *const greyImage, int numLinhas, int numColunas)
{
	// pegar o endereço da tread que vai ser 'convertida' para cinza
	const int id = blockIdx.x*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;
	
	if (id < numLinhas*numColunas)
	{
		const unsigned char R = rgbaImage[id].x;
		const unsigned char G = rgbaImage[id].y;
		const unsigned char B = rgbaImage[id].z;
		greyImage[id] = 0.21f*R + 0.71f*G + 0.07*B;   //altera o valor contido no thread, assim 'pintado' com a tonalidade cinza
	}
}


long wtime();
void imprime_tempo(long start, long end);
void imprime_tempo_total(long start, long end);

int main (int argc, char *argv[]){

	if(argc !=3){
		fprintf (stderr,"informe o executavel, a imagem de entrada e a imagem de saída \n",argv[0]);
		fprintf (stderr,"ex: ./executavel 'imagem_entrada.png' 'imagem_saida.png'\n",argv[0]);

      	exit (1);
	}

	string input_file =  (argv[1]); //imagem de entrada
	string output_file = (argv[2]);	//imagem que sera gravada
	uchar4* h_rgbaImage, *dev_rgbaImage;
	unsigned char *h_greyImage, *dev_greyImage;

	long start_cuda = wtime();
	criaMatriz(&h_rgbaImage, &h_greyImage, &dev_rgbaImage, &dev_greyImage, input_file);
	const int thread = 10;
	const int threads = (numLinhas()*numColunas() + thread - 1) / (thread*thread);
	const int bloco =  128;

	//tempo do calculo na GPU
	 long start = wtime();
	rgba_to_greyscale <<<threads, bloco>>> (dev_rgbaImage, dev_greyImage, numLinhas(), numColunas());
	
	long end = wtime();
	//sincroniza os threads
	cudaDeviceSynchronize();

	//rertona o resultado do Host
	cudaMemcpy(h_greyImage, dev_greyImage, sizeof(unsigned char)*numLinhas()*numColunas(), cudaMemcpyDeviceToHost);
	
	//escreve a imagem de convertida
	cv::Mat outImage(numLinhas(), numColunas(), CV_8UC1, h_greyImage);
	cv::imwrite(output_file, outImage);
	
	long end_cuda = wtime();
            

	//limpa memoria cuda;
	cudaFree(dev_rgbaImage);
	cudaFree(dev_greyImage);

	//imprime tempo de execução 
	imprime_tempo(start, end);
    imprime_tempo_total(start_cuda, end_cuda);
}

/* wall_time */
long wtime() {
	struct timeval t;
	gettimeofday(&t, NULL);
	return t.tv_sec*1000000 + t.tv_usec;
}

void imprime_tempo(long start, long end) {
	double time = (end - start) / 1000000.0;
	printf("Tempo do cálculo: %f segundos\n", time);
}

void imprime_tempo_total(long start, long end) {
	double time = (end - start) / 1000000.0;
	printf("Tempo do cálculo e cópias: %f segundos\n", time);
}
