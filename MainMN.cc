#include <iostream>
#include "readMnist.h"

int main(){

   mnistparams in;
   char img[28*28];
   char *filename = "./MNIST_data/train-images.idx3-ubyte";
   char *filenameLabels = "./MNIST_data/train-labels.idx1-ubyte";
   init_read(filename, in);

   std::cout << in.status.n <<std::endl;

   int  i = 0;
   while( !readData(in , img) ) i++;
   std::cout << i <<std::endl;

   char labels[60000];
   std::cout << readLabels(filenameLabels, labels)<< std::endl;

   return 0;
}


