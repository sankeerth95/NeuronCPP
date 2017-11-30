#include <stdio.h>
typedef struct setparams{
   int magic;
   int n;
   int rows, cols;

} mnistread;

typedef struct mnistr{

   mnistread status;
   FILE *fp;

}mnistparams;

int init_read(char *filename, mnistparams& in);
int readData(mnistparams& in , char* img);
int readLabels(char *filename, char* labels );

