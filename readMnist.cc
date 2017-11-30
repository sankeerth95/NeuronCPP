#include <iostream>
#include "readMnist.h"

int init_read(char *filename, mnistparams& in){
   in.fp = fopen(filename, "rb");
   if(in.fp == NULL)
      return -1;
   if(fread(&(in.status), sizeof(int), 4, in.fp) == 0)
      return -1;

   char *end;
   end = (char *)&in.status.magic;
   in.status.magic = (((int)end[0])<<24)|( ((int)end[1])<<16)|( ((int)end[2])<<8)|(int)end[3];

   end = (char *)&in.status.n;
   in.status.n =  (((int)end[0])<<24)|( ((int)end[1])<<16)|( ((int)end[2])<<8 )|(int)end[3];

   end = (char *)&in.status.rows;
   in.status.rows = (((int)end[0])<<24)|( ((int)end[1])<<16)|( ((int)end[2])<<8 )|(int)end[3];

   end = (char *)&in.status.cols;
   in.status.cols = (((int)end[0])<<24)|( ((int)end[1])<<16)|( ((int)end[2])<<8 )|(int)end[3];

   return 0;
}

int readData(mnistparams& in , char* img){
//   if( fread(img, sizeof(char), in.status.rows*in.status.cols, in.fp) != 0)
   if( fread(img, sizeof(char), in.status.cols*in.status.rows, in.fp) == sizeof(char)*in.status.cols*in.status.rows)
      return 0;
   return -1;
}

int readLabels(char *filename, char* labels ){

   int magic, num;
   FILE *fp = fopen(filename, "rb");
   if(fp == NULL)
      return -1;

   if( fread(&magic, sizeof(int), 1, fp) == 0)
      return -2;

   if( fread(&num, sizeof(int), 1, fp) == 0)
      return -3;


   char *end;
   end = (char *)&magic;
   magic = (((int)end[0])<<24)|( ((int)end[1])<<16)|( ((int)end[2])<<8)|(int)end[3];

   end = (char *)&num;
   num = (((int)end[0])<<24)|( ((int)end[1])<<16)|( ((int)end[2])<<8 )|(int)end[3];

//   std::cout << num << std::endl;
   num = 60000;
   if( fread(labels, sizeof(char), num, fp) != num*sizeof(char) )
      return -4;

   fclose(fp);
   return 0;
}


